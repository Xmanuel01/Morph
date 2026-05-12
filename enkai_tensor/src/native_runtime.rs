use std::collections::HashMap;
use std::thread;
use std::time::Instant;

#[derive(Clone, Debug, PartialEq)]
pub struct NativeTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl NativeTensor {
    pub fn zeros(shape: &[usize]) -> Result<Self, String> {
        let len = checked_len(shape)?;
        Ok(Self {
            shape: shape.to_vec(),
            data: vec![0.0; len],
        })
    }

    pub fn from_vec(shape: &[usize], data: Vec<f32>) -> Result<Self, String> {
        let expected = checked_len(shape)?;
        if expected != data.len() {
            return Err(format!(
                "tensor shape/data mismatch: expected {expected}, got {}",
                data.len()
            ));
        }
        Ok(Self {
            shape: shape.to_vec(),
            data,
        })
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn rows_cols(&self) -> Result<(usize, usize), String> {
        match self.shape.as_slice() {
            [r, c] => Ok((*r, *c)),
            _ => Err(format!(
                "expected rank-2 tensor, got shape {:?}",
                self.shape
            )),
        }
    }

    pub fn bytes(&self) -> u64 {
        (self.data.len() * std::mem::size_of::<f32>()) as u64
    }
}

fn checked_len(shape: &[usize]) -> Result<usize, String> {
    if shape.is_empty() {
        return Err("tensor shape must not be empty".to_string());
    }
    shape.iter().try_fold(1usize, |acc, dim| {
        if *dim == 0 {
            return Err("tensor dimensions must be > 0".to_string());
        }
        acc.checked_mul(*dim)
            .ok_or_else(|| "tensor shape overflows usize".to_string())
    })
}

fn default_parallelism() -> usize {
    thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 16)
}

#[derive(Clone, Debug, Default)]
pub struct MemoryStats {
    pub peak_bytes: u64,
    pub allocated_bytes: u64,
    pub freed_bytes: u64,
    pub live_bytes: u64,
    pub reuse_count: u64,
}

#[derive(Clone, Debug, Default)]
pub struct MemoryPlanner {
    stats: MemoryStats,
    free: HashMap<usize, Vec<Vec<f32>>>,
}

impl MemoryPlanner {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn stats(&self) -> MemoryStats {
        self.stats.clone()
    }

    pub fn allocate(&mut self, shape: &[usize]) -> Result<NativeTensor, String> {
        let len = checked_len(shape)?;
        let bytes = (len * std::mem::size_of::<f32>()) as u64;
        let data = match self.free.get_mut(&len).and_then(|list| list.pop()) {
            Some(mut buf) => {
                self.stats.reuse_count += 1;
                buf.fill(0.0);
                buf
            }
            None => {
                self.stats.allocated_bytes += bytes;
                vec![0.0; len]
            }
        };
        self.stats.live_bytes = self.stats.live_bytes.saturating_add(bytes);
        self.stats.peak_bytes = self.stats.peak_bytes.max(self.stats.live_bytes);
        Ok(NativeTensor {
            shape: shape.to_vec(),
            data,
        })
    }

    pub fn release(&mut self, tensor: NativeTensor) {
        let len = tensor.data.len();
        let bytes = (len * std::mem::size_of::<f32>()) as u64;
        self.stats.live_bytes = self.stats.live_bytes.saturating_sub(bytes);
        self.stats.freed_bytes = self.stats.freed_bytes.saturating_add(bytes);
        self.free.entry(len).or_default().push(tensor.data);
    }
}

pub trait ExecutionBackend {
    fn name(&self) -> &'static str;
    fn zeros(&mut self, shape: &[usize]) -> Result<NativeTensor, String>;
    fn add(&mut self, a: &NativeTensor, b: &NativeTensor) -> Result<NativeTensor, String>;
    fn mul(&mut self, a: &NativeTensor, b: &NativeTensor) -> Result<NativeTensor, String>;
    fn matmul(&mut self, a: &NativeTensor, b: &NativeTensor) -> Result<NativeTensor, String>;
    fn relu(&mut self, x: &NativeTensor) -> Result<NativeTensor, String>;
    fn softmax(&mut self, x: &NativeTensor) -> Result<NativeTensor, String>;
    fn cross_entropy(&mut self, logits: &NativeTensor, targets: &[usize]) -> Result<f32, String>;
    fn mean(&mut self, x: &NativeTensor) -> Result<f32, String>;
    fn sum(&mut self, x: &NativeTensor) -> Result<f32, String>;
    fn release(&mut self, _tensor: NativeTensor) {}
    fn memory_stats(&self) -> MemoryStats;
}

#[derive(Clone, Debug, Default)]
pub struct CpuBackend {
    planner: MemoryPlanner,
    parallelism: usize,
}
impl CpuBackend {
    pub fn new() -> Self {
        Self {
            planner: MemoryPlanner::new(),
            parallelism: default_parallelism(),
        }
    }
    pub fn with_parallelism(parallelism: usize) -> Self {
        Self {
            planner: MemoryPlanner::new(),
            parallelism: parallelism.max(1),
        }
    }
    pub fn parallelism(&self) -> usize {
        self.parallelism
    }
    pub fn release(&mut self, tensor: NativeTensor) {
        self.planner.release(tensor);
    }
}

impl ExecutionBackend for CpuBackend {
    fn name(&self) -> &'static str {
        "enkai_native_cpu"
    }
    fn zeros(&mut self, shape: &[usize]) -> Result<NativeTensor, String> {
        self.planner.allocate(shape)
    }

    fn add(&mut self, a: &NativeTensor, b: &NativeTensor) -> Result<NativeTensor, String> {
        same_shape(a, b, "add")?;
        let mut out = self.planner.allocate(&a.shape)?;
        for i in 0..a.len() {
            out.data[i] = a.data[i] + b.data[i];
        }
        Ok(out)
    }

    fn mul(&mut self, a: &NativeTensor, b: &NativeTensor) -> Result<NativeTensor, String> {
        same_shape(a, b, "multiply")?;
        let mut out = self.planner.allocate(&a.shape)?;
        for i in 0..a.len() {
            out.data[i] = a.data[i] * b.data[i];
        }
        Ok(out)
    }

    fn matmul(&mut self, a: &NativeTensor, b: &NativeTensor) -> Result<NativeTensor, String> {
        let (m, k) = a.rows_cols()?;
        let (k2, n) = b.rows_cols()?;
        if k != k2 {
            return Err(format!("matmul inner mismatch: {k} != {k2}"));
        }
        let mut out = self.planner.allocate(&[m, n])?;
        matmul_parallel_into(a, b, &mut out, self.parallelism)?;
        Ok(out)
    }

    fn relu(&mut self, x: &NativeTensor) -> Result<NativeTensor, String> {
        let mut out = self.planner.allocate(&x.shape)?;
        for i in 0..x.len() {
            out.data[i] = x.data[i].max(0.0);
        }
        Ok(out)
    }

    fn softmax(&mut self, x: &NativeTensor) -> Result<NativeTensor, String> {
        let (rows, cols) = x.rows_cols()?;
        let mut out = self.planner.allocate(&x.shape)?;
        softmax_rows_into(x, &mut out, rows, cols)?;
        Ok(out)
    }

    fn cross_entropy(&mut self, logits: &NativeTensor, targets: &[usize]) -> Result<f32, String> {
        cross_entropy_from_logits(logits, targets)
    }
    fn mean(&mut self, x: &NativeTensor) -> Result<f32, String> {
        Ok(self.sum(x)? / x.len() as f32)
    }
    fn sum(&mut self, x: &NativeTensor) -> Result<f32, String> {
        Ok(x.data.iter().copied().sum())
    }
    fn release(&mut self, tensor: NativeTensor) {
        self.planner.release(tensor);
    }
    fn memory_stats(&self) -> MemoryStats {
        self.planner.stats()
    }
}

#[derive(Clone, Debug)]
pub struct CudaBackendHook {
    pub backend: &'static str,
    pub available: bool,
    pub reason: &'static str,
}

impl Default for CudaBackendHook {
    fn default() -> Self {
        Self {
            backend: "enkai_native_cuda_hook",
            available: false,
            reason: "CUDA/Triton backend hook reserved; no PyTorch execution dependency",
        }
    }
}

impl CudaBackendHook {
    pub fn dispatch_error(&self) -> String {
        format!("{} unavailable: {}", self.backend, self.reason)
    }
}

#[derive(Clone, Debug, Default)]
pub struct CudaNativeBackend {
    stats: MemoryStats,
}

impl CudaNativeBackend {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn available() -> bool {
        crate::cuda_kernels::cuda_device_count()
            .map(|count| count > 0)
            .unwrap_or(false)
    }
}

impl ExecutionBackend for CudaNativeBackend {
    fn name(&self) -> &'static str {
        "enkai_native_cuda"
    }

    fn zeros(&mut self, shape: &[usize]) -> Result<NativeTensor, String> {
        let tensor = NativeTensor::zeros(shape)?;
        self.stats.allocated_bytes = self.stats.allocated_bytes.saturating_add(tensor.bytes());
        self.stats.live_bytes = self.stats.live_bytes.saturating_add(tensor.bytes());
        self.stats.peak_bytes = self.stats.peak_bytes.max(self.stats.live_bytes);
        Ok(tensor)
    }

    fn add(&mut self, a: &NativeTensor, b: &NativeTensor) -> Result<NativeTensor, String> {
        same_shape(a, b, "cuda add")?;
        let data = crate::cuda_kernels::cuda_vec_add_f32_host(&a.data, &b.data)?;
        self.track_alloc((data.len() * std::mem::size_of::<f32>()) as u64);
        NativeTensor::from_vec(&a.shape, data)
    }

    fn mul(&mut self, a: &NativeTensor, b: &NativeTensor) -> Result<NativeTensor, String> {
        same_shape(a, b, "cuda multiply")?;
        let data = crate::cuda_kernels::cuda_vec_mul_f32_host(&a.data, &b.data)?;
        self.track_alloc((data.len() * std::mem::size_of::<f32>()) as u64);
        NativeTensor::from_vec(&a.shape, data)
    }

    fn matmul(&mut self, a: &NativeTensor, b: &NativeTensor) -> Result<NativeTensor, String> {
        let (m, k) = a.rows_cols()?;
        let (k2, n) = b.rows_cols()?;
        if k != k2 {
            return Err(format!("cuda matmul inner mismatch: {k} != {k2}"));
        }
        let data = crate::cuda_kernels::cuda_matmul_bias_f32_host(&a.data, &b.data, None, m, n, k)?;
        self.track_alloc((data.len() * std::mem::size_of::<f32>()) as u64);
        NativeTensor::from_vec(&[m, n], data)
    }

    fn relu(&mut self, x: &NativeTensor) -> Result<NativeTensor, String> {
        let data = x.data.iter().map(|v| v.max(0.0)).collect::<Vec<_>>();
        self.track_alloc((data.len() * std::mem::size_of::<f32>()) as u64);
        NativeTensor::from_vec(&x.shape, data)
    }

    fn softmax(&mut self, x: &NativeTensor) -> Result<NativeTensor, String> {
        let (rows, cols) = x.rows_cols()?;
        let data = crate::cuda_kernels::cuda_softmax_f32_host(&x.data, rows, cols)?;
        self.track_alloc((data.len() * std::mem::size_of::<f32>()) as u64);
        NativeTensor::from_vec(&x.shape, data)
    }

    fn cross_entropy(&mut self, logits: &NativeTensor, targets: &[usize]) -> Result<f32, String> {
        let (rows, cols) = logits.rows_cols()?;
        let targets_i64 = targets.iter().map(|v| *v as i64).collect::<Vec<_>>();
        let losses = crate::cuda_kernels::cuda_cross_entropy_losses_f32_host(
            &logits.data,
            &targets_i64,
            rows,
            cols,
        )?;
        if losses.iter().any(|v| !v.is_finite()) {
            return Err("E_CUDA_NUMERIC: cross entropy produced non-finite loss".to_string());
        }
        Ok(losses.iter().copied().sum::<f32>() / rows as f32)
    }

    fn mean(&mut self, x: &NativeTensor) -> Result<f32, String> {
        Ok(self.sum(x)? / x.len() as f32)
    }

    fn sum(&mut self, x: &NativeTensor) -> Result<f32, String> {
        Ok(x.data.iter().copied().sum())
    }

    fn release(&mut self, tensor: NativeTensor) {
        let bytes = tensor.bytes();
        self.stats.live_bytes = self.stats.live_bytes.saturating_sub(bytes);
        self.stats.freed_bytes = self.stats.freed_bytes.saturating_add(bytes);
    }

    fn memory_stats(&self) -> MemoryStats {
        self.stats.clone()
    }
}

impl CudaNativeBackend {
    fn track_alloc(&mut self, bytes: u64) {
        self.stats.allocated_bytes = self.stats.allocated_bytes.saturating_add(bytes);
        self.stats.live_bytes = self.stats.live_bytes.saturating_add(bytes);
        self.stats.peak_bytes = self.stats.peak_bytes.max(self.stats.live_bytes);
    }
}

#[derive(Clone, Debug)]
pub enum GraphOp {
    Input(NativeTensor),
    Zeros(Vec<usize>),
    Add(usize, usize),
    Multiply(usize, usize),
    Matmul(usize, usize),
    Relu(usize),
    Softmax(usize),
    Mean(usize),
    Sum(usize),
    CrossEntropy { logits: usize, targets: Vec<usize> },
}

#[derive(Clone, Debug, Default)]
pub struct TensorGraph {
    nodes: Vec<GraphOp>,
}

#[derive(Clone, Debug)]
pub enum GraphValue {
    Tensor(NativeTensor),
    Scalar(f32),
}

impl TensorGraph {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn push(&mut self, op: GraphOp) -> usize {
        let id = self.nodes.len();
        self.nodes.push(op);
        id
    }
    pub fn nodes(&self) -> &[GraphOp] {
        &self.nodes
    }

    pub fn execute<B: ExecutionBackend>(&self, backend: &mut B) -> Result<Vec<GraphValue>, String> {
        let mut values = Vec::with_capacity(self.nodes.len());
        for op in &self.nodes {
            let value = match op {
                GraphOp::Input(t) => GraphValue::Tensor(t.clone()),
                GraphOp::Zeros(shape) => GraphValue::Tensor(backend.zeros(shape)?),
                GraphOp::Add(a, b) => {
                    GraphValue::Tensor(backend.add(tensor(&values, *a)?, tensor(&values, *b)?)?)
                }
                GraphOp::Multiply(a, b) => {
                    GraphValue::Tensor(backend.mul(tensor(&values, *a)?, tensor(&values, *b)?)?)
                }
                GraphOp::Matmul(a, b) => {
                    GraphValue::Tensor(backend.matmul(tensor(&values, *a)?, tensor(&values, *b)?)?)
                }
                GraphOp::Relu(x) => GraphValue::Tensor(backend.relu(tensor(&values, *x)?)?),
                GraphOp::Softmax(x) => GraphValue::Tensor(backend.softmax(tensor(&values, *x)?)?),
                GraphOp::Mean(x) => GraphValue::Scalar(backend.mean(tensor(&values, *x)?)?),
                GraphOp::Sum(x) => GraphValue::Scalar(backend.sum(tensor(&values, *x)?)?),
                GraphOp::CrossEntropy { logits, targets } => {
                    GraphValue::Scalar(backend.cross_entropy(tensor(&values, *logits)?, targets)?)
                }
            };
            values.push(value);
        }
        Ok(values)
    }
}

fn tensor(values: &[GraphValue], idx: usize) -> Result<&NativeTensor, String> {
    match values.get(idx) {
        Some(GraphValue::Tensor(t)) => Ok(t),
        Some(GraphValue::Scalar(_)) => Err(format!("graph node {idx} is scalar, expected tensor")),
        None => Err(format!("graph node {idx} missing")),
    }
}

pub fn fused_add_relu<B: ExecutionBackend>(
    backend: &mut B,
    a: &NativeTensor,
    b: &NativeTensor,
) -> Result<NativeTensor, String> {
    same_shape(a, b, "add_relu")?;
    let mut out = backend.zeros(&a.shape)?;
    for i in 0..a.len() {
        out.data[i] = (a.data[i] + b.data[i]).max(0.0);
    }
    Ok(out)
}

pub fn fused_matmul_bias<B: ExecutionBackend>(
    backend: &mut B,
    a: &NativeTensor,
    b: &NativeTensor,
    bias: &NativeTensor,
) -> Result<NativeTensor, String> {
    let (m, _k) = a.rows_cols()?;
    let (_k2, n) = b.rows_cols()?;
    if bias.shape.as_slice() != [n] {
        return Err(format!(
            "matmul_bias expected bias shape [{n}], got {:?}",
            bias.shape
        ));
    }
    let mut out = backend.matmul(a, b)?;
    for row in 0..m {
        for col in 0..n {
            out.data[row * n + col] += bias.data[col];
        }
    }
    Ok(out)
}

pub fn fused_matmul_bias_relu<B: ExecutionBackend>(
    backend: &mut B,
    a: &NativeTensor,
    b: &NativeTensor,
    bias: &NativeTensor,
) -> Result<NativeTensor, String> {
    let (m, k) = a.rows_cols()?;
    let (k2, n) = b.rows_cols()?;
    if k != k2 {
        return Err(format!("matmul inner mismatch: {k} != {k2}"));
    }
    if bias.shape.as_slice() != [n] {
        return Err(format!(
            "matmul_bias_relu expected bias shape [{n}], got {:?}",
            bias.shape
        ));
    }
    let mut out = backend.zeros(&[m, n])?;
    for row in 0..m {
        let out_row = row * n;
        for col in 0..n {
            out.data[out_row + col] = bias.data[col];
        }
        for kk in 0..k {
            let aik = a.data[row * k + kk];
            let b_row = kk * n;
            for col in 0..n {
                out.data[out_row + col] += aik * b.data[b_row + col];
            }
        }
        for col in 0..n {
            out.data[out_row + col] = out.data[out_row + col].max(0.0);
        }
    }
    Ok(out)
}

pub fn fused_matmul_bias_relu_cpu(
    backend: &mut CpuBackend,
    a: &NativeTensor,
    b: &NativeTensor,
    bias: &NativeTensor,
) -> Result<NativeTensor, String> {
    let (m, k) = a.rows_cols()?;
    let (k2, n) = b.rows_cols()?;
    if k != k2 {
        return Err(format!("matmul inner mismatch: {k} != {k2}"));
    }
    if bias.shape.as_slice() != [n] {
        return Err(format!(
            "matmul_bias_relu expected bias shape [{n}], got {:?}",
            bias.shape
        ));
    }
    let mut out = backend.zeros(&[m, n])?;
    matmul_bias_relu_parallel_into(a, b, bias, &mut out, backend.parallelism())?;
    Ok(out)
}

pub fn fused_softmax_cross_entropy(
    logits: &NativeTensor,
    targets: &[usize],
) -> Result<f32, String> {
    cross_entropy_from_logits(logits, targets)
}

pub fn unfused_softmax_cross_entropy<B: ExecutionBackend>(
    backend: &mut B,
    logits: &NativeTensor,
    targets: &[usize],
) -> Result<f32, String> {
    let probs = backend.softmax(logits)?;
    let (rows, cols) = probs.rows_cols()?;
    if targets.len() != rows {
        return Err("target length mismatch".to_string());
    }
    let mut loss = 0.0;
    for row in 0..rows {
        let target = targets[row];
        if target >= cols {
            return Err(format!("target class {target} out of range {cols}"));
        }
        loss -= probs.data[row * cols + target].max(1e-20).ln();
    }
    Ok(loss / rows as f32)
}

#[derive(Clone, Debug)]
pub struct MlpTrainingReport {
    pub loss_initial: f32,
    pub loss_final: f32,
    pub steps: usize,
    pub peak_memory_bytes: u64,
    pub allocated_bytes: u64,
    pub reuse_count: u64,
}

#[derive(Clone, Debug)]
pub struct AdamWConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            lr: 1e-2,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct AdamWState {
    pub step: usize,
    pub m: Vec<NativeTensor>,
    pub v: Vec<NativeTensor>,
}

pub fn train_mlp_sgd(steps: usize, lr: f32) -> Result<MlpTrainingReport, String> {
    let x = NativeTensor::from_vec(&[4, 2], vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])?;
    let y = vec![0usize, 1, 1, 0];
    let mut w1 = NativeTensor::from_vec(
        &[2, 4],
        vec![0.10, -0.20, 0.30, 0.25, -0.15, 0.35, -0.25, 0.20],
    )?;
    let mut b1 = NativeTensor::from_vec(&[4], vec![0.0; 4])?;
    let mut w2 = NativeTensor::from_vec(
        &[4, 2],
        vec![0.25, -0.10, -0.30, 0.20, 0.15, 0.35, -0.20, -0.25],
    )?;
    let mut b2 = NativeTensor::from_vec(&[2], vec![0.0; 2])?;
    let mut backend = CpuBackend::new();
    let mut loss_initial = 0.0;
    let mut loss_final = 0.0;

    for step in 0..steps {
        let (loss, grads) = mlp_loss_and_grads(&mut backend, &x, &y, &w1, &b1, &w2, &b2)?;
        if step == 0 {
            loss_initial = loss;
        }
        loss_final = loss;
        sgd_update(&mut w1, &grads.dw1, lr);
        sgd_update(&mut b1, &grads.db1, lr);
        sgd_update(&mut w2, &grads.dw2, lr);
        sgd_update(&mut b2, &grads.db2, lr);
    }
    let stats = backend.memory_stats();
    Ok(MlpTrainingReport {
        loss_initial,
        loss_final,
        steps,
        peak_memory_bytes: stats.peak_bytes,
        allocated_bytes: stats.allocated_bytes,
        reuse_count: stats.reuse_count,
    })
}

pub fn train_mlp_adamw(steps: usize, config: AdamWConfig) -> Result<MlpTrainingReport, String> {
    let x = NativeTensor::from_vec(&[4, 2], vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])?;
    let y = vec![0usize, 1, 1, 0];
    let mut w1 = NativeTensor::from_vec(
        &[2, 4],
        vec![0.10, -0.20, 0.30, 0.25, -0.15, 0.35, -0.25, 0.20],
    )?;
    let mut b1 = NativeTensor::from_vec(&[4], vec![0.0; 4])?;
    let mut w2 = NativeTensor::from_vec(
        &[4, 2],
        vec![0.25, -0.10, -0.30, 0.20, 0.15, 0.35, -0.20, -0.25],
    )?;
    let mut b2 = NativeTensor::from_vec(&[2], vec![0.0; 2])?;
    let mut state = AdamWState::default();
    let mut backend = CpuBackend::new();
    let mut loss_initial = 0.0;
    let mut loss_final = 0.0;

    for step in 0..steps {
        let (loss, grads) = mlp_loss_and_grads(&mut backend, &x, &y, &w1, &b1, &w2, &b2)?;
        if step == 0 {
            loss_initial = loss;
        }
        loss_final = loss;
        adamw_step_multi(
            [&mut w1, &mut b1, &mut w2, &mut b2],
            [&grads.dw1, &grads.db1, &grads.dw2, &grads.db2],
            &mut state,
            &config,
        )?;
    }
    let stats = backend.memory_stats();
    Ok(MlpTrainingReport {
        loss_initial,
        loss_final,
        steps,
        peak_memory_bytes: stats.peak_bytes,
        allocated_bytes: stats.allocated_bytes,
        reuse_count: stats.reuse_count,
    })
}

pub fn adamw_step_multi<const N: usize>(
    params: [&mut NativeTensor; N],
    grads: [&NativeTensor; N],
    state: &mut AdamWState,
    config: &AdamWConfig,
) -> Result<(), String> {
    if !(config.lr.is_finite()
        && config.beta1.is_finite()
        && config.beta2.is_finite()
        && config.eps.is_finite()
        && config.weight_decay.is_finite())
    {
        return Err("AdamW config contains non-finite value".to_string());
    }
    if config.lr <= 0.0
        || !(0.0..1.0).contains(&config.beta1)
        || !(0.0..1.0).contains(&config.beta2)
        || config.eps <= 0.0
    {
        return Err("AdamW config out of range".to_string());
    }
    if state.m.is_empty() && state.v.is_empty() {
        state.m = params
            .iter()
            .map(|p| NativeTensor::zeros(&p.shape))
            .collect::<Result<Vec<_>, _>>()?;
        state.v = params
            .iter()
            .map(|p| NativeTensor::zeros(&p.shape))
            .collect::<Result<Vec<_>, _>>()?;
    }
    if state.m.len() != N || state.v.len() != N {
        return Err(format!(
            "AdamW state parameter count mismatch: expected {N}, got {}/{}",
            state.m.len(),
            state.v.len()
        ));
    }
    state.step += 1;
    let beta1_corr = 1.0 - config.beta1.powi(state.step as i32);
    let beta2_corr = 1.0 - config.beta2.powi(state.step as i32);
    for idx in 0..N {
        adamw_update_param(
            params[idx],
            grads[idx],
            &mut state.m[idx],
            &mut state.v[idx],
            config,
            beta1_corr,
            beta2_corr,
        )?;
    }
    Ok(())
}

fn adamw_update_param(
    param: &mut NativeTensor,
    grad: &NativeTensor,
    m: &mut NativeTensor,
    v: &mut NativeTensor,
    config: &AdamWConfig,
    beta1_corr: f32,
    beta2_corr: f32,
) -> Result<(), String> {
    same_shape(param, grad, "adamw param/grad")?;
    same_shape(param, m, "adamw param/m")?;
    same_shape(param, v, "adamw param/v")?;
    for i in 0..param.len() {
        let g = grad.data[i];
        if !g.is_finite() {
            return Err(format!("AdamW gradient at index {i} is non-finite"));
        }
        m.data[i] = config.beta1 * m.data[i] + (1.0 - config.beta1) * g;
        v.data[i] = config.beta2 * v.data[i] + (1.0 - config.beta2) * g * g;
        let m_hat = m.data[i] / beta1_corr.max(f32::MIN_POSITIVE);
        let v_hat = v.data[i] / beta2_corr.max(f32::MIN_POSITIVE);
        if config.weight_decay != 0.0 {
            param.data[i] -= config.lr * config.weight_decay * param.data[i];
        }
        param.data[i] -= config.lr * m_hat / (v_hat.sqrt() + config.eps);
    }
    Ok(())
}

pub fn mlp_forward_loss<B: ExecutionBackend>(backend: &mut B) -> Result<f32, String> {
    let x = NativeTensor::from_vec(&[4, 2], vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])?;
    let y = vec![0usize, 1, 1, 0];
    let w1 = NativeTensor::from_vec(
        &[2, 4],
        vec![0.10, -0.20, 0.30, 0.25, -0.15, 0.35, -0.25, 0.20],
    )?;
    let b1 = NativeTensor::from_vec(&[4], vec![0.0; 4])?;
    let w2 = NativeTensor::from_vec(
        &[4, 2],
        vec![0.25, -0.10, -0.30, 0.20, 0.15, 0.35, -0.20, -0.25],
    )?;
    let b2 = NativeTensor::from_vec(&[2], vec![0.0; 2])?;
    let z1 = fused_matmul_bias(backend, &x, &w1, &b1)?;
    let h = backend.relu(&z1)?;
    let logits = fused_matmul_bias(backend, &h, &w2, &b2)?;
    cross_entropy_from_logits(&logits, &y)
}

struct MlpGrads {
    dw1: NativeTensor,
    db1: NativeTensor,
    dw2: NativeTensor,
    db2: NativeTensor,
}

fn mlp_loss_and_grads<B: ExecutionBackend>(
    backend: &mut B,
    x: &NativeTensor,
    targets: &[usize],
    w1: &NativeTensor,
    b1: &NativeTensor,
    w2: &NativeTensor,
    b2: &NativeTensor,
) -> Result<(f32, MlpGrads), String> {
    let (batch, input_dim) = x.rows_cols()?;
    let (_, hidden) = w1.rows_cols()?;
    let (_, classes) = w2.rows_cols()?;
    let z1_linear = fused_matmul_bias(backend, x, w1, b1)?;
    let h = backend.relu(&z1_linear)?;
    let logits = fused_matmul_bias(backend, &h, w2, b2)?;
    let loss = cross_entropy_from_logits(&logits, targets)?;
    let mut probs = backend.softmax(&logits)?;
    for row in 0..batch {
        probs.data[row * classes + targets[row]] -= 1.0;
    }
    for v in &mut probs.data {
        *v /= batch as f32;
    }

    let mut dw2 = NativeTensor::zeros(&[hidden, classes])?;
    for i in 0..hidden {
        for c in 0..classes {
            let mut s = 0.0;
            for r in 0..batch {
                s += h.data[r * hidden + i] * probs.data[r * classes + c];
            }
            dw2.data[i * classes + c] = s;
        }
    }
    let mut db2 = NativeTensor::zeros(&[classes])?;
    for c in 0..classes {
        db2.data[c] = (0..batch).map(|r| probs.data[r * classes + c]).sum();
    }

    let mut dh = NativeTensor::zeros(&[batch, hidden])?;
    for r in 0..batch {
        for i in 0..hidden {
            let mut s = 0.0;
            for c in 0..classes {
                s += probs.data[r * classes + c] * w2.data[i * classes + c];
            }
            dh.data[r * hidden + i] = if z1_linear.data[r * hidden + i] > 0.0 {
                s
            } else {
                0.0
            };
        }
    }

    let mut dw1 = NativeTensor::zeros(&[input_dim, hidden])?;
    for i in 0..input_dim {
        for hidx in 0..hidden {
            let mut s = 0.0;
            for r in 0..batch {
                s += x.data[r * input_dim + i] * dh.data[r * hidden + hidx];
            }
            dw1.data[i * hidden + hidx] = s;
        }
    }
    let mut db1 = NativeTensor::zeros(&[hidden])?;
    for hidx in 0..hidden {
        db1.data[hidx] = (0..batch).map(|r| dh.data[r * hidden + hidx]).sum();
    }
    backend.release(z1_linear);
    backend.release(h);
    backend.release(logits);
    backend.release(probs);
    backend.release(dh);
    Ok((loss, MlpGrads { dw1, db1, dw2, db2 }))
}

fn sgd_update(param: &mut NativeTensor, grad: &NativeTensor, lr: f32) {
    for i in 0..param.len() {
        param.data[i] -= lr * grad.data[i];
    }
}

pub fn benchmark_native_runtime(iterations: usize) -> Result<serde_json::Value, String> {
    let n = 128usize;
    let a = NativeTensor::from_vec(
        &[n, n],
        (0..n * n).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect(),
    )?;
    let b = NativeTensor::from_vec(
        &[n, n],
        (0..n * n).map(|i| ((i % 19) as f32 - 9.0) * 0.01).collect(),
    )?;
    let bias = NativeTensor::from_vec(&[n], (0..n).map(|i| (i as f32) * 0.001).collect())?;
    let targets: Vec<usize> = (0..n).map(|i| i % n).collect();

    let mut vector_backend = CpuBackend::new();
    let va = NativeTensor::from_vec(
        &[n * n],
        (0..n * n).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect(),
    )?;
    let vb = NativeTensor::from_vec(
        &[n * n],
        (0..n * n).map(|i| ((i % 19) as f32 - 9.0) * 0.01).collect(),
    )?;
    let started = Instant::now();
    let mut vector_checksum = 0.0;
    for _ in 0..iterations {
        let out = vector_backend.add(&va, &vb)?;
        vector_checksum += vector_backend.sum(&out)?;
        vector_backend.release(out);
    }
    let vector_add_ms = started.elapsed().as_secs_f64() * 1000.0;

    let mut matmul_backend = CpuBackend::new();
    let started = Instant::now();
    let mut matmul_checksum = 0.0;
    for _ in 0..iterations {
        let out = matmul_backend.matmul(&a, &b)?;
        matmul_checksum += matmul_backend.sum(&out)?;
        matmul_backend.release(out);
    }
    let matmul_ms = started.elapsed().as_secs_f64() * 1000.0;

    let mut mlp_forward_backend = CpuBackend::new();
    let started = Instant::now();
    let mut mlp_forward_loss_sum = 0.0;
    for _ in 0..iterations {
        mlp_forward_loss_sum += mlp_forward_loss(&mut mlp_forward_backend)?;
    }
    let mlp_forward_ms = started.elapsed().as_secs_f64() * 1000.0;

    let started = Instant::now();
    let training = train_mlp_sgd(200, 0.4)?;
    let training_ms = started.elapsed().as_secs_f64() * 1000.0;

    let adamw_training = train_mlp_adamw(
        200,
        AdamWConfig {
            lr: 0.05,
            ..AdamWConfig::default()
        },
    )?;

    let mut stress_backend = CpuBackend::new();
    let started = Instant::now();
    let mut stress_checksum = 0.0;
    for _ in 0..iterations.max(1) {
        let mut buffers = Vec::new();
        for i in 0..64 {
            let mut t = stress_backend.zeros(&[256, 64])?;
            t.data[i] = i as f32;
            stress_checksum += t.data[i];
            buffers.push(t);
        }
        for buffer in buffers {
            stress_backend.release(buffer);
        }
    }
    let memory_stress_ms = started.elapsed().as_secs_f64() * 1000.0;

    let mut backend = CpuBackend::new();
    let started = Instant::now();
    let mut checksum = 0.0;
    for _ in 0..iterations {
        let mm = backend.matmul(&a, &b)?;
        let biased = backend.add(&mm, &broadcast_bias(&bias, n)?)?;
        let relu = backend.relu(&biased)?;
        checksum += backend.sum(&relu)?;
        backend.release(mm);
        backend.release(biased);
        backend.release(relu);
    }
    let unfused_ms = started.elapsed().as_secs_f64() * 1000.0;

    let mut fused_backend = CpuBackend::new();
    let started = Instant::now();
    let mut fused_checksum = 0.0;
    for _ in 0..iterations {
        let relu = fused_matmul_bias_relu_cpu(&mut fused_backend, &a, &b, &bias)?;
        fused_checksum += fused_backend.sum(&relu)?;
        fused_backend.release(relu);
    }
    let fused_ms = started.elapsed().as_secs_f64() * 1000.0;

    let mut ce_backend = CpuBackend::new();
    let started = Instant::now();
    let mut ce = 0.0;
    for _ in 0..iterations {
        ce += unfused_softmax_cross_entropy(&mut ce_backend, &a, &targets)?;
    }
    let ce_unfused_ms = started.elapsed().as_secs_f64() * 1000.0;
    let started = Instant::now();
    let mut ce_fused = 0.0;
    for _ in 0..iterations {
        ce_fused += fused_softmax_cross_entropy(&a, &targets)?;
    }
    let ce_fused_ms = started.elapsed().as_secs_f64() * 1000.0;

    Ok(serde_json::json!({
        "schema_version": 1,
        "runtime": "enkai_native_training_runtime_cpu",
        "iterations": iterations,
        "cuda_backend_hook": CudaBackendHook::default().dispatch_error(),
        "benchmarks": {
            "vector_add": {"elements": va.len(), "elapsed_ms": vector_add_ms, "checksum": vector_checksum, "memory": memory_stats_json(&vector_backend.memory_stats())},
            "matmul": {"shape": [n, n], "elapsed_ms": matmul_ms, "checksum": matmul_checksum, "memory": memory_stats_json(&matmul_backend.memory_stats())},
            "mlp_forward": {"elapsed_ms": mlp_forward_ms, "loss_sum": mlp_forward_loss_sum, "memory": memory_stats_json(&mlp_forward_backend.memory_stats())},
            "mlp_training_step": {"optimizer": "sgd", "elapsed_ms": training_ms, "steps": training.steps, "loss_initial": training.loss_initial, "loss_final": training.loss_final, "peak_memory_bytes": training.peak_memory_bytes},
            "mlp_training_adamw": {"optimizer": "adamw", "steps": adamw_training.steps, "loss_initial": adamw_training.loss_initial, "loss_final": adamw_training.loss_final, "peak_memory_bytes": adamw_training.peak_memory_bytes, "reuse_count": adamw_training.reuse_count},
            "softmax_cross_entropy": {"unfused_ms": ce_unfused_ms, "fused_ms": ce_fused_ms},
            "memory_stress": {"elapsed_ms": memory_stress_ms, "checksum": stress_checksum, "memory": memory_stats_json(&stress_backend.memory_stats())}
        },
        "matmul_bias_relu": {"unfused_ms": unfused_ms, "fused_ms": fused_ms, "checksum_delta_abs": (checksum - fused_checksum).abs()},
        "softmax_cross_entropy": {"unfused_ms": ce_unfused_ms, "fused_ms": ce_fused_ms, "loss_delta_abs": (ce - ce_fused).abs()},
        "memory": {"unfused": memory_stats_json(&backend.memory_stats()), "fused": memory_stats_json(&fused_backend.memory_stats()), "ce_unfused": memory_stats_json(&ce_backend.memory_stats())},
        "training": {"loss_initial": training.loss_initial, "loss_final": training.loss_final, "steps": training.steps, "peak_memory_bytes": training.peak_memory_bytes, "allocated_bytes": training.allocated_bytes, "reuse_count": training.reuse_count, "adamw_loss_final": adamw_training.loss_final},
        "claims": {"pytorch_core_execution_dependency": false, "python_core_execution_dependency": false, "cuda_without_pytorch": "hook_only_not_claimed"}
    }))
}

fn broadcast_bias(bias: &NativeTensor, rows: usize) -> Result<NativeTensor, String> {
    if bias.shape.len() != 1 {
        return Err("bias must be rank-1".to_string());
    }
    let cols = bias.shape[0];
    let mut out = NativeTensor::zeros(&[rows, cols])?;
    for r in 0..rows {
        for c in 0..cols {
            out.data[r * cols + c] = bias.data[c];
        }
    }
    Ok(out)
}

fn matmul_parallel_into(
    a: &NativeTensor,
    b: &NativeTensor,
    out: &mut NativeTensor,
    requested_threads: usize,
) -> Result<(), String> {
    let (m, k) = a.rows_cols()?;
    let (k2, n) = b.rows_cols()?;
    if k != k2 || out.shape.as_slice() != [m, n] {
        return Err("matmul output shape mismatch".to_string());
    }
    let work = m.saturating_mul(n).saturating_mul(k);
    let threads = requested_threads.min(m).max(1);
    if threads == 1 || work < 1_000_000 {
        return matmul_into(a, b, out);
    }
    let rows_per_chunk = (m + threads - 1) / threads;
    thread::scope(|scope| {
        for (chunk_idx, chunk) in out.data.chunks_mut(rows_per_chunk * n).enumerate() {
            let start_row = chunk_idx * rows_per_chunk;
            scope.spawn(move || {
                let rows = chunk.len() / n;
                for local_row in 0..rows {
                    let row = start_row + local_row;
                    let out_row = local_row * n;
                    for j in 0..n {
                        chunk[out_row + j] = 0.0;
                    }
                    for kk in 0..k {
                        let aik = a.data[row * k + kk];
                        let b_row = kk * n;
                        for j in 0..n {
                            chunk[out_row + j] += aik * b.data[b_row + j];
                        }
                    }
                }
            });
        }
    });
    Ok(())
}

fn matmul_bias_relu_parallel_into(
    a: &NativeTensor,
    b: &NativeTensor,
    bias: &NativeTensor,
    out: &mut NativeTensor,
    requested_threads: usize,
) -> Result<(), String> {
    let (m, k) = a.rows_cols()?;
    let (k2, n) = b.rows_cols()?;
    if k != k2 || out.shape.as_slice() != [m, n] {
        return Err("matmul_bias_relu output shape mismatch".to_string());
    }
    if bias.shape.as_slice() != [n] {
        return Err(format!(
            "matmul_bias_relu expected bias shape [{n}], got {:?}",
            bias.shape
        ));
    }
    let work = m.saturating_mul(n).saturating_mul(k);
    let threads = requested_threads.min(m).max(1);
    if threads == 1 || work < 1_000_000 {
        for row in 0..m {
            let out_row = row * n;
            for col in 0..n {
                out.data[out_row + col] = bias.data[col];
            }
            for kk in 0..k {
                let aik = a.data[row * k + kk];
                let b_row = kk * n;
                for col in 0..n {
                    out.data[out_row + col] += aik * b.data[b_row + col];
                }
            }
            for col in 0..n {
                out.data[out_row + col] = out.data[out_row + col].max(0.0);
            }
        }
        return Ok(());
    }
    let rows_per_chunk = (m + threads - 1) / threads;
    thread::scope(|scope| {
        for (chunk_idx, chunk) in out.data.chunks_mut(rows_per_chunk * n).enumerate() {
            let start_row = chunk_idx * rows_per_chunk;
            scope.spawn(move || {
                let rows = chunk.len() / n;
                for local_row in 0..rows {
                    let row = start_row + local_row;
                    let out_row = local_row * n;
                    for col in 0..n {
                        chunk[out_row + col] = bias.data[col];
                    }
                    for kk in 0..k {
                        let aik = a.data[row * k + kk];
                        let b_row = kk * n;
                        for col in 0..n {
                            chunk[out_row + col] += aik * b.data[b_row + col];
                        }
                    }
                    for col in 0..n {
                        chunk[out_row + col] = chunk[out_row + col].max(0.0);
                    }
                }
            });
        }
    });
    Ok(())
}

fn matmul_into(a: &NativeTensor, b: &NativeTensor, out: &mut NativeTensor) -> Result<(), String> {
    let (m, k) = a.rows_cols()?;
    let (k2, n) = b.rows_cols()?;
    if k != k2 || out.shape.as_slice() != [m, n] {
        return Err("matmul output shape mismatch".to_string());
    }
    for i in 0..m {
        let out_row = i * n;
        for j in 0..n {
            out.data[out_row + j] = 0.0;
        }
        for kk in 0..k {
            let aik = a.data[i * k + kk];
            let b_row = kk * n;
            for j in 0..n {
                out.data[out_row + j] += aik * b.data[b_row + j];
            }
        }
    }
    Ok(())
}

fn softmax_rows_into(
    x: &NativeTensor,
    out: &mut NativeTensor,
    rows: usize,
    cols: usize,
) -> Result<(), String> {
    if out.shape != x.shape {
        return Err("softmax output shape mismatch".to_string());
    }
    for r in 0..rows {
        let base = r * cols;
        let max_v = x.data[base..base + cols]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut denom = 0.0;
        for c in 0..cols {
            let v = (x.data[base + c] - max_v).exp();
            out.data[base + c] = v;
            denom += v;
        }
        if !denom.is_finite() || denom <= 0.0 {
            return Err("softmax denominator invalid".to_string());
        }
        for c in 0..cols {
            out.data[base + c] /= denom;
        }
    }
    Ok(())
}

fn cross_entropy_from_logits(logits: &NativeTensor, targets: &[usize]) -> Result<f32, String> {
    let (rows, cols) = logits.rows_cols()?;
    if targets.len() != rows {
        return Err(format!(
            "target length mismatch: expected {rows}, got {}",
            targets.len()
        ));
    }
    let mut loss = 0.0;
    for (r, target) in targets.iter().copied().enumerate() {
        if target >= cols {
            return Err(format!("target class {target} out of range {cols}"));
        }
        let base = r * cols;
        let max_v = logits.data[base..base + cols]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let denom: f32 = (0..cols)
            .map(|c| (logits.data[base + c] - max_v).exp())
            .sum();
        loss += denom.max(1e-20).ln() + max_v - logits.data[base + target];
    }
    Ok(loss / rows as f32)
}

fn same_shape(a: &NativeTensor, b: &NativeTensor, op: &str) -> Result<(), String> {
    if a.shape != b.shape {
        return Err(format!(
            "{op} shape mismatch: {:?} vs {:?}",
            a.shape, b.shape
        ));
    }
    Ok(())
}

fn memory_stats_json(stats: &MemoryStats) -> serde_json::Value {
    serde_json::json!({
        "peak_bytes": stats.peak_bytes,
        "allocated_bytes": stats.allocated_bytes,
        "freed_bytes": stats.freed_bytes,
        "live_bytes": stats.live_bytes,
        "reuse_count": stats.reuse_count,
    })
}
