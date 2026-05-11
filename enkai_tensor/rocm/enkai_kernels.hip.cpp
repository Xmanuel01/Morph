#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

extern "C" {

__device__ float enkai_rocm_gelu_f32(float v) {
    float c = 0.7978845608028654f * (v + 0.044715f * v * v * v);
    return 0.5f * v * (1.0f + tanhf(c));
}

__global__ void enkai_rocm_vec_add_f32_kernel(const float* a, const float* b, float* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void enkai_rocm_vec_mul_f32_kernel(const float* a, const float* b, float* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

__global__ void enkai_rocm_vec_scale_f32_kernel(const float* x, float scale, float* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] * scale;
}

__global__ void enkai_rocm_bias_gelu_f32_kernel(const float* x, const float* bias, float* out, int64_t rows, int64_t cols) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = rows * cols;
    if (i >= n) return;
    out[i] = enkai_rocm_gelu_f32(x[i] + bias[i % cols]);
}

__global__ void enkai_rocm_matmul_bias_f32_kernel(const float* a, const float* b, const float* bias, float* out, int64_t m, int64_t n, int64_t k) {
    int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;
    float acc = bias ? bias[col] : 0.0f;
    for (int64_t i = 0; i < k; ++i) acc += a[row * k + i] * b[i * n + col];
    out[row * n + col] = acc;
}

__global__ void enkai_rocm_softmax_f32_kernel(const float* x, float* out, int64_t rows, int64_t cols) {
    int64_t row = blockIdx.x;
    if (row >= rows) return;
    const float* xr = x + row * cols;
    float* yr = out + row * cols;
    __shared__ float scratch[256];
    float max_v = -INFINITY;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) max_v = fmaxf(max_v, xr[c]);
    scratch[threadIdx.x] = max_v;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        __syncthreads();
    }
    max_v = scratch[0];
    float sum = 0.0f;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
        float e = expf(xr[c] - max_v);
        yr[c] = e;
        sum += e;
    }
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    float denom = fmaxf(scratch[0], 1.0e-20f);
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) yr[c] /= denom;
}

__global__ void enkai_rocm_cross_entropy_forward_f32_kernel(const float* logits, const int64_t* targets, float* losses, int64_t rows, int64_t cols) {
    int64_t row = blockIdx.x;
    if (row >= rows) return;
    int64_t target = targets[row];
    if (target < 0 || target >= cols) { if (threadIdx.x == 0) losses[row] = INFINITY; return; }
    const float* xr = logits + row * cols;
    __shared__ float scratch[256];
    float max_v = -INFINITY;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) max_v = fmaxf(max_v, xr[c]);
    scratch[threadIdx.x] = max_v;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) { if (threadIdx.x < stride) scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]); __syncthreads(); }
    max_v = scratch[0];
    float sum = 0.0f;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) sum += expf(xr[c] - max_v);
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) { if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride]; __syncthreads(); }
    if (threadIdx.x == 0) losses[row] = logf(fmaxf(scratch[0], 1.0e-20f)) + max_v - xr[target];
}

__global__ void enkai_rocm_cross_entropy_backward_f32_kernel(const float* logits, const int64_t* targets, float* grad, int64_t rows, int64_t cols, float scale) {
    int64_t row = blockIdx.x;
    if (row >= rows) return;
    int64_t target = targets[row];
    const float* xr = logits + row * cols;
    float* gr = grad + row * cols;
    if (target < 0 || target >= cols) { for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) gr[c] = NAN; return; }
    __shared__ float scratch[256];
    float max_v = -INFINITY;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) max_v = fmaxf(max_v, xr[c]);
    scratch[threadIdx.x] = max_v;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) { if (threadIdx.x < stride) scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]); __syncthreads(); }
    max_v = scratch[0];
    float sum = 0.0f;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) sum += expf(xr[c] - max_v);
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) { if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride]; __syncthreads(); }
    float denom = fmaxf(scratch[0], 1.0e-20f);
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) gr[c] = (expf(xr[c] - max_v) / denom - (c == target ? 1.0f : 0.0f)) * scale;
}

__global__ void enkai_rocm_embedding_forward_f32_kernel(const float* weights, const int64_t* ids, float* out, int64_t ids_len, int64_t dim, int64_t vocab) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = ids_len * dim;
    if (i >= n) return;
    int64_t row = i / dim, d = i % dim, id = ids[row];
    out[i] = (id >= 0 && id < vocab) ? weights[id * dim + d] : NAN;
}

__global__ void enkai_rocm_embedding_backward_f32_kernel(const float* grad_out, const int64_t* ids, float* grad_weights, int64_t ids_len, int64_t dim, int64_t vocab) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = ids_len * dim;
    if (i >= n) return;
    int64_t row = i / dim, d = i % dim, id = ids[row];
    if (id >= 0 && id < vocab) atomicAdd(grad_weights + id * dim + d, grad_out[i]);
}

__global__ void enkai_rocm_adamw_update_f32_kernel(float* param, const float* grad, float* m, float* v, int64_t n, float lr, float beta1, float beta2, float eps, float wd, int64_t step) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = grad[i] + wd * param[i];
    float mt = beta1 * m[i] + (1.0f - beta1) * g;
    float vt = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = mt; v[i] = vt;
    param[i] -= lr * (mt / (1.0f - powf(beta1, (float)step))) / (sqrtf(vt / (1.0f - powf(beta2, (float)step))) + eps);
}

__global__ void enkai_rocm_clip_grad_norm_f32_kernel(float* grad, int64_t n, float max_norm, float eps, float* out_norm) {
    __shared__ float scratch[256];
    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) sum += grad[i] * grad[i];
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) { if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride]; __syncthreads(); }
    float norm = sqrtf(scratch[0]);
    float coeff = fminf(max_norm / (norm + eps), 1.0f);
    if (threadIdx.x == 0 && out_norm) *out_norm = norm;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) grad[i] *= coeff;
}

#define ENKAI_ROCM_LAUNCH_1D(kernel, n, ...) do { int threads = 256; int64_t blocks = ((n) + threads - 1) / threads; hipLaunchKernelGGL(kernel, dim3((unsigned int)blocks), dim3(threads), 0, stream, __VA_ARGS__); return (int)hipGetLastError(); } while (0)

int enkai_rocm_vec_add_f32(const float* a, const float* b, float* out, int64_t n, hipStream_t stream) { if (!a || !b || !out || n < 0) return 1; ENKAI_ROCM_LAUNCH_1D(enkai_rocm_vec_add_f32_kernel, n, a, b, out, n); }
int enkai_rocm_vec_mul_f32(const float* a, const float* b, float* out, int64_t n, hipStream_t stream) { if (!a || !b || !out || n < 0) return 1; ENKAI_ROCM_LAUNCH_1D(enkai_rocm_vec_mul_f32_kernel, n, a, b, out, n); }
int enkai_rocm_vec_scale_f32(const float* x, float scale, float* out, int64_t n, hipStream_t stream) { if (!x || !out || n < 0) return 1; ENKAI_ROCM_LAUNCH_1D(enkai_rocm_vec_scale_f32_kernel, n, x, scale, out, n); }
int enkai_rocm_bias_gelu_f32(const float* x, const float* bias, float* out, int64_t rows, int64_t cols, hipStream_t stream) { if (!x || !bias || !out || rows < 0 || cols <= 0) return 1; ENKAI_ROCM_LAUNCH_1D(enkai_rocm_bias_gelu_f32_kernel, rows * cols, x, bias, out, rows, cols); }
int enkai_rocm_matmul_bias_f32(const float* a, const float* b, const float* bias, float* out, int64_t m, int64_t n, int64_t k, hipStream_t stream) { if (!a || !b || !out || m < 0 || n <= 0 || k <= 0) return 1; dim3 threads(16,16); dim3 blocks((unsigned int)((n+15)/16),(unsigned int)((m+15)/16)); hipLaunchKernelGGL(enkai_rocm_matmul_bias_f32_kernel, blocks, threads, 0, stream, a, b, bias, out, m, n, k); return (int)hipGetLastError(); }
int enkai_rocm_softmax_f32(const float* x, float* out, int64_t rows, int64_t cols, hipStream_t stream) { if (!x || !out || rows < 0 || cols <= 0 || cols > 4096) return 1; hipLaunchKernelGGL(enkai_rocm_softmax_f32_kernel, dim3((unsigned int)rows), dim3(256), 0, stream, x, out, rows, cols); return (int)hipGetLastError(); }
int enkai_rocm_cross_entropy_forward_f32(const float* logits, const int64_t* targets, float* losses, int64_t rows, int64_t cols, hipStream_t stream) { if (!logits || !targets || !losses || rows < 0 || cols <= 1 || cols > 4096) return 1; hipLaunchKernelGGL(enkai_rocm_cross_entropy_forward_f32_kernel, dim3((unsigned int)rows), dim3(256), 0, stream, logits, targets, losses, rows, cols); return (int)hipGetLastError(); }
int enkai_rocm_cross_entropy_backward_f32(const float* logits, const int64_t* targets, float* grad, int64_t rows, int64_t cols, float scale, hipStream_t stream) { if (!logits || !targets || !grad || rows < 0 || cols <= 1 || cols > 4096) return 1; hipLaunchKernelGGL(enkai_rocm_cross_entropy_backward_f32_kernel, dim3((unsigned int)rows), dim3(256), 0, stream, logits, targets, grad, rows, cols, scale); return (int)hipGetLastError(); }
int enkai_rocm_embedding_forward_f32(const float* weights, const int64_t* ids, float* out, int64_t ids_len, int64_t dim, int64_t vocab, hipStream_t stream) { if (!weights || !ids || !out || ids_len < 0 || dim <= 0 || vocab <= 0) return 1; ENKAI_ROCM_LAUNCH_1D(enkai_rocm_embedding_forward_f32_kernel, ids_len * dim, weights, ids, out, ids_len, dim, vocab); }
int enkai_rocm_embedding_backward_f32(const float* grad_out, const int64_t* ids, float* grad_weights, int64_t ids_len, int64_t dim, int64_t vocab, hipStream_t stream) { if (!grad_out || !ids || !grad_weights || ids_len < 0 || dim <= 0 || vocab <= 0) return 1; ENKAI_ROCM_LAUNCH_1D(enkai_rocm_embedding_backward_f32_kernel, ids_len * dim, grad_out, ids, grad_weights, ids_len, dim, vocab); }
int enkai_rocm_adamw_update_f32(float* param, const float* grad, float* m, float* v, int64_t n, float lr, float beta1, float beta2, float eps, float wd, int64_t step, hipStream_t stream) { if (!param || !grad || !m || !v || n < 0 || step <= 0) return 1; ENKAI_ROCM_LAUNCH_1D(enkai_rocm_adamw_update_f32_kernel, n, param, grad, m, v, n, lr, beta1, beta2, eps, wd, step); }
int enkai_rocm_clip_grad_norm_f32(float* grad, int64_t n, float max_norm, float eps, float* out_norm, hipStream_t stream) { if (!grad || n < 0 || max_norm < 0.0f || eps <= 0.0f) return 1; hipLaunchKernelGGL(enkai_rocm_clip_grad_norm_f32_kernel, dim3(1), dim3(256), 0, stream, grad, n, max_norm, eps, out_norm); return (int)hipGetLastError(); }

}
