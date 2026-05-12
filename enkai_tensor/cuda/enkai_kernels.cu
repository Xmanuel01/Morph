#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <math_constants.h>
#include <stdint.h>

extern "C" {

static cublasHandle_t enkai_cublas_handle = nullptr;

int enkai_cuda_cublas_ready() {
    if (enkai_cublas_handle) return 0;
    cublasStatus_t status = cublasCreate(&enkai_cublas_handle);
    return status == CUBLAS_STATUS_SUCCESS ? 0 : (int)status;
}


__device__ float enkai_bf16_to_float(uint16_t x) {
    union { uint32_t u; float f; } v;
    v.u = ((uint32_t)x) << 16;
    return v.f;
}

__device__ uint16_t enkai_float_to_bf16(float x) {
    union { uint32_t u; float f; } v;
    v.f = x;
    uint32_t rounded = v.u + 0x7fffu + ((v.u >> 16) & 1u);
    return (uint16_t)(rounded >> 16);
}

__device__ float enkai_gelu_f32(float v) {
    float c = 0.7978845608028654f * (v + 0.044715f * v * v * v);
    return 0.5f * v * (1.0f + tanhf(c));
}

__global__ void enkai_vec_add_f16_kernel(const __half* a, const __half* b, __half* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(__half2float(a[i]) + __half2float(b[i]));
}

__global__ void enkai_vec_mul_f16_kernel(const __half* a, const __half* b, __half* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(__half2float(a[i]) * __half2float(b[i]));
}

__global__ void enkai_vec_scale_f16_kernel(const __half* x, float scale, __half* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(__half2float(x[i]) * scale);
}

__global__ void enkai_vec_add_bf16_kernel(const uint16_t* a, const uint16_t* b, uint16_t* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = enkai_float_to_bf16(enkai_bf16_to_float(a[i]) + enkai_bf16_to_float(b[i]));
}

__global__ void enkai_vec_mul_bf16_kernel(const uint16_t* a, const uint16_t* b, uint16_t* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = enkai_float_to_bf16(enkai_bf16_to_float(a[i]) * enkai_bf16_to_float(b[i]));
}

__global__ void enkai_vec_scale_bf16_kernel(const uint16_t* x, float scale, uint16_t* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = enkai_float_to_bf16(enkai_bf16_to_float(x[i]) * scale);
}

__global__ void enkai_bias_gelu_f16_kernel(const __half* x, const __half* bias, __half* out, int64_t rows, int64_t cols) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = rows * cols;
    if (i >= n) return;
    float v = __half2float(x[i]) + __half2float(bias[i % cols]);
    out[i] = __float2half(enkai_gelu_f32(v));
}

__global__ void enkai_bias_gelu_bf16_kernel(const uint16_t* x, const uint16_t* bias, uint16_t* out, int64_t rows, int64_t cols) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = rows * cols;
    if (i >= n) return;
    float v = enkai_bf16_to_float(x[i]) + enkai_bf16_to_float(bias[i % cols]);
    out[i] = enkai_float_to_bf16(enkai_gelu_f32(v));
}

__global__ void enkai_matmul_bias_f16_kernel(const __half* a, const __half* b, const __half* bias, __half* out, int64_t m, int64_t n, int64_t k) {
    int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;
    float acc = bias ? __half2float(bias[col]) : 0.0f;
    for (int64_t i = 0; i < k; ++i) acc += __half2float(a[row * k + i]) * __half2float(b[i * n + col]);
    out[row * n + col] = __float2half(acc);
}

__global__ void enkai_matmul_bias_bf16_kernel(const uint16_t* a, const uint16_t* b, const uint16_t* bias, uint16_t* out, int64_t m, int64_t n, int64_t k) {
    int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;
    float acc = bias ? enkai_bf16_to_float(bias[col]) : 0.0f;
    for (int64_t i = 0; i < k; ++i) acc += enkai_bf16_to_float(a[row * k + i]) * enkai_bf16_to_float(b[i * n + col]);
    out[row * n + col] = enkai_float_to_bf16(acc);
}

__global__ void enkai_cross_entropy_forward_f32_kernel(const float* logits, const int64_t* targets, float* losses, int64_t rows, int64_t cols) {
    int64_t row = blockIdx.x;
    if (row >= rows) return;
    int64_t target = targets[row];
    if (target < 0 || target >= cols) {
        if (threadIdx.x == 0) losses[row] = CUDART_INF_F;
        return;
    }
    const float* xr = logits + row * cols;
    __shared__ float scratch[256];
    float max_v = -CUDART_INF_F;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) max_v = fmaxf(max_v, xr[c]);
    scratch[threadIdx.x] = max_v;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        __syncthreads();
    }
    max_v = scratch[0];
    float sum = 0.0f;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) sum += expf(xr[c] - max_v);
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) losses[row] = logf(fmaxf(scratch[0], 1.0e-20f)) + max_v - xr[target];
}

__global__ void enkai_cross_entropy_backward_f32_kernel(const float* logits, const int64_t* targets, float* grad, int64_t rows, int64_t cols, float scale) {
    int64_t row = blockIdx.x;
    if (row >= rows) return;
    int64_t target = targets[row];
    const float* xr = logits + row * cols;
    float* gr = grad + row * cols;
    if (target < 0 || target >= cols) {
        for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) gr[c] = CUDART_NAN_F;
        return;
    }
    __shared__ float scratch[256];
    float max_v = -CUDART_INF_F;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) max_v = fmaxf(max_v, xr[c]);
    scratch[threadIdx.x] = max_v;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        __syncthreads();
    }
    max_v = scratch[0];
    float sum = 0.0f;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) sum += expf(xr[c] - max_v);
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    float denom = fmaxf(scratch[0], 1.0e-20f);
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
        float softmax = expf(xr[c] - max_v) / denom;
        gr[c] = (softmax - (c == target ? 1.0f : 0.0f)) * scale;
    }
}

__global__ void enkai_embedding_forward_f32_kernel(const float* weights, const int64_t* ids, float* out, int64_t ids_len, int64_t dim, int64_t vocab) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = ids_len * dim;
    if (i >= n) return;
    int64_t row = i / dim;
    int64_t d = i % dim;
    int64_t id = ids[row];
    out[i] = (id >= 0 && id < vocab) ? weights[id * dim + d] : CUDART_NAN_F;
}

__global__ void enkai_embedding_backward_f32_kernel(const float* grad_out, const int64_t* ids, float* grad_weights, int64_t ids_len, int64_t dim, int64_t vocab) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = ids_len * dim;
    if (i >= n) return;
    int64_t row = i / dim;
    int64_t d = i % dim;
    int64_t id = ids[row];
    if (id >= 0 && id < vocab) atomicAdd(grad_weights + id * dim + d, grad_out[i]);
}

__global__ void enkai_clip_grad_norm_f32_kernel(float* grad, int64_t n, float max_norm, float eps, float* out_norm) {
    __shared__ float scratch[256];
    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) sum += grad[i] * grad[i];
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    float norm = sqrtf(scratch[0]);
    float coeff = max_norm / (norm + eps);
    if (threadIdx.x == 0 && out_norm) *out_norm = norm;
    coeff = coeff < 1.0f ? coeff : 1.0f;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) grad[i] *= coeff;
}

__global__ void enkai_causal_attention_backward_value_f32_kernel(const float* q, const float* k, const float* grad_out, float* grad_v, int64_t batch_heads, int64_t seq, int64_t head_dim) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = batch_heads * seq * head_dim;
    if (i >= n) return;
    int64_t vpos = (i / head_dim) % seq;
    int64_t d = i % head_dim;
    int64_t bh = i / (seq * head_dim);
    const float scale = rsqrtf((float)head_dim);
    const float* krow = k + (bh * seq + vpos) * head_dim;
    float acc = 0.0f;
    for (int64_t qpos = vpos; qpos < seq; ++qpos) {
        const float* qrow = q + (bh * seq + qpos) * head_dim;
        float max_score = -CUDART_INF_F;
        for (int64_t t = 0; t <= qpos; ++t) {
            const float* kt = k + (bh * seq + t) * head_dim;
            float score = 0.0f;
            for (int64_t c = 0; c < head_dim; ++c) score += qrow[c] * kt[c];
            max_score = fmaxf(max_score, score * scale);
        }
        float denom = 0.0f;
        float score_v = 0.0f;
        for (int64_t c = 0; c < head_dim; ++c) score_v += qrow[c] * krow[c];
        for (int64_t t = 0; t <= qpos; ++t) {
            const float* kt = k + (bh * seq + t) * head_dim;
            float score = 0.0f;
            for (int64_t c = 0; c < head_dim; ++c) score += qrow[c] * kt[c];
            denom += expf(score * scale - max_score);
        }
        float weight = expf(score_v * scale - max_score) / fmaxf(denom, 1.0e-20f);
        acc += weight * grad_out[(bh * seq + qpos) * head_dim + d];
    }
    grad_v[i] = acc;
}


__device__ float enkai_attention_score_f32(const float* qrow, const float* krow, int64_t head_dim, float scale) {
    float score = 0.0f;
    for (int64_t c = 0; c < head_dim; ++c) score += qrow[c] * krow[c];
    return score * scale;
}

__device__ void enkai_attention_row_stats_f32(const float* q, const float* k, const float* v, const float* grad_out, int64_t bh, int64_t qpos, int64_t seq, int64_t head_dim, float scale, float* max_score, float* denom, float* weighted_go_v) {
    const float* qrow = q + (bh * seq + qpos) * head_dim;
    *max_score = -CUDART_INF_F;
    for (int64_t t = 0; t <= qpos; ++t) {
        const float* krow = k + (bh * seq + t) * head_dim;
        *max_score = fmaxf(*max_score, enkai_attention_score_f32(qrow, krow, head_dim, scale));
    }
    *denom = 0.0f;
    *weighted_go_v = 0.0f;
    const float* gorow = grad_out + (bh * seq + qpos) * head_dim;
    for (int64_t t = 0; t <= qpos; ++t) {
        const float* krow = k + (bh * seq + t) * head_dim;
        const float* vrow = v + (bh * seq + t) * head_dim;
        float weight = expf(enkai_attention_score_f32(qrow, krow, head_dim, scale) - *max_score);
        *denom += weight;
        float go_dot_v = 0.0f;
        for (int64_t c = 0; c < head_dim; ++c) go_dot_v += gorow[c] * vrow[c];
        *weighted_go_v += weight * go_dot_v;
    }
    *denom = fmaxf(*denom, 1.0e-20f);
    *weighted_go_v /= *denom;
}

__global__ void enkai_causal_attention_backward_q_f32_kernel(const float* q, const float* k, const float* v, const float* grad_out, float* grad_q, int64_t batch_heads, int64_t seq, int64_t head_dim) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = batch_heads * seq * head_dim;
    if (i >= n) return;
    int64_t d = i % head_dim;
    int64_t qpos = (i / head_dim) % seq;
    int64_t bh = i / (seq * head_dim);
    const float scale = rsqrtf((float)head_dim);
    const float* qrow = q + (bh * seq + qpos) * head_dim;
    const float* gorow = grad_out + (bh * seq + qpos) * head_dim;
    float max_score, denom, center;
    enkai_attention_row_stats_f32(q, k, v, grad_out, bh, qpos, seq, head_dim, scale, &max_score, &denom, &center);
    float acc = 0.0f;
    for (int64_t t = 0; t <= qpos; ++t) {
        const float* krow = k + (bh * seq + t) * head_dim;
        const float* vrow = v + (bh * seq + t) * head_dim;
        float weight = expf(enkai_attention_score_f32(qrow, krow, head_dim, scale) - max_score) / denom;
        float go_dot_v = 0.0f;
        for (int64_t c = 0; c < head_dim; ++c) go_dot_v += gorow[c] * vrow[c];
        float ds = weight * (go_dot_v - center);
        acc += ds * krow[d] * scale;
    }
    grad_q[i] = acc;
}

__global__ void enkai_causal_attention_backward_k_f32_kernel(const float* q, const float* k, const float* v, const float* grad_out, float* grad_k, int64_t batch_heads, int64_t seq, int64_t head_dim) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = batch_heads * seq * head_dim;
    if (i >= n) return;
    int64_t d = i % head_dim;
    int64_t kpos = (i / head_dim) % seq;
    int64_t bh = i / (seq * head_dim);
    const float scale = rsqrtf((float)head_dim);
    const float* krow = k + (bh * seq + kpos) * head_dim;
    float acc = 0.0f;
    for (int64_t qpos = kpos; qpos < seq; ++qpos) {
        const float* qrow = q + (bh * seq + qpos) * head_dim;
        const float* gorow = grad_out + (bh * seq + qpos) * head_dim;
        const float* vrow = v + (bh * seq + kpos) * head_dim;
        float max_score, denom, center;
        enkai_attention_row_stats_f32(q, k, v, grad_out, bh, qpos, seq, head_dim, scale, &max_score, &denom, &center);
        float weight = expf(enkai_attention_score_f32(qrow, krow, head_dim, scale) - max_score) / denom;
        float go_dot_v = 0.0f;
        for (int64_t c = 0; c < head_dim; ++c) go_dot_v += gorow[c] * vrow[c];
        float ds = weight * (go_dot_v - center);
        acc += ds * qrow[d] * scale;
    }
    grad_k[i] = acc;
}


__global__ void enkai_vec_add_f32_kernel(const float* a, const float* b, float* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void enkai_vec_mul_f32_kernel(const float* a, const float* b, float* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

__global__ void enkai_vec_scale_f32_kernel(const float* x, float scale, float* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] * scale;
}

__global__ void enkai_bias_gelu_f32_kernel(const float* x, const float* bias, float* out, int64_t rows, int64_t cols) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = rows * cols;
    if (i >= n) return;
    float v = x[i] + bias[i % cols];
    float c = 0.7978845608028654f * (v + 0.044715f * v * v * v);
    out[i] = 0.5f * v * (1.0f + tanhf(c));
}

__global__ void enkai_matmul_bias_f32_kernel(const float* a, const float* b, const float* bias, float* out, int64_t m, int64_t n, int64_t k) {
    int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;
    float acc = bias ? bias[col] : 0.0f;
    for (int64_t i = 0; i < k; ++i) {
        acc += a[row * k + i] * b[i * n + col];
    }
    out[row * n + col] = acc;
}

__global__ void enkai_add_bias_inplace_f32_kernel(float* out, const float* bias, int64_t rows, int64_t cols) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = rows * cols;
    if (i < total) out[i] += bias[i % cols];
}

__global__ void enkai_layernorm_f32_kernel(const float* x, const float* gamma, const float* beta, float* out, int64_t rows, int64_t cols, float eps) {
    int64_t row = blockIdx.x;
    if (row >= rows) return;
    const float* xr = x + row * cols;
    float* yr = out + row * cols;
    float mean = 0.0f;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) mean += xr[c];
    __shared__ float scratch[256];
    scratch[threadIdx.x] = mean;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    mean = scratch[0] / (float)cols;
    float var = 0.0f;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
        float d = xr[c] - mean;
        var += d * d;
    }
    scratch[threadIdx.x] = var;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    float inv = rsqrtf(scratch[0] / (float)cols + eps);
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
        yr[c] = (xr[c] - mean) * inv * gamma[c] + beta[c];
    }
}

__global__ void enkai_softmax_f32_kernel(const float* x, float* out, int64_t rows, int64_t cols) {
    int64_t row = blockIdx.x;
    if (row >= rows) return;
    const float* xr = x + row * cols;
    float* yr = out + row * cols;
    __shared__ float scratch[256];
    float max_v = -CUDART_INF_F;
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
    float denom = scratch[0];
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) yr[c] /= denom;
}

__global__ void enkai_masked_softmax_f32_kernel(const float* x, const uint8_t* mask, float* out, int64_t rows, int64_t cols) {
    int64_t row = blockIdx.x;
    if (row >= rows) return;
    const float* xr = x + row * cols;
    const uint8_t* mr = mask + row * cols;
    float* yr = out + row * cols;
    __shared__ float scratch[256];
    float max_v = -CUDART_INF_F;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
        if (mr[c]) max_v = fmaxf(max_v, xr[c]);
    }
    scratch[threadIdx.x] = max_v;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        __syncthreads();
    }
    max_v = scratch[0];
    float sum = 0.0f;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
        float e = mr[c] ? expf(xr[c] - max_v) : 0.0f;
        yr[c] = e;
        sum += e;
    }
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    float denom = scratch[0] > 0.0f ? scratch[0] : 1.0f;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) yr[c] /= denom;
}

__global__ void enkai_causal_attention_prefill_f32_kernel(const float* q, const float* k, const float* v, float* out, int64_t batch_heads, int64_t seq, int64_t head_dim) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = batch_heads * seq * head_dim;
    if (i >= n) return;
    int64_t d = i % head_dim;
    int64_t qpos = (i / head_dim) % seq;
    int64_t bh = i / (seq * head_dim);
    const float scale = rsqrtf((float)head_dim);
    const float* qrow = q + (bh * seq + qpos) * head_dim;
    const float* kbase = k + bh * seq * head_dim;
    const float* vbase = v + bh * seq * head_dim;
    float max_score = -CUDART_INF_F;
    for (int64_t t = 0; t <= qpos; ++t) {
        float score = 0.0f;
        const float* krow = kbase + t * head_dim;
        for (int64_t c = 0; c < head_dim; ++c) score += qrow[c] * krow[c];
        max_score = fmaxf(max_score, score * scale);
    }
    float denom = 0.0f;
    float acc = 0.0f;
    for (int64_t t = 0; t <= qpos; ++t) {
        float score = 0.0f;
        const float* krow = kbase + t * head_dim;
        for (int64_t c = 0; c < head_dim; ++c) score += qrow[c] * krow[c];
        float weight = expf(score * scale - max_score);
        denom += weight;
        acc += weight * vbase[t * head_dim + d];
    }
    out[i] = acc / fmaxf(denom, 1.0e-20f);
}

__global__ void enkai_kv_cache_decode_f32_kernel(const float* q, const float* k_cache, const float* v_cache, float* out, int64_t batch_heads, int64_t cache_len, int64_t head_dim) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = batch_heads * head_dim;
    if (i >= n) return;
    int64_t d = i % head_dim;
    int64_t bh = i / head_dim;
    const float scale = rsqrtf((float)head_dim);
    const float* qrow = q + bh * head_dim;
    const float* kbase = k_cache + bh * cache_len * head_dim;
    const float* vbase = v_cache + bh * cache_len * head_dim;
    float max_score = -CUDART_INF_F;
    for (int64_t t = 0; t < cache_len; ++t) {
        float score = 0.0f;
        const float* krow = kbase + t * head_dim;
        for (int64_t c = 0; c < head_dim; ++c) score += qrow[c] * krow[c];
        max_score = fmaxf(max_score, score * scale);
    }
    float denom = 0.0f;
    float acc = 0.0f;
    for (int64_t t = 0; t < cache_len; ++t) {
        float score = 0.0f;
        const float* krow = kbase + t * head_dim;
        for (int64_t c = 0; c < head_dim; ++c) score += qrow[c] * krow[c];
        float weight = expf(score * scale - max_score);
        denom += weight;
        acc += weight * vbase[t * head_dim + d];
    }
    out[i] = acc / fmaxf(denom, 1.0e-20f);
}

__global__ void enkai_adamw_update_f32_kernel(float* param, const float* grad, float* m, float* v, int64_t n, float lr, float beta1, float beta2, float eps, float wd, int64_t step) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = grad[i] + wd * param[i];
    float mt = beta1 * m[i] + (1.0f - beta1) * g;
    float vt = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = mt;
    v[i] = vt;
    float bc1 = 1.0f - powf(beta1, (float)step);
    float bc2 = 1.0f - powf(beta2, (float)step);
    param[i] -= lr * (mt / bc1) / (sqrtf(vt / bc2) + eps);
}

int enkai_cuda_vec_add_f32(const float* a, const float* b, float* out, int64_t n, cudaStream_t stream) {
    if (!a || !b || !out || n < 0) return 1;
    int threads = 256;
    int64_t blocks = (n + threads - 1) / threads;
    enkai_vec_add_f32_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(a, b, out, n);
    return (int)cudaGetLastError();
}

int enkai_cuda_vec_mul_f32(const float* a, const float* b, float* out, int64_t n, cudaStream_t stream) {
    if (!a || !b || !out || n < 0) return 1;
    int threads = 256;
    int64_t blocks = (n + threads - 1) / threads;
    enkai_vec_mul_f32_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(a, b, out, n);
    return (int)cudaGetLastError();
}

int enkai_cuda_vec_scale_f32(const float* x, float scale, float* out, int64_t n, cudaStream_t stream) {
    if (!x || !out || n < 0) return 1;
    int threads = 256;
    int64_t blocks = (n + threads - 1) / threads;
    enkai_vec_scale_f32_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(x, scale, out, n);
    return (int)cudaGetLastError();
}

int enkai_cuda_bias_gelu_f32(const float* x, const float* bias, float* out, int64_t rows, int64_t cols, cudaStream_t stream) {
    if (!x || !bias || !out || rows < 0 || cols <= 0) return 1;
    int threads = 256;
    int64_t n = rows * cols;
    int64_t blocks = (n + threads - 1) / threads;
    enkai_bias_gelu_f32_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(x, bias, out, rows, cols);
    return (int)cudaGetLastError();
}

int enkai_cuda_matmul_bias_f32(const float* a, const float* b, const float* bias, float* out, int64_t m, int64_t n, int64_t k, cudaStream_t stream) {
    if (!a || !b || !out || m < 0 || n <= 0 || k <= 0) return 1;
    dim3 threads(16, 16);
    dim3 blocks((unsigned int)((n + 15) / 16), (unsigned int)((m + 15) / 16));
    enkai_matmul_bias_f32_kernel<<<blocks, threads, 0, stream>>>(a, b, bias, out, m, n, k);
    return (int)cudaGetLastError();
}

int enkai_cuda_matmul_bias_cublas_f32(const float* a, const float* b, const float* bias, float* out, int64_t m, int64_t n, int64_t k, cudaStream_t stream) {
    if (!a || !b || !out || m <= 0 || n <= 0 || k <= 0) return 1;
    int ready = enkai_cuda_cublas_ready();
    if (ready != 0) return ready;
    cublasStatus_t s = cublasSetStream(enkai_cublas_handle, stream);
    if (s != CUBLAS_STATUS_SUCCESS) return (int)s;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // Row-major A[M,K] * B[K,N] is equivalent to column-major B^T[N,K] * A^T[K,M] -> C^T[N,M].
    s = cublasSgemm(
        enkai_cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        (int)n,
        (int)m,
        (int)k,
        &alpha,
        b,
        (int)n,
        a,
        (int)k,
        &beta,
        out,
        (int)n
    );
    if (s != CUBLAS_STATUS_SUCCESS) return (int)s;
    if (bias) {
        int threads = 256;
        int64_t total = m * n;
        int64_t blocks = (total + threads - 1) / threads;
        enkai_add_bias_inplace_f32_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(out, bias, m, n);
        return (int)cudaGetLastError();
    }
    return (int)cudaGetLastError();
}

int enkai_cuda_layernorm_f32(const float* x, const float* gamma, const float* beta, float* out, int64_t rows, int64_t cols, float eps, cudaStream_t stream) {
    if (!x || !gamma || !beta || !out || rows < 0 || cols <= 0 || cols > 4096) return 1;
    enkai_layernorm_f32_kernel<<<(unsigned int)rows, 256, 0, stream>>>(x, gamma, beta, out, rows, cols, eps);
    return (int)cudaGetLastError();
}

int enkai_cuda_softmax_f32(const float* x, float* out, int64_t rows, int64_t cols, cudaStream_t stream) {
    if (!x || !out || rows < 0 || cols <= 0 || cols > 4096) return 1;
    enkai_softmax_f32_kernel<<<(unsigned int)rows, 256, 0, stream>>>(x, out, rows, cols);
    return (int)cudaGetLastError();
}

int enkai_cuda_masked_softmax_f32(const float* x, const uint8_t* mask, float* out, int64_t rows, int64_t cols, cudaStream_t stream) {
    if (!x || !mask || !out || rows < 0 || cols <= 0 || cols > 4096) return 1;
    enkai_masked_softmax_f32_kernel<<<(unsigned int)rows, 256, 0, stream>>>(x, mask, out, rows, cols);
    return (int)cudaGetLastError();
}

int enkai_cuda_causal_attention_prefill_f32(const float* q, const float* k, const float* v, float* out, int64_t batch_heads, int64_t seq, int64_t head_dim, cudaStream_t stream) {
    if (!q || !k || !v || !out || batch_heads <= 0 || seq <= 0 || head_dim <= 0 || head_dim > 4096) return 1;
    int threads = 256;
    int64_t n = batch_heads * seq * head_dim;
    int64_t blocks = (n + threads - 1) / threads;
    enkai_causal_attention_prefill_f32_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(q, k, v, out, batch_heads, seq, head_dim);
    return (int)cudaGetLastError();
}

int enkai_cuda_kv_cache_decode_f32(const float* q, const float* k_cache, const float* v_cache, float* out, int64_t batch_heads, int64_t cache_len, int64_t head_dim, cudaStream_t stream) {
    if (!q || !k_cache || !v_cache || !out || batch_heads <= 0 || cache_len <= 0 || head_dim <= 0 || head_dim > 4096) return 1;
    int threads = 256;
    int64_t n = batch_heads * head_dim;
    int64_t blocks = (n + threads - 1) / threads;
    enkai_kv_cache_decode_f32_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(q, k_cache, v_cache, out, batch_heads, cache_len, head_dim);
    return (int)cudaGetLastError();
}

int enkai_cuda_adamw_update_f32(float* param, const float* grad, float* m, float* v, int64_t n, float lr, float beta1, float beta2, float eps, float wd, int64_t step, cudaStream_t stream) {
    if (!param || !grad || !m || !v || n < 0 || step <= 0) return 1;
    int threads = 256;
    int64_t blocks = (n + threads - 1) / threads;
    enkai_adamw_update_f32_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(param, grad, m, v, n, lr, beta1, beta2, eps, wd, step);
    return (int)cudaGetLastError();
}


int enkai_cuda_vec_add_f16(const void* a, const void* b, void* out, int64_t n, cudaStream_t stream) {
    if (!a || !b || !out || n < 0) return 1;
    int threads = 256;
    int64_t blocks = (n + threads - 1) / threads;
    enkai_vec_add_f16_kernel<<<(unsigned int)blocks, threads, 0, stream>>>((const __half*)a, (const __half*)b, (__half*)out, n);
    return (int)cudaGetLastError();
}

int enkai_cuda_vec_mul_f16(const void* a, const void* b, void* out, int64_t n, cudaStream_t stream) {
    if (!a || !b || !out || n < 0) return 1;
    int threads = 256;
    int64_t blocks = (n + threads - 1) / threads;
    enkai_vec_mul_f16_kernel<<<(unsigned int)blocks, threads, 0, stream>>>((const __half*)a, (const __half*)b, (__half*)out, n);
    return (int)cudaGetLastError();
}

int enkai_cuda_vec_scale_f16(const void* x, float scale, void* out, int64_t n, cudaStream_t stream) {
    if (!x || !out || n < 0) return 1;
    int threads = 256;
    int64_t blocks = (n + threads - 1) / threads;
    enkai_vec_scale_f16_kernel<<<(unsigned int)blocks, threads, 0, stream>>>((const __half*)x, scale, (__half*)out, n);
    return (int)cudaGetLastError();
}

int enkai_cuda_vec_add_bf16(const uint16_t* a, const uint16_t* b, uint16_t* out, int64_t n, cudaStream_t stream) {
    if (!a || !b || !out || n < 0) return 1;
    int threads = 256;
    int64_t blocks = (n + threads - 1) / threads;
    enkai_vec_add_bf16_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(a, b, out, n);
    return (int)cudaGetLastError();
}

int enkai_cuda_vec_mul_bf16(const uint16_t* a, const uint16_t* b, uint16_t* out, int64_t n, cudaStream_t stream) {
    if (!a || !b || !out || n < 0) return 1;
    int threads = 256;
    int64_t blocks = (n + threads - 1) / threads;
    enkai_vec_mul_bf16_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(a, b, out, n);
    return (int)cudaGetLastError();
}

int enkai_cuda_vec_scale_bf16(const uint16_t* x, float scale, uint16_t* out, int64_t n, cudaStream_t stream) {
    if (!x || !out || n < 0) return 1;
    int threads = 256;
    int64_t blocks = (n + threads - 1) / threads;
    enkai_vec_scale_bf16_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(x, scale, out, n);
    return (int)cudaGetLastError();
}

int enkai_cuda_bias_gelu_f16(const void* x, const void* bias, void* out, int64_t rows, int64_t cols, cudaStream_t stream) {
    if (!x || !bias || !out || rows < 0 || cols <= 0) return 1;
    int threads = 256;
    int64_t blocks = (rows * cols + threads - 1) / threads;
    enkai_bias_gelu_f16_kernel<<<(unsigned int)blocks, threads, 0, stream>>>((const __half*)x, (const __half*)bias, (__half*)out, rows, cols);
    return (int)cudaGetLastError();
}

int enkai_cuda_bias_gelu_bf16(const uint16_t* x, const uint16_t* bias, uint16_t* out, int64_t rows, int64_t cols, cudaStream_t stream) {
    if (!x || !bias || !out || rows < 0 || cols <= 0) return 1;
    int threads = 256;
    int64_t blocks = (rows * cols + threads - 1) / threads;
    enkai_bias_gelu_bf16_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(x, bias, out, rows, cols);
    return (int)cudaGetLastError();
}

int enkai_cuda_matmul_bias_f16(const void* a, const void* b, const void* bias, void* out, int64_t m, int64_t n, int64_t k, cudaStream_t stream) {
    if (!a || !b || !out || m < 0 || n <= 0 || k <= 0) return 1;
    dim3 threads(16, 16);
    dim3 blocks((unsigned int)((n + 15) / 16), (unsigned int)((m + 15) / 16));
    enkai_matmul_bias_f16_kernel<<<blocks, threads, 0, stream>>>((const __half*)a, (const __half*)b, (const __half*)bias, (__half*)out, m, n, k);
    return (int)cudaGetLastError();
}

int enkai_cuda_matmul_bias_bf16(const uint16_t* a, const uint16_t* b, const uint16_t* bias, uint16_t* out, int64_t m, int64_t n, int64_t k, cudaStream_t stream) {
    if (!a || !b || !out || m < 0 || n <= 0 || k <= 0) return 1;
    dim3 threads(16, 16);
    dim3 blocks((unsigned int)((n + 15) / 16), (unsigned int)((m + 15) / 16));
    enkai_matmul_bias_bf16_kernel<<<blocks, threads, 0, stream>>>(a, b, bias, out, m, n, k);
    return (int)cudaGetLastError();
}

int enkai_cuda_cross_entropy_forward_f32(const float* logits, const int64_t* targets, float* losses, int64_t rows, int64_t cols, cudaStream_t stream) {
    if (!logits || !targets || !losses || rows < 0 || cols <= 1 || cols > 4096) return 1;
    enkai_cross_entropy_forward_f32_kernel<<<(unsigned int)rows, 256, 0, stream>>>(logits, targets, losses, rows, cols);
    return (int)cudaGetLastError();
}

int enkai_cuda_cross_entropy_backward_f32(const float* logits, const int64_t* targets, float* grad, int64_t rows, int64_t cols, float scale, cudaStream_t stream) {
    if (!logits || !targets || !grad || rows < 0 || cols <= 1 || cols > 4096) return 1;
    enkai_cross_entropy_backward_f32_kernel<<<(unsigned int)rows, 256, 0, stream>>>(logits, targets, grad, rows, cols, scale);
    return (int)cudaGetLastError();
}

int enkai_cuda_embedding_forward_f32(const float* weights, const int64_t* ids, float* out, int64_t ids_len, int64_t dim, int64_t vocab, cudaStream_t stream) {
    if (!weights || !ids || !out || ids_len < 0 || dim <= 0 || vocab <= 0) return 1;
    int threads = 256;
    int64_t blocks = (ids_len * dim + threads - 1) / threads;
    enkai_embedding_forward_f32_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(weights, ids, out, ids_len, dim, vocab);
    return (int)cudaGetLastError();
}

int enkai_cuda_embedding_backward_f32(const float* grad_out, const int64_t* ids, float* grad_weights, int64_t ids_len, int64_t dim, int64_t vocab, cudaStream_t stream) {
    if (!grad_out || !ids || !grad_weights || ids_len < 0 || dim <= 0 || vocab <= 0) return 1;
    int threads = 256;
    int64_t blocks = (ids_len * dim + threads - 1) / threads;
    enkai_embedding_backward_f32_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(grad_out, ids, grad_weights, ids_len, dim, vocab);
    return (int)cudaGetLastError();
}

int enkai_cuda_clip_grad_norm_f32(float* grad, int64_t n, float max_norm, float eps, float* out_norm, cudaStream_t stream) {
    if (!grad || n < 0 || max_norm < 0.0f || eps <= 0.0f) return 1;
    enkai_clip_grad_norm_f32_kernel<<<1, 256, 0, stream>>>(grad, n, max_norm, eps, out_norm);
    return (int)cudaGetLastError();
}

int enkai_cuda_causal_attention_backward_value_f32(const float* q, const float* k, const float* grad_out, float* grad_v, int64_t batch_heads, int64_t seq, int64_t head_dim, cudaStream_t stream) {
    if (!q || !k || !grad_out || !grad_v || batch_heads <= 0 || seq <= 0 || head_dim <= 0 || head_dim > 4096) return 1;
    int threads = 256;
    int64_t blocks = (batch_heads * seq * head_dim + threads - 1) / threads;
    enkai_causal_attention_backward_value_f32_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(q, k, grad_out, grad_v, batch_heads, seq, head_dim);
    return (int)cudaGetLastError();
}


int enkai_cuda_causal_attention_backward_f32(const float* q, const float* k, const float* v, const float* grad_out, float* grad_q, float* grad_k, float* grad_v, int64_t batch_heads, int64_t seq, int64_t head_dim, cudaStream_t stream) {
    if (!q || !k || !v || !grad_out || !grad_q || !grad_k || !grad_v || batch_heads <= 0 || seq <= 0 || head_dim <= 0 || head_dim > 4096) return 1;
    int threads = 256;
    int64_t blocks = (batch_heads * seq * head_dim + threads - 1) / threads;
    enkai_causal_attention_backward_q_f32_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(q, k, v, grad_out, grad_q, batch_heads, seq, head_dim);
    cudaError_t q_err = cudaGetLastError();
    if (q_err != cudaSuccess) return (int)q_err;
    enkai_causal_attention_backward_k_f32_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(q, k, v, grad_out, grad_k, batch_heads, seq, head_dim);
    cudaError_t k_err = cudaGetLastError();
    if (k_err != cudaSuccess) return (int)k_err;
    enkai_causal_attention_backward_value_f32_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(q, k, grad_out, grad_v, batch_heads, seq, head_dim);
    return (int)cudaGetLastError();
}

}
