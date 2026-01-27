# Tensor API (v0.9)

The `std::tensor` module provides GPU tensor operations backed by `enkai_tensor`.

## Quick start

```
import std::tensor

fn main() ::
    let dev := tensor.device("cuda:0")
    let x := tensor.randn([2,2], "fp16", dev)
    let y := tensor.randn([2,2], "fp16", dev)
    let z := tensor.matmul(x, y)
    print(tensor.shape(z))
::
```

## Supported ops

- `tensor.device(spec)` -> Device handle (`"cpu"`, `"cuda:0"`)
- `tensor.randn(shape, dtype, device)` -> Tensor
- `tensor.zeros(shape, dtype, device)` -> Tensor
- `tensor.matmul(a, b)` -> Tensor
- `tensor.add(a, b)` -> Tensor
- `tensor.mul(a, b)` -> Tensor
- `tensor.reshape(x, shape)` -> Tensor
- `tensor.transpose(x, dim0, dim1)` -> Tensor
- `tensor.concat(handles, dim)` -> Tensor (handles is a JSON array or list of tensor handles)
- `tensor.sum(x, dim, keepdim)` -> Tensor (`dim=-1` sums all dims)
- `tensor.mean(x, dim, keepdim)` -> Tensor (`dim=-1` averages all dims)
- `tensor.softmax(x, dim)` -> Tensor
- `tensor.masked_softmax(x, mask, dim, mask_type)` -> Tensor
- `tensor.relu(x)` -> Tensor
- `tensor.sigmoid(x)` -> Tensor
- `tensor.dropout(x, p, train)` -> Tensor
- `tensor.slice(x, dim, start, end, step)` -> Tensor (use `-1` for start/end to mean "none")
- `tensor.view(x, shape)` -> Tensor
- `tensor.layernorm(x, w, b, eps)` -> Tensor
- `tensor.embedding(w, ids)` -> Tensor
- `tensor.linear(x, w, b)` -> Tensor
- `tensor.gelu(x)` -> Tensor
- `tensor.to_device(x, device)` -> Tensor
- `tensor.to_dtype(x, dtype)` -> Tensor
- `tensor.shape(x)` -> JSON array of ints

## Autograd helpers

- `tensor.cross_entropy(logits, target_ids)` -> loss tensor
- `tensor.backward(loss)`
- `tensor.adamw_step(param, grad, state, lr, beta1, beta2, eps, wd)` -> state handle
- `tensor.adamw_step_multi(params, grads, state, lr, beta1, beta2, eps, wd)` -> state handle
- `tensor.layernorm_backward(x, w, b, eps, grad_out)` -> record with `dx`, `dw`, `db`
- `tensor.masked_softmax_backward(grad_output, output, mask, dim)` -> Tensor

## Parameter collections

- `tensor.param_group(params, grads)` -> record with `params` and `grads`
- `tensor.param_group_step(group, state, lr, beta1, beta2, eps, wd)` -> state handle

Notes:
- Reuse the same `state` handle across multiple parameters to maintain optimizer state.
- `mask_type` for masked softmax is passed through to the backend; use `0` for boolean masks.

## Common errors

- `0` handle means backend error; check `enkai_tensor_last_error`.
- CUDA not available on the host.
