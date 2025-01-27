# tensor-webgpu

Tensor Library that uses https://github.com/xnought/webgpu-compute for WebGPU computation.

**Roadmap**

- [x] Random uniform, zeros, and ones array generation
- [ ] Unary Ops (Power, ...)
- [ ] Shape Ops (Reshape, perumute, transpose, expand)
- [ ] Binary Ops (Matmul, add, ...)
- [ ] ...

## Install

Download both JS (they have no dependencies) and simply import and use them.

```bash
wget https://raw.githubusercontent.com/xnought/webgpu-compute/refs/heads/main/webgpu-compute.js
wget https://raw.githubusercontent.com/xnought/tensor-webgpu/refs/heads/main/tensor-webgpu.js
```

## Usage

The array generations methods require a gpu and the imported Tensor js module

```js
import { Tensor } from "./tensor-webgpu";
import { GPU } from "./wbgpu-compute";

const gpu = await GPU.init(); // will be used in Tensor functions
```

### Tensor.random

Random value between `[0, 1)`.

In this case random values of a tensor shaped of `[4,1]` of 32 bit floating points.

```js
const a = Tensor.random(gpu, [4, 1], "f32");
await a.print();
```

```js
// console >
gpuBuffer=
[[0.11176824569702148],
 [0.5849729776382446],
 [0.9514476656913757],
 [0.7110687494277954]]

dtype='f32', shape=[4,1], strides=[1,1],
```
### Tensor.fill

Fill an entire tensor with a single scalar value. In this case a `[2,2,2]` shaped tensor of unsigned integers.

```js
const a = Tensor.fill(gpu, 1, [2, 2, 2], "u32");
await a.print();
```

```js
gpuBuffer=
[[[1, 1],
  [1, 1]],

 [[1, 1],
  [1, 1]]]

dtype='u32', shape=[2,2,2], strides=[4,2,1], 
```

## Dev

```bash
cd example
pnpm install
pnpm dev
```