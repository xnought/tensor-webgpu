# tensor-webgpu

Tensor Library that uses https://github.com/xnought/webgpu-compute for WebGPU computation.

Nearly all computations done on the GPU.

**Roadmap**

- [x] Random uniform, zeros, and ones array generation
- [ ] Shape Ops (Reshape, perumute, transpose, expand)
- [ ] Unary Ops (Power, ...)
- [ ] Binary Ops (Matmul, add, ...)
- [ ] ...

## Install

Download both JS (they have no dependencies) and simply import and use them.

```bash
wget https://raw.githubusercontent.com/xnought/webgpu-compute/refs/heads/main/webgpu-compute.js
wget https://raw.githubusercontent.com/xnought/tensor-webgpu/refs/heads/main/tensor-webgpu.js
```

## Unary Operations

Operations that take in one tensor and return one tensor.

### transpose()


Transposes the first and last dimension.

Or alias `.T` does the same thing

```js
const a = Tensor.tensor([1,2,3], [3, 1]);
console.log("a")
await a.print();

console.log("a.transpose()")
await a.transpose().print();

console.log("a.T")
await a.T.print();
```

`console outputs ↓`

```js
a
dtype='f32', shape=[3,1], strides=[1,1],
gpuBuffer=
[[1],
 [2],
 [3]]

a.transpose()
dtype='1,1', shape=[1,3], strides=[3,1],
gpuBuffer=
[[1, 2, 3]]

a.T
dtype='1,1', shape=[1,3], strides=[3,1],
gpuBuffer=
[[1, 2, 3]]
```


## Data Generation

The generation methods require a gpu and the imported Tensor js module

```js
import { Tensor } from "./tensor-webgpu";
import { GPU } from "./webgpu-compute";

const gpu = await GPU.init(); // will be used in Tensor functions
```

### Tensor.tensor()

Initialized GPU Tensor given array, dtype and shape.

In this case I want to send the vector `[1,2,3]` to the gpu with shape `[3,1]`.

```js
const a = Tensor.tensor(gpu, [1,2,3], [3,1], "f32");
await a.print();
```
`console outputs ↓`

```js
dtype='f32', shape=[3,1], strides=[1,1],
gpuBuffer=
[[1],
 [2],
 [3]]
```

### Tensor.random

Random value between `[0, 1)`.

In this case random values of a tensor shaped of `[4,1]` of 32 bit floating points.

```js
const a = Tensor.random(gpu, [4, 1], "f32");
await a.print();
```

`console outputs ↓`

```js
dtype='f32', shape=[4,1], strides=[1,1],
gpuBuffer=
[[0.11176824569702148],
 [0.5849729776382446],
 [0.9514476656913757],
 [0.7110687494277954]]
```

### Tensor.fill

Fill an entire tensor with a single scalar value. In this case a `[2,2,2]` shaped tensor of unsigned integers.

```js
const a = Tensor.fill(gpu, 1, [2, 2, 2], "u32");
await a.print();
```

`console outputs ↓`

```js
dtype='u32', shape=[2,2,2], strides=[4,2,1], 
gpuBuffer=
[[[1, 1],
  [1, 1]],

 [[1, 1],
  [1, 1]]]
```

## Dev

```bash
cd example
pnpm install
pnpm dev
```
