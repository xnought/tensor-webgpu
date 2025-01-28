# tensor-webgpu

Tensor Library that uses https://github.com/xnought/webgpu-compute for WebGPU computation.

Nearly all computations done on the GPU.

**Roadmap**

- [x] Random uniform, zeros, and ones array generation
- [x] pow
- [x] sum
- [ ] add
- [ ] sub
- [ ] mult
- [ ] dot

## Install

Download both JS (they have no dependencies) and simply import and use them.

```bash
wget https://raw.githubusercontent.com/xnought/tensor-webgpu/refs/heads/main/webgpu-compute.js
wget https://raw.githubusercontent.com/xnought/tensor-webgpu/refs/heads/main/tensor-webgpu.js
```

## Tensor Operations


### sum(dim)

Sums over the given axis/dimension. 

```js
const a = await Tensor.tensor(gpu, [0,1,2,3,4,5,6,7], [2,2,2], "f32");
console.log("a");
await a.print();

console.log("Sum across")
await (await a.sum(-1)).print();
```

`console outputs ↓`

```js
a
dtype='f32', shape=[2,2,2], strides=[4,2,1],
gpuBuffer=
[[[0, 1],
  [2, 3]],

 [[4, 5],
  [6, 7]]]

Sum across
type='f32', shape=[2,2,1], strides=[2,1,1],
gpuBuffer=
[[[1],
  [5]],

 [[9],
  [13]]]

```

### pow(number)

Raises every element in the Tensor to the given power. 

```js
const a = await Tensor.tensor(gpu, [1, 2, -3], [3, 1], "f32");
console.log("a");
await a.print();

console.log("a^5");
await (await a.pow(5)).print();
```

`console outputs ↓`

```js
a
dtype='f32', shape=[3,1], strides=[1,1],
gpuBuffer=
[[1],
 [2],
 [-3]]

a^5
dtype='f32', shape=[3,1], strides=[1,1],
gpuBuffer=
[[1],
 [32],
 [242.99996948242188]]
```

### transpose()


Transposes the first and last dimension.

Or alias `.T` does the same thing

```js
const a = await Tensor.tensor(gpu, [1, 2, 3], [3, 1]);
console.log("a");
await a.print();

console.log("a.transpose()");
await a.transpose().print();

console.log("a.T");
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
dtype='1,1', shape=[1,3], strides=[1,1],
gpuBuffer=
[[1, 2, 3]]

a.T
dtype='1,1', shape=[1,3], strides=[1,1],
gpuBuffer=
[[1, 2, 3]]
```

## Functional Tensor Operations

Every tensor operation (above in [tensor-operations](#tensor-operations)) has a functional definition where you allocate the output (or choose to do the op in-place)

```js
const a = await Tensor.tensor([1,2,3], [3,1]);
const aSum = await a.sum(0);
```

Is perfectly fine, but the a.sum() function allocates the shape of the output and returns it to you. 

Instead you could explicitly use the functional call `Tensor.<op name>(gpu, destination,...args)`;

```js
const aSum = await Tensor.empty([1,1]); // destination/result empty allocation
await Tensor.sum(gpu, aSum, a, 0); // compute sum down a and store in aSum
```

This function API also allows for in place operations. Like squaring

```js
await Tensor.pow(gpu, a, a, 2); // compute a^2 then override a with result 
```


## Data Generation

The generation methods require a gpu and the imported Tensor js module

```js
import { Tensor } from "./tensor-webgpu";
import { GPU } from "./webgpu-compute";

const gpu = await GPU.init(); // will be used in Tensor functions
```

### Tensor.tensor

Initialized GPU Tensor given array, dtype and shape.

In this case I want to send the vector `[1,2,3]` to the gpu with shape `[3,1]`.

```js
const a = await Tensor.tensor(gpu, [1,2,3], [3,1], "f32");
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
const a = await Tensor.random(gpu, [4, 1], "f32");
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
const a = await Tensor.fill(gpu, 1, [2, 2, 2], "u32");
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
