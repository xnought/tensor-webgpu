# tensor-webgpu

Tensor Library that uses https://github.com/xnought/webgpu-compute for WebGPU computation.

Nearly all computations done on the GPU.

**Roadmap**

- [x] Random uniform, zeros, and ones array generation
- [x] pow
- [x] sum
- [x] add
- [x] sub
- [x] mult
- [x] div
- [x] matmul
- [x] contiguous
- [ ] Broadcasting
- [ ] Reshape
- [ ] Squeeze
- [ ] Linear Regression
- [ ] MLP Example

## Getting Started

Download both JS (they have no dependencies) and simply import and use them.

```bash
wget https://raw.githubusercontent.com/xnought/tensor-webgpu/refs/heads/main/webgpu-compute.js
wget https://raw.githubusercontent.com/xnought/tensor-webgpu/refs/heads/main/tensor-webgpu.js
```

Before you can do anything, you must set the global GPU/device to your GPU

```js
import {Tensor} from "./tensor-webgpu";
import {GPU} from "./webgpu-compute";
Tensor.setDevice(await GPU.init());
```

Then go wild!


## Tensor Data

### Tensor.tensor

Initialized GPU Tensor given array, dtype and shape.

In this case I want to send the vector `[1,2,3]` to the gpu with shape `[3,1]`.

```js
const a = await Tensor.tensor([1,2,3], [3,1], "f32");
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
const a = await Tensor.random([4, 1], "f32");
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
const a = await Tensor.fill(1, [2, 2, 2], "u32");
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

### cpuBuffer()

Asks the GPU to return the data to the CPU. Returns as a JS array given a tensor `a` in this case;

```js
const arr = await a.cpuBuffer();
```

### contiguous()

Deep copies the tensor and makes sure the memory is contiguous with how it's printing. 

```js
const a = await Tensor.tensor([1, 2, 3, 4], [2, 2]);
const aT = a.T;

console.log("a");
await a.print();

console.log("a.T");
await aT.print();

console.log("a.T buffer", await aT.cpuBuffer());
console.log("a.T.contiguous() buffer", await (await aT.contiguous()).cpuBuffer());
```

`console outputs ↓`

```js
a
dtype='f32', shape=[2,2], strides=[2,1],
gpuBuffer=
[[1, 2],
 [3, 4]]

a.T
dtype='f32', shape=[2,2], strides=[1,2],
gpuBuffer=
[[1, 3],
 [2, 4]]

a.T buffer Float32Array(4) [1, 2, 3, 4, buffer: ArrayBuffer(16), byteLength: 16, byteOffset: 0, length: 4, Symbol(Symbol.toStringTag): 'Float32Array']

a.T.contiguous() buffer Float32Array(4) [1, 3, 2, 4, buffer: ArrayBuffer(16), byteLength: 16, byteOffset: 0, length: 4, Symbol(Symbol.toStringTag): 'Float32Array']
```


## Tensor Operations

### matmul(other)

Matrix multiplication between two matrices. Must have 2D shape and matching inner dimension!

```js
const a = await Tensor.tensor([1, 2, 3, 4, 5, 6], [2, 3]);
const b = await Tensor.tensor([0, 1, 2, 3, 4, 5], [3, 2]);
const c = await a.matmul(b);

await a.print();
await b.print();

console.log("GPU RESULT");
await c.print();
```

`console outputs ↓`

```js
dtype='f32', shape=[2,3], strides=[3,1],
gpuBuffer=
[[1, 2, 3],
 [4, 5, 6]]

dtype='f32', shape=[3,2], strides=[2,1],
gpuBuffer=
[[0, 1],
 [2, 3],
 [4, 5]]

GPU RESULT
dtype='f32', shape=[2,2], strides=[2,1],
gpuBuffer=
[[16, 22],
 [34, 49]]
```

### pow(other)

Raises tensor to the power of all entries elementwise in other.

```js
const a = await Tensor.tensor([1, 2, -3], [3, 1], "f32");
const b = await Tensor.fill(2, a.shape);
const c = await a.pow(b);

console.log("a");
await a.print();

console.log("b");
await b.print();

console.log("c=a^b");
await c.print();
```

`console outputs ↓`

```js
a
dtype='f32', shape=[3,1], strides=[1,1],
gpuBuffer=
[[1],
 [2],
 [-3]]

b
dtype='f32', shape=[3,1], strides=[1,1],
gpuBuffer=
[[2],
 [2],
 [2]]

c=a^b
dtype='f32', shape=[3,1], strides=[1,1],
gpuBuffer=
[[1],
 [4],
 [9]]
```

### div(other)

Divides two tensors.

```js
const a = await Tensor.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
const b = await Tensor.fill(2, a.shape);
const c = await a.div(b);

console.log("a");
await a.print();

console.log("b");
await b.print();

console.log("c = a/b");
await c.print();
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

b
dtype='f32', shape=[2,2,2], strides=[4,2,1],
gpuBuffer=
[[[2, 2],
  [2, 2]],

 [[2, 2],
  [2, 2]]]

c = a/b
dtype='f32', shape=[2,2,2], strides=[4,2,1],
gpuBuffer=
[[[0, 0.5],
  [1, 1.5]],

 [[2, 2.5],
  [3, 3.5]]]
```
### mul(other)

Multiplies two tensors.

```js
const a = await Tensor.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
const b = await Tensor.fill(-1, a.shape);
const c = await a.mul(b);

console.log("a");
await a.print();

console.log("b");
await b.print();

console.log("c = a*b");
await c.print();
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

b
dtype='f32', shape=[2,2,2], strides=[4,2,1],
gpuBuffer=
[[[-1, -1],
  [-1, -1]],

 [[-1, -1],
  [-1, -1]]]

c = a*b
dtype='f32', shape=[2,2,2], strides=[4,2,1],
gpuBuffer=
[[[0, -1],
  [-2, -3]],

 [[-4, -5],
  [-6, -7]]]
```

### sub(other)

Subtracts two tensors.

```js
const a = await Tensor.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
const b = await Tensor.fill(1, a.shape);
const c = await a.sub(b);

console.log("a");
await a.print();

console.log("b");
await b.print();

console.log("c = a-b");
await c.print();
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

gpu.js:614 b
dtype='f32', shape=[2,2,2], strides=[4,2,1],
gpuBuffer=
[[[1, 1],
  [1, 1]],

 [[1, 1],
  [1, 1]]]

c = a-b
dtype='f32', shape=[2,2,2], strides=[4,2,1],
gpuBuffer=
[[[-1, 0],
  [1, 2]],

 [[3, 4],
  [5, 6]]]
```

### add(other) 

Adds together two tensors.

```js
const a = await Tensor.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
const b = await Tensor.fill(1, a.shape);
const c = await a.add(b);

console.log("a");
await a.print();

console.log("b");
await b.print();

console.log("c = a+b");
await c.print();
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

b
dtype='f32', shape=[2,2,2], strides=[4,2,1],
gpuBuffer=
[[[1, 1],
  [1, 1]],

 [[1, 1],
  [1, 1]]]

c = a+b
dtype='f32', shape=[2,2,2], strides=[4,2,1],
gpuBuffer=
[[[1, 2],
  [3, 4]],

 [[5, 6],
  [7, 8]]]
```

### sum(dim)

Sums over the given axis/dimension. 

```js
const a = await Tensor.tensor([0,1,2,3,4,5,6,7], [2,2,2], "f32");
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

### transpose()


Transposes the first and last dimension.

Or alias `.T` does the same thing

```js
const a = await Tensor.tensor([1, 2, 3], [3, 1]);
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

Instead you could explicitly use the functional call `Tensor.<op name>(destination,...args)`;

```js
const aSum = await Tensor.empty([1,1]); // destination/result empty allocation
await Tensor.sum(aSum, a, 0); // compute sum down a and store in aSum
```

This function API also allows for in place operations. Like squaring

```js
await Tensor.pow(a, a, 2); // compute a^2 then override a with result 
```



## Dev

```bash
cd example
pnpm install
pnpm dev
```
