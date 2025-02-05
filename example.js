import { Tensor } from "./tensorscript";
import { Lazy, OptimSGD } from "./lazy";

main();

async function main() {
	const adapter = await navigator.gpu.requestAdapter();
	const device = await adapter.requestDevice();
	Tensor.setDevice(device);

	// await maxExample();
	// await mnistExampleOneBatch();
	// await bmmExample();
	// await softmaxJacobianExample();
	// await logExample();
	// await softmaxBackwardExample();
	// await reluBackwardExample();
	await mnistExample();
	// await reluExample();
	// await softmaxExample();
	// await linearRegressionInterceptExample();
	// await mulBackwardExample();
	// await scalarForwardExample();
	// await numberBinaryOpExample();
	// await expandExample2();
	// await sumGradExample();
	// await linearRegressionExample();
	// await expandExample();
	// await unsqueezeExample();
	// await copyExample();
	// await divExample();
	// await matmulExample3();
	// await matmulExample2();
	// await powExample();
	// await mulExample();
	// await subExample();
	// await addExample();
	// await sumExample();
	// await inverseIndexing();
	// await inverseIndexing31();
}

async function maxExample() {
	const a = Tensor.tensor([1, 2, 3, 5, -5, 0], [2, 3]);
	await a.max(0).print();
}

async function bmmExample() {
	const a = Tensor.tensor([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6], [2, 2, 3]);
	const b = Tensor.tensor([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6], [2, 3, 2]);
	await a.print();
	await b.print();

	const a0 = Tensor.tensor([1, 2, 3, 4, 5, 6], [2, 3]);
	const b0 = Tensor.tensor([1, 2, 3, 4, 5, 6], [3, 2]);
	await a0.print();
	await b0.print();
	await a0.matmul(b0).print();

	await a.bmm(b).print();
}

async function softmaxJacobianExample() {
	const a = Tensor.tensor([0, 1, -1, 1, 2, 3], [2, 3]);
	const s = a.softmax(-1);
	const dst = Tensor.empty([...a.shape, 3]);
	console.log("a");
	await a.print();
	console.log("softmax(a)");
	await s.print();
	Tensor._softmaxJacobianLastDim(dst, s);
	console.log("dsoftmax(a)");
	await dst.print();
	await Tensor.fill(10, [2, 1, 3]).bmm(dst).print();
}

async function logExample() {
	const a = Tensor.tensor([0.1, 2, 3], [3, 1]);
	await a.log().print();
}

async function softmaxBackwardExample() {
	const x = Lazy.tensor(Tensor.tensor([-1, 1, 0], [1, 3]), true);
	const softmax = x.softmax(-1);
	const summed = softmax.sum(-1);
	summed.forward();
	await softmax.print();
	summed.backward();
	await x.grad.print();
}
async function reluBackwardExample() {
	const x = Lazy.tensor(Tensor.tensor([-1, 1, 0], [3, 1]), true);
	const relu = x.relu();
	const summed = relu.sum(0).mul(2);
	summed.forward();
	summed.backward();
	await x.grad.print();
}

function linearWeights(inNeurons, outNeurons) {
	const mean = 0;
	const stdev = 0.1;
	const w = Lazy.tensor(Tensor.randn([inNeurons, outNeurons], mean, stdev), true);
	const b = Lazy.tensor(Tensor.randn([1, outNeurons], mean, stdev), true);
	return [w, b];
}

/**
 * Relu model mnist MLP
 * @param {Lazy} x
 * @param {number[]} layers
 * @param {number} batches
 * @param {number} classes
 * @returns {Lazy}
 */
function createClassifierMLP(x, layers = [728, 128], batches = 32, classes = 10) {
	let parameters = [];
	for (let i = 0; i < layers.length - 1; i++) {
		const [w, b] = linearWeights(layers[i], layers[i + 1]);
		parameters.push(w);
		parameters.push(b);
		x = x
			.matmul(w)
			.add(b.expand([batches, layers[i + 1]]))
			.relu();
	}
	const [w, b] = linearWeights(layers.at(-1), classes);
	parameters.push(w);
	parameters.push(b);
	const logits = x.matmul(w).add(b.expand([batches, classes]));
	const probs = logits.softmax(-1);
	return [probs, parameters];
}

/**
 * y must be onehot encoded
 * @param {Lazy} yhat
 * @param {Lazy} y
 */
function lossCCE(yhat, y, batchSize = 32) {
	return yhat
		.log()
		.mul(y)
		.sum(-1)
		.sum(0)
		.mul(-1 / batchSize);
}

function flat2D(arr2D) {
	const out = new Float32Array(arr2D.length * arr2D[0].length);
	for (let i = 0; i < arr2D.length; i++) {
		for (let j = 0; j < arr2D[0].length; j++) {
			out[i * arr2D[0].length + j] = arr2D[i][j];
		}
	}
	return out;
}

async function fetchMnist10k() {
	const d = await fetch("mnist_test.json");
	const j = await d.json();
	const x = j["x"],
		y = j["y"];
	return [x, y];
}
async function mnistExampleOneBatch() {
	const batchSize = 64;
	const x = Lazy.tensor(Tensor.fill(1, [batchSize, 28 * 28]));
	const [yhat, params] = createClassifierMLP(x, [28 * 28, 128, 128], batchSize);
	const y = Lazy.tensor(Tensor.fill(1, [batchSize, 10]));
	const loss = lossCCE(yhat, y, batchSize);
	const lr = 0.1;
	const optim = new OptimSGD(params, lr);

	const [xCpu, yCpu] = await fetchMnist10k();
	const getBatch = (cpu, i) => flat2D(cpu.slice(i, i + batchSize));
	x.tensor.setGPUBuffer(getBatch(xCpu, 0));
	y.tensor.setGPUBuffer(getBatch(yCpu, 0));

	for (let i = 0; i < 100; i++) {
		// forward through the model
		loss.forward();
		// backprop and update
		loss.zeroGrad();
		loss.backward();
		optim.update();

		const l = await loss.tensor.cpuBuffer();
		console.log("loss", l[0]);
	}
}
async function mnistExample() {
	const batchSize = 64;
	const x = Lazy.tensor(Tensor.fill(1, [batchSize, 28 * 28]));
	const [yhat, params] = createClassifierMLP(x, [28 * 28, 256, 128], batchSize);
	const y = Lazy.tensor(Tensor.fill(1, [batchSize, 10]));
	const loss = lossCCE(yhat, y, batchSize);
	const lr = 1e-2;
	const optim = new OptimSGD(params, lr);

	const [xCpu, yCpu] = await fetchMnist10k();
	const getBatch = (cpu, i) => flat2D(cpu.slice(i, i + batchSize));
	let maxEpochs = 5;
	for (let epoch = 0; epoch < maxEpochs; epoch++) {
		console.log("EPOCH" + (epoch + 1));
		for (let i = 0; i < xCpu.length - batchSize; i += batchSize) {
			x.tensor.setGPUBuffer(getBatch(xCpu, i));
			y.tensor.setGPUBuffer(getBatch(yCpu, i));

			// forward through the model
			loss.forward();

			// backprop and update
			loss.zeroGrad();
			loss.backward();
			optim.update();

			if (i % 10 === 0) {
				const l = await loss.tensor.cpuBuffer();
				console.log("loss", l[0]);
			}
		}
	}
}

async function reluExample() {
	const a = Tensor.tensor([-1, 2, 3], [3, 1]);
	await a.relu().print();
}

async function softmaxExample() {
	const a = Tensor.tensor([1, 2, 3], [3, 1]);
	await a.softmax(0).print();
}

async function mulBackwardExample() {
	const y = Lazy.tensor(Tensor.tensor([1, 2, 3, 4], [4, 1]), true);
	await y.print();

	const mean = y.sum(0).mul(1 / 4);

	mean.forward();
	await mean.print();

	mean.backward();
	await y.grad.print();
}
async function scalarForwardExample() {
	const y = Lazy.tensor(Tensor.tensor([1, 2, 3, 4], [4, 1]));
	await y.print();

	const square = y.square();
	const summed = square.sum(0);

	summed.forward();

	await square.print();
	await summed.print();

	summed.backward();
	await y.grad.print();
}
async function sumGradExample() {
	const y = Lazy.tensor(Tensor.tensor([1, 2, 3, 4], [4, 1]));
	const yhat = Lazy.tensor(Tensor.tensor([1, 0, 3, 1], [4, 1]));
	const loss = y.add(yhat).sum(0);
	loss.forward(); // now compute everything (the c = a + b)
	loss.backward();
	await loss.grad.print();
}

async function divExample() {
	const a = Tensor.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
	const b = Tensor.fill(2, a.shape);
	const c = a.div(b);

	console.log("a");
	await a.print();

	console.log("b");
	await b.print();

	console.log("c = a/b");
	await c.print();
}
async function mulExample() {
	const a = Tensor.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
	const b = Tensor.fill(-1, a.shape);
	const c = a.mul(b);

	console.log("a");
	await a.print();

	console.log("b");
	await b.print();

	console.log("c = a*b");
	await c.print();
}

async function subExample() {
	const a = Tensor.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
	const b = Tensor.fill(1, a.shape);
	const c = a.sub(b);

	console.log("a");
	await a.print();

	console.log("b");
	await b.print();

	console.log("c = a-b");
	await c.print();
}

async function addExample() {
	const a = Tensor.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
	const b = Tensor.fill(1, a.shape);
	const c = a.add(b);

	console.log("a");
	await a.print();

	console.log("b");
	await b.print();

	console.log("c = a+b");
	await c.print();
}

async function sumExample() {
	const a = Tensor.tensor([0, 1, 2, 3, 4, 5, 6, 7], [2, 2, 2], "f32");
	console.log("a");
	await a.print();

	console.log("Sum across");
	await a.sum(-1).print();
}
async function transposeExample() {
	const a = Tensor.tensor([1, 2, 3], [3, 1]);
	console.log("a");
	await a.print();

	console.log("a.transpose()");
	await a.transpose().print();

	console.log("a.T");
	await a.T.print();
}
async function powExample() {
	const a = Tensor.tensor([1, 2, -3], [3, 1], "f32");
	const b = Tensor.fill(2, a.shape);
	const c = a.pow(b);

	console.log("a");
	await a.print();

	console.log("b");
	await b.print();

	console.log("c=a^b");
	await c.print();
}
async function randomExample() {
	const a = Tensor.random([4, 1], "f32");
	await a.print();
}
async function fillExample() {
	const a = Tensor.fill(1, [2, 2, 2], "u32");
	await a.print();
}

async function inverseIndexing() {
	const a = Tensor.tensor([0, 1, 2, 3, 4, 5, 6, 7], [2, 2, 2], "f32");
	const c = a.transpose([0, 2, 1]);

	await c.print();
	for (let i = 0; i < 4; i++) {
		const idx = [Math.floor(i / c.shape[1]) % c.shape[0], Math.floor(i) % c.shape[1]];
		const wgslIdx = wgslBaseIdx(c.shape, c.strides, `${i}`, -2, "Math.floor");
		const a = eval(wgslIdx);

		console.log(idx[0] * c.strides[0] + idx[1] * c.strides[1]);
		console.log(a);
	}
}

async function inverseIndexing31() {
	const c = Tensor.fill(1, [3, 2], "f32");
	c.print();

	for (let i = 0; i < 6; i++) {
		const idx = [Math.floor(i / c.shape[1]) % c.shape[0], Math.floor(i) % c.shape[1]];
		const wgslIdx = wgslBaseIdx(c.shape, c.strides, `${i}`, -1, "Math.floor");
		const a = eval(wgslIdx);

		console.log(idx[0] * c.strides[0] + idx[1] * c.strides[1], a);
		// console.log(idx);
	}
}

async function matmulExample3() {
	const a = Tensor.tensor([1, 2, 3, 4, 5, 6], [2, 3]);
	const b = Tensor.tensor([0, 1, 2, 3, 4, 5], [3, 2]);
	const c = a.matmul(b);

	await a.print();
	await b.print();

	console.log("GPU RESULT");
	await c.print();
}

async function matmulExample2() {
	const shape = [784, 784];
	const a = Tensor.fill(1, shape);
	const b = Tensor.fill(1, shape);
	const c = Tensor.empty(shape);
	Tensor.matmul(c, a, b);

	await a.print();
	await b.print();

	console.log("GPU RESULT");
	await c.print();

	cpuMatmul: {
		const acpu = await a.cpuBuffer();
		const bcpu = await b.cpuBuffer();
		const ccpu = new Float32Array(length(c.shape));
		const m = a.shape[0];
		const n = a.shape[1];
		const l = b.shape[1];
		for (let i = 0; i < m; i++) {
			for (let j = 0; j < l; j++) {
				let cidx = i * c.strides[0] + j * c.strides[1];
				for (let k = 0; k < n; k++) {
					let aidx = i * a.strides[0] + k * a.strides[1];
					let bidx = k * b.strides[0] + j * b.strides[1];
					ccpu[cidx] += acpu[aidx] * bcpu[bidx];
				}
			}
		}
		console.log("CPU COMPUTED ACTUAL RESULT!");
		console.log(ndarrayToString(ccpu, c.shape, c.strides, 8));
	}
}
async function matmulExample() {
	const a = Tensor.tensor([1, 2, 3, 4, 5, 6], [2, 3]);
	const b = Tensor.tensor([0, 1, 2, 3, 4, 5], [3, 2]);
	const c = Tensor.empty([a.shape[0], b.shape[1]]);
	Tensor.matmul(c, a, b);

	await a.print();
	await b.print();

	console.log("GPU RESULT");
	await c.print();

	cpuMatmul: {
		const acpu = await a.cpuBuffer();
		const bcpu = await b.cpuBuffer();
		const ccpu = [0, 0, 0, 0];
		const m = a.shape[0];
		const n = a.shape[1];
		const l = b.shape[1];
		for (let i = 0; i < m; i++) {
			for (let j = 0; j < l; j++) {
				let cidx = i * c.strides[0] + j * c.strides[1];
				for (let k = 0; k < n; k++) {
					let aidx = i * a.strides[0] + k * a.strides[1];
					let bidx = k * b.strides[0] + j * b.strides[1];
					ccpu[cidx] += acpu[aidx] * bcpu[bidx];
				}
			}
		}
		console.log("CPU COMPUTED ACTUAL RESULT!");
		console.log(ndarrayToString(ccpu, c.shape, c.strides));
	}
}

async function copyExample() {
	const a = Tensor.tensor([1, 2, 3, 4], [2, 2]);
	const aT = a.T;

	console.log("a");
	await a.print();

	console.log("a.T");
	await aT.print();

	console.log("a.T buffer", await aT.cpuBuffer());
	console.log();
	console.log("a.T.contiguous() buffer", await aT.contiguous().cpuBuffer());
}

async function linearRegressionInterceptExample() {
	const n = 100;
	const line = Array(n)
		.fill(0)
		.map((_, i) => i / n);

	// Graph model functional spec
	const dshape = [n, 1];
	const x = Lazy.tensor(Tensor.tensor(line, dshape));
	const y = Lazy.tensor(Tensor.tensor(line, dshape));
	const w = Lazy.tensor(Tensor.tensor([-5], [1, 1]), /*requiresGrad=*/ true);
	const b = Lazy.tensor(Tensor.tensor([1], [1, 1]), /*requiresGrad=*/ true);

	const yhat = x.matmul(w).add(b.expand(dshape));
	const loss = yhat
		.sub(y)
		.square()
		.sum(0)
		.mul(1 / n); // mse loss

	const iterations = 100;
	const lr = 2;
	const optim = new OptimSGD([w, b], lr);
	for (let i = 0; i < iterations; i++) {
		console.time("iteration" + i);
		loss.forward();
		loss.zeroGrad();
		loss.backward();
		optim.update();
		console.timeEnd("iteration" + i);
	}

	console.log("W");
	await w.print();
	console.log("B");
	await b.print();
	console.log("LOSS");
	await loss.print();

	loss.freeGraph();
}
async function linearRegressionExample() {
	const n = 1_000;
	const line = Array(n)
		.fill(0)
		.map((_, i) => i / n);

	// Graph model functional spec
	const x = Lazy.tensor(Tensor.tensor(line, [1, n]));
	const y = Lazy.tensor(Tensor.tensor(line, [1, n]));
	const w = Lazy.tensor(Tensor.tensor([0], [1, 1]), /*requiresGrad=*/ true);
	const yhat = x.T.matmul(w);
	const loss = yhat
		.sub(y.T)
		.square()
		.sum(0)
		.mul(1 / n); // mse loss

	const iterations = 100;
	const lr = 0.1;
	const optim = new OptimSGD([w], lr);
	for (let i = 0; i < iterations; i++) {
		console.time("iteration" + i);
		loss.forward();
		loss.zeroGrad();
		loss.backward();
		optim.update();
		console.timeEnd("iteration" + i);
	}

	console.log("W");
	await w.print();
	console.log("LOSS");
	await loss.print();

	loss.freeGraph();
}

async function unsqueezeExample() {
	const a = Tensor.tensor([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
	await a.print();
	await a.unsqueeze(0).print();
}

async function expandExample() {
	const a = Tensor.tensor([1, 2, 3], [1, 3]);
	await a.print();
	console.log("expand the first dimension to 3");
	await a.expandTo(3, 0).print();

	console.log("useful example");
	let scalar = Tensor.tensor([7], [1, 1]);
	const tensor = Tensor.tensor([1, 2, 3, 4], [2, 2]);

	console.log("scalar");
	await scalar.print();
	console.log("tensor");
	await tensor.print();

	// scalar.mul(tensor) fails since tensor.shape != scalar.shape
	// instead expand to shape [2,2] ↓
	scalar = scalar.expandTo(2, 0).expandTo(2, 1);

	console.log("scalar expanded");
	await scalar.print();

	console.log("scalar*tensor now works");
	const result = await scalar.mul(tensor);
	await result.print();
}

async function expandExample2() {
	let scalar = Tensor.tensor([7], [1, 1]);
	const tensor = Tensor.tensor([1, 2, 3, 4], [2, 2]);

	console.log("scalar");
	await scalar.print();
	console.log("tensor");
	await tensor.print();

	console.log("scalar expanded");
	await scalar.expand(tensor.shape).print();

	console.log("scalar*tensor now works");
	const result = scalar.expand(tensor.shape).mul(tensor);
	await result.print();
}

async function numberBinaryOpExample() {
	const tensor = Tensor.tensor([1, 2, 3, 4], [2, 2]);
	await tensor.print();
	await tensor.div(2).print();
	await tensor.mul(2).print();
	await tensor.add(2).print();
	await tensor.pow(2).print();
}
