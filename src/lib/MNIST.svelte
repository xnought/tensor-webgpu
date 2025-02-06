<script>
	import { onMount } from "svelte";
	import { Tensor } from "../../tensorscript";
	import { Lazy, OptimSGD } from "../../lazy";
	import LossVis from "./LossVis.svelte";
	import MnistDigit from "./digit/MnistDigit.svelte";
	import * as d3 from "d3";

	let loadingData = false;
	async function fetchMnist10k() {
		loadingData = true;
		const d = await fetch("mnist_test.json");
		const j = await d.json();
		const x = j["x"],
			y = j["y"];
		loadingData = false;
		return [x, y];
	}

	function linearWeights(inNeurons, outNeurons) {
		const mean = 0;
		const stdev = 0.1;
		const w = Lazy.tensor(Tensor.randn([inNeurons, outNeurons], mean, stdev), true);
		const b = Lazy.tensor(Tensor.randn([1, outNeurons], mean, stdev), true);
		return [w, b];
	}

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

	const batchSize = 64;
	const lr = 0.1;
	const maxEpochs = 10;
	let lossHistory = [];
	let deviceInfo;
	let trainingFinished = false;
	let yhat, x;
	onMount(async () => {
		const adapter = await navigator.gpu.requestAdapter();
		const device = await adapter.requestDevice();
		deviceInfo = { vendor: device.adapterInfo.vendor, architecture: device.adapterInfo.architecture };
		Tensor.setDevice(device);

		const [xCpu, yCpu] = await fetchMnist10k();
		x = Lazy.tensor(Tensor.fill(1, [batchSize, 28 * 28]));
		let params;
		[yhat, params] = createClassifierMLP(x, [28 * 28, 256, 128], batchSize);
		const y = Lazy.tensor(Tensor.fill(1, [batchSize, 10]));
		const loss = lossCCE(yhat, y, batchSize);
		const optim = new OptimSGD(params, lr);

		const getBatch = (cpu, i) => flat2D(cpu.slice(i, i + batchSize));
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
					lossHistory.push(l[0]);
					lossHistory = lossHistory;
				}
			}
		}
		trainingFinished = true;
	});
	let drawnDigit = new Float32Array(28 * 28).fill(0);
	let updateCounter = 0;
	let confs;
	async function updateDraw(d, skip = 5) {
		drawnDigit = d;
		if (updateCounter % skip === 0) {
			if (yhat && x) {
				x.tensor.setGPUBuffer(drawnDigit);
				yhat.forward();
				confs = (await yhat.tensor.cpuBuffer()).slice(0, 10);
			}
		}
		updateCounter++;
	}
	function _argmax(arr) {
		let idx = 0;
		let gmax = arr[idx];
		for (let i = 0; i < arr.length; i++) {
			if (arr[i] > gmax) {
				gmax = arr[i];
				idx = i;
			}
		}
		return idx;
	}
</script>

{#if loadingData}
	<div>Loading MNIST Test 10k data...</div>
{/if}
{#if deviceInfo}
	<div>Running on {JSON.stringify(deviceInfo, null, 4)}</div>
{/if}
{#if !loadingData && !trainingFinished}
	<div>Training MLP on MNIST 10k Images</div>
{/if}
<div>
	<LossVis loss={lossHistory} />
</div>

{#if trainingFinished}
	<div>
		<button
			on:click={() => {
				confs = undefined;
				drawnDigit = new Float32Array(28 * 28).fill(0);
			}}>Clear</button
		>
	</div>
	<div>
		<MnistDigit data={drawnDigit} maxVal={1} enableDrawing onChange={async (d) => await updateDraw(d)} />
	</div>
	{#if confs}
		<div>Predicted Digit: {_argmax(confs)}</div>
		<div style="width: 300px;">
			{#each confs as c, i}
				<div style="width: {c * 300}px; height: 20px; background: {d3.schemeCategory10[i]};">{i}</div>
			{/each}
		</div>
	{/if}
{/if}

<style>
	/*  put stuff here */
</style>
