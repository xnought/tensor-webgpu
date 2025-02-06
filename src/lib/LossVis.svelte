<script>
	import { scaleLinear } from "d3";

	export let height = 250;
	export let width = 500;
	export let loss = [];

	function _max(arr) {
		if (arr.length === 0) return 0;

		let gmax = arr[0];
		for (let i = 0; i < arr.length; i++) {
			if (arr[i] > gmax) {
				gmax = arr[i];
			}
		}
		return gmax;
	}

	$: xScale = scaleLinear()
		.domain([0, loss.length - 1])
		.range([0, width]);
	$: yScale = scaleLinear()
		.domain([0, _max(loss)])
		.range([height, 0]);
</script>

<div>
	<div>Loss History</div>
	<div>
		<svg {width} {height} style="overflow: visible; outline: 1px solid black;">
			{#if loss.length > 0}
				{#each { length: loss.length - 1 } as _, i}
					{@const x1 = i}
					{@const x2 = i + 1}
					{@const y1 = loss[x1]}
					{@const y2 = loss[x2]}
					<line x1={xScale(x1)} x2={xScale(x2)} y1={yScale(y1)} y2={yScale(y2)} stroke="black" stroke-width={2}></line>
				{/each}
				<circle fill="black" r={4} cx={width} cy={yScale(loss.at(-1))}></circle>
				<text x={width + 4} y={yScale(loss.at(-1))}>{loss.at(-1).toFixed(5)}</text>
			{/if}
		</svg>
	</div>
</div>
