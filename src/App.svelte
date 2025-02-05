<script>
	import Dev from "./lib/Dev.svelte";
	import Mnist from "./lib/MNIST.svelte";

	const DEV_MODE = false;
</script>

<main>
	{#if !DEV_MODE}
		<Mnist />
	{:else}
		<Dev />
	{/if}
</main>

<style>
</style>
