<script>
	import { onMount } from "svelte";
	import { main } from "../../example";
	onMount(async () => {
		await main();
	});
</script>

<div>DEV MODE CHECK CONSOLE AND CHANGE EXAMPLE.JS MAIN</div>
