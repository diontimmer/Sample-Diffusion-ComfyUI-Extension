// https://github.com/pythongosssss/ComfyUI-Custom-Scripts/blob/main/play-audio/playAudio.js

import { app } from "/scripts/app.js";

app.registerExtension({
	name: "pysssss.PlayAudio",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		const AudioPreviews = ["PreviewAudioFile", "PreviewAudioTensor"]
		if (AudioPreviews.includes(nodeData.name)) {
			const WIDGETS = Symbol();
			nodeType.prototype.onExecuted = function (data) {
				if (WIDGETS in this) {
					// Clear all other widgets
					if (this.widgets) {
						this.widgets.length = this[WIDGETS];
					}
					if (this.widgets_values) {
						this.widgets_values.length = this.widgets.length;
					}
				} else {
					// On first execute store widget count
					this[WIDGETS] = this.widgets?.length || 0;
				}

				// For each file create a seek bar + play button
				for(let i = 0; i < data.length; i++) { 
					const file = data[i];
					let isTick = true;
					const audio = new Audio(`/view?type=temp&filename=${encodeURIComponent(file)}`);
					const slider = this.addWidget(
						"slider",
						"loading",
						0,
						(v) => {
							if (!isTick) {
								audio.currentTime = v;
							}
							isTick = false;
						},
						{
							min: 0,
							max: 0,
						}
					);

					const button = this.addWidget("button", `Play Batch #${i}`, "play", () => {
						try {
							if (audio.paused) {
								audio.play();
								button.name = `Pause Batch #${i}`;
							} else {
								audio.pause();
								button.name = `Play Batch #${i}`;
							}
						} catch (error) {
							alert(error);
						}
						app.canvas.setDirty(true);
					});
					audio.addEventListener("timeupdate", () => {
						isTick = true;
						slider.value = audio.currentTime;
						app.canvas.setDirty(true);
					});
					audio.addEventListener("ended", () => {
						button.name = `Play Batch #${i}`;
						app.canvas.setDirty(true);
					});
					audio.addEventListener("loadedmetadata", () => {
						slider.options.max = audio.duration;
						slider.name = `(${audio.duration})`;
						app.canvas.setDirty(true);
					});
				}
			};
		}
	},
});
