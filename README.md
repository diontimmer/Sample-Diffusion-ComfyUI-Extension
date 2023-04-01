
# Sample Diffusion ComfyUI extension


## Features
Allows the use of trained [dance diffusion/sample generator](https://github.com/Harmonai-org/sample-generator) models in ComfyUI.<br>
Also included are two optional extensions of the extension (lol); Wave Generator for creating primitive waves aswell as a wrapper for the Pedalboard library.<br>
The pedalboard wrapper allows us to wrap most vst3s and control them, for now only a wrapper for OTT is included. Any suggestions are welcome.<br>
Includes a couple helper functions.

## Installation
1. Install [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
2. Clone/download as zip and place ```SampleDiffusion``` folder in ```ComfyUI/custom_nodes```.
3. Move/copy playAudio.js from ```SampleDiffusion``` to ```ComfyUI/web/extensions```
4. Place models in ```ComfyUI/models/audio_diffusion``` ('audio_diffusion' entry in extra_model_paths.yaml is accepted).
4.5 (Optional) Install [xfer OTT VST3](https://xferrecords.com/freeware)
5. Launch!

## Example

(Tip: right-click a node to convert an input to a pin you can connect to!)

![App Screenshot](https://i.imgur.com/cxNlYpU.png)

Feel free to check out my [other nodes](https://github.com/diontimmer/ComfyUI-Vextra-Nodes).

## Acknowledgements

 - [sample-diffusion](https://github.com/sudosilico/sample-diffusion)
 - [pythongosssss](https://github.com/pythongosssss) for the preview audio node & javascript magic.
 - [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
 - [Harmonai](https://github.com/Harmonai-org/sample-generator)
 - [pedalboard](https://github.com/spotify/pedalboard)
 - [xfer OTT](https://xferrecords.com/freeware)
