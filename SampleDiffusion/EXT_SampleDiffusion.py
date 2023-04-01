# Imports

import subprocess, sys, os
import torch
import random
from folder_paths import folder_names_and_paths, models_dir, supported_pt_extensions, get_filename_list, get_full_path
try:
    import soundfile
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "soundfile"])

try:
    import torchaudio
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torchaudio"])
    import torchaudio

def get_comfy_dir():
    dirs = __file__.split('\\')
    comfy_index = None
    for i, dir in enumerate(dirs):
        if dir == "ComfyUI":
            comfy_index = i
            break
    if comfy_index is not None:
        # Join the list up to the "ComfyUI" folder
        return '\\'.join(dirs[:comfy_index+1])
    else:
        return None


# init and sample_diffusion lib load

folder_names_and_paths["audio_diffusion"] = ([os.path.join(models_dir, "audio_diffusion")], supported_pt_extensions)


comfy_dir = get_comfy_dir()   
if not os.path.exists(os.path.join(comfy_dir, 'custom_nodes/SampleDiffusion/libs')):
    os.makedirs(os.path.join(comfy_dir, 'custom_nodes/SampleDiffusion/libs'))
lib = os.path.join(comfy_dir, 'custom_nodes/SampleDiffusion/libs/sample_generator') 
if not os.path.exists(os.path.join(comfy_dir, lib)):
    os.system(f'git clone https://github.com/sudosilico/sample-diffusion.git {os.path.join(comfy_dir, lib)}')
sys.path.append(os.path.join(comfy_dir, lib))
from util.util import load_audio
from util.platform import get_torch_device_type
from dance_diffusion.api import RequestHandler, Request, SamplerType, SchedulerType, ModelType

def save_audio(audio_out, output_path: str, sample_rate, id_str:str = None):
    out_files = []
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for ix, sample in enumerate(audio_out):
        output_file = os.path.join(output_path, f"sample_{id_str}_{ix + 1}.wav" if(id_str!=None) else f"sample_{ix + 1}.wav")
        open(output_file, "a").close()
        
        output = sample.cpu()
        torchaudio.save(output_file, output, sample_rate)
        out_files.append(output_file)
    return out_files


# ****************************************************************************
# *                                   NODES                                  *
# ****************************************************************************





class AudioInference():
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        Input Types
        """
        return {
            "required": {
                "model": (get_filename_list("audio_diffusion"), ),
                "mode": (['Generation', 'Variation'],),
                "chunk_size": ("INT", {"default": 65536, "min": 32768, "max": 10000000000, "step": 32768}),
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 10000000000, "step": 1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 10000000000, "step": 1}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 10000000000, "step": 1}),
                "sampler": (SamplerType._member_names_, {"default": "IPLMS"}),
                "scheduler": (SchedulerType._member_names_, {"default": "CrashSchedule"}),
                "noise_level": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": -1}),
                },
            "optional": {
                "input_audio_path": ("STRING", {"default": '', "forceInput": True}),
                },
            }

    RETURN_TYPES = ("LIST", "AUDIO", "INT")
    RETURN_NAMES = ("out_paths", "tensor", "sample_rate")
    FUNCTION = "do_sample"

    CATEGORY = "Audio/SampleDiffusion"

    def do_sample(self, model, mode, chunk_size, sample_rate, batch_size, steps, sampler, scheduler, input_audio_path='', noise_level=0.7, seed=-1):
        model = get_full_path('audio_diffusion', model)
        device_type_accelerator = get_torch_device_type()
        device_accelerator = torch.device(device_type_accelerator)
        device_offload = torch.device('cuda')
        input_audio_path = None if input_audio_path == '' else input_audio_path
        crop = lambda audio: audio
        load_input = lambda source: crop(load_audio(device_accelerator, source, sample_rate)) if source is not None else None
        
        request_handler = RequestHandler(device_accelerator, device_offload, optimize_memory_use=False, use_autocast=True)
        
        seed = seed if(seed!=-1) else torch.randint(0, 4294967294, [1], device=device_type_accelerator).item()
        print(f"Using accelerator: {device_type_accelerator}, Seed: {seed}.")
        
        request = Request(
            request_type=mode,
            model_path=model,
            model_type=ModelType.DD,
            model_chunk_size=chunk_size,
            model_sample_rate=sample_rate,
            
            seed=seed,
            batch_size=batch_size,
            
            audio_source=load_input(input_audio_path),
            audio_target=None,
            
            mask=None,
            
            noise_level=noise_level,
            interpolation_positions=None,
            resamples=None,
            keep_start=True,
                    
            steps=steps,
            
            sampler_type=sampler,
            sampler_args={'use_tqdm': True},
            
            scheduler_type=scheduler,
            scheduler_args={}
        )
        
        response = request_handler.process_request(request)#, lambda **kwargs: print(f"{kwargs['step'] / kwargs['x']}"))
        paths = save_audio(response.result, f"{comfy_dir}/temp", sample_rate, f"{seed}_{random.randint(0, 100000)}")
        return (paths, response.result, sample_rate)

class SaveAudio():
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        Input Types
        """
        return {
            "required": {
                "tensor": ("AUDIO", ),
                "output_path": ("STRING", {"default": 'ComfyUI/output/audio_samples'}),
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 10000000000, "step": 1}),
                "id_string": ("STRING", {"default": 'ComfyUI'}),
                "tame": (['Enabled', 'Disabled'],)
                },
            "optional": {
                },
            }

    RETURN_TYPES = ("LIST", )
    RETURN_NAMES = ("out_paths", )
    FUNCTION = "save_audio_ui"
    OUTPUT_NODE = True

    CATEGORY = "Audio/SampleDiffusion"

    def save_audio_ui(self, tensor, output_path, sample_rate, id_string, tame):
        return (save_audio(audio_out=(0.5 * tensor).clamp(-1,1) if(tame == 'Enabled') else tensor, output_path=output_path, sample_rate=sample_rate, id_str=id_string), )

class LoadAudio():
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        Input Types
        """
        return {
            "required": {
                "file_path": ("STRING", {"default": ''}),
                },
            "optional": {
                },
            }

    RETURN_TYPES = ("STRING", "INT", "AUDIO")
    RETURN_NAMES = ("path", "sample_rate", "tensor")
    FUNCTION = "LoadAudio"
    OUTPUT_NODE = True

    CATEGORY = "Audio/SampleDiffusion"

    def LoadAudio(self, file_path):
        if file_path != '':
            waveform, samplerate = torchaudio.load(file_path)
        else:
            waveform, samplerate = None, None
        return (file_path, samplerate, waveform)

class PreviewAudioFile():
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        Input Types
        """
        return {
            "required": {
                "paths": ("LIST",),
                },
            "optional": {
                },
            }

    RETURN_TYPES = ()
    FUNCTION = "PreviewAudioFile"
    OUTPUT_NODE = True

    CATEGORY = "Audio/SampleDiffusion"

    def PreviewAudioFile(self, paths):
        # fix slashes
        paths = [path.replace("\\", "/") for path in paths]
        # get filenames with extensions from paths

        filenames = [os.path.basename(path) for path in paths]
        return {"result": (filenames,), "ui": filenames}

class PreviewAudioTensor():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("AUDIO",),
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 10000000000, "step": 1, "forceInput": True}),
                "tame": (['Enabled', 'Disabled'],)
                },
            "optional": {
                },
            }

    RETURN_TYPES = ("LIST", )
    RETURN_NAMES = ("paths", )
    FUNCTION = "PreviewAudioTensor"
    OUTPUT_NODE = True

    CATEGORY = "Audio/SampleDiffusion"

    def PreviewAudioTensor(self, tensor, sample_rate, tame):
        # fix slashes
        paths = save_audio((0.5 * tensor).clamp(-1,1) if(tame == 'Enabled') else tensor, f"{comfy_dir}/temp", sample_rate, f"{random.randint(0, 10000000000)}")
        paths = [path.replace("\\", "/") for path in paths]
        return {"result": (paths,), "ui": paths}

class StringListIndex:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list": ("LIST", ),
                "index": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "doStuff"
    CATEGORY = "Audio/SampleDiffusion/Helpers"

    def doStuff(self, list, index):
        return (list[index],)

    
NODE_CLASS_MAPPINGS = {
    "GenerateAudioSample": AudioInference,
    "SaveAudioTensor": SaveAudio,
    "LoadAudioFile": LoadAudio,
    "PreviewAudioFile": PreviewAudioFile,
    "PreviewAudioTensor": PreviewAudioTensor,
    "GetStringByIndex": StringListIndex,
}
