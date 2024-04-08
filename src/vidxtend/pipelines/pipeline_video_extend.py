from __future__ import annotations

import os
import gc
import re
import json
import torch

from typing import Optional, Union
from PIL import Image
from contextlib import ExitStack

from transformers import (
    CLIPTextModel,
    CLIPTextConfig,
    CLIPTokenizer
)
from transformers.modeling_utils import no_init_weights

from diffusers import AutoencoderKL
from diffusers.schedulers import KarrasDiffusionSchedulers, DDIMScheduler
from diffusers.utils import is_accelerate_available, is_xformers_available

from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

from huggingface_hub import hf_hub_download

from vidxtend.models import (
    UNet3DConditionModel,
    ControlNetModel,
    FrozenOpenCLIPImageEmbedder,
    ImageEmbeddingContextResampler,
    NoiseGenerator,
    MaskGenerator,
)
from vidxtend.utils import (
    logger,
    fit_image,
    iterate_state_dict,
    IMAGE_ANCHOR_LITERAL,
    IMAGE_FIT_LITERAL
)
from vidxtend.pipelines.pipeline_text_to_video import TextToVideoSDPipeline

def is_accelerate_available():
    return False

class VideoExtendPipeline(TextToVideoSDPipeline):
    _exclude_from_cpu_offload = ["controlnet"]
    model_cpu_offload_seq = "image_encoder->resampler->text_encoder->unet->vae"

    @classmethod
    def from_single_file(
        cls,
        file_path_or_repository: str,
        filename: str="vidxtend.safetensors",
        config_filename: str="config.json",
        variant: Optional[str]=None,
        subfolder: Optional[str]=None,
        device: Optional[Union[str, torch.device]]=None,
        torch_dtype: Optional[torch.dtype]=None,
        cache_dir: Optional[str]=None,
    ) -> StreamingTextToVideoPipeline:
        """
        Load a streaming text-to-video pipeline from a file or repository.
        """
        if variant is not None:
            filename, ext = os.path.splitext(filename)
            filename = f"{filename}.{variant}{ext}"

        if device is None:
            device = "cpu"
        else:
            device = str(device)

        if os.path.isdir(file_path_or_repository):
            model_dir = file_path_or_repository
            if subfolder:
                model_dir = os.path.join(model_dir, subfolder)
            file_path = os.path.join(model_dir, filename)
            config_path = os.path.join(model_dir, config_filename)
        elif os.path.isfile(file_path_or_repository):
            file_path = file_path_or_repository
            if os.path.isfile(config_filename):
                config_path = config_filename
            else:
                config_path = os.path.join(os.path.dirname(file_path), config_filename)
                if not os.path.exists(config_path) and subfolder:
                    config_path = os.path.join(os.path.dirname(file_path), subfolder, config_filename)
        elif re.search(r"^[a-zA-Z0-9_-]+\/[a-zA-Z0-9_-]+$", file_path_or_repository):
            file_path = hf_hub_download(
                file_path_or_repository,
                filename,
                subfolder=subfolder,
                cache_dir=cache_dir,
            )
            try:
                config_path = hf_hub_download(
                    file_path_or_repository,
                    config_filename,
                    subfolder=subfolder,
                    cache_dir=cache_dir,
                )
            except:
                config_path = hf_hub_download(
                    file_path_or_repository,
                    config_filename,
                    cache_dir=cache_dir,
                )

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"File {config_path} not found.")

        with open(config_path, "r") as f:
            vidxtend_config = json.load(f)

        repo_id = "damo-vilab/text-to-video-ms-1.7b"
        # Create the scheduler
        scheduler = DDIMScheduler.from_pretrained(repo_id, subfolder="scheduler")

        # Create the noise generator
        noise_generator = None

        # Create the mask generator
        mask_generator = MaskGenerator.from_config(vidxtend_config["mask_generator"])

        # Create tokenizer (downloaded)
        tokenizer = CLIPTokenizer.from_pretrained(
            vidxtend_config["tokenizer"]["model"],
            subfolder=vidxtend_config["tokenizer"].get("subfolder", None),
            cache_dir=cache_dir,
        )

        # Create the base models
        context = ExitStack()
        if is_accelerate_available():
            context.enter_context(no_init_weights())
            context.enter_context(init_empty_weights())

        # UNet
        unet = UNet3DConditionModel.from_pretrained(repo_id, subfolder="unet")

        # VAE
        vae = AutoencoderKL.from_pretrained(repo_id, subfolder="vae")

        # Text encoder
        text_encoder = CLIPTextModel.from_pretrained(repo_id, subfolder="text_encoder")

        # Resampler
        resampler = ImageEmbeddingContextResampler(**vidxtend_config["resampler"])

        # Image encoder
        image_encoder = FrozenOpenCLIPImageEmbedder(**vidxtend_config["image_encoder"])

        # Load the weights
        logger.debug("Models created, loading weights...")
        state_dicts = {}
        for key, value in iterate_state_dict(file_path):
            try:
                module, _, key = key.partition(".")
                if is_accelerate_available():
                    if module == "unet":
                        set_module_tensor_to_device(unet, key, device=device, value=value)
                    elif module == "vae":
                        set_module_tensor_to_device(vae, key, device=device, value=value)
                    elif module == "image_encoder":
                        set_module_tensor_to_device(image_encoder, key, device=device, value=value)
                    elif module == "text_encoder":
                        set_module_tensor_to_device(text_encoder, key, device=device, value=value)
                    elif module == "resampler":
                        set_module_tensor_to_device(resampler, key, device=device, value=value)
                    elif module == "controlnet":
                        if "controlnet" not in state_dicts:
                            state_dicts["controlnet"] = {}
                        state_dicts["controlnet"][key] = value
                    else:
                        raise ValueError(f"Unknown module: {module}")
                else:
                    if module not in state_dicts:
                        state_dicts[module] = {}
                    state_dicts[module][key] = value
            except (AttributeError, KeyError, ValueError) as ex:
                logger.warning(f"Skipping module {module} key {key} due to {type(ex)}: {ex}")

        if not is_accelerate_available():
            try:
                unet.load_state_dict(state_dicts["unet"], strict=False)
                vae.load_state_dict(state_dicts["vae"], strict=False)
                image_encoder.load_state_dict(state_dicts["image_encoder"], strict=False)
                text_encoder.load_state_dict(state_dicts["text_encoder"], strict=False)
                resampler.load_state_dict(state_dicts["resampler"], strict=False)
            except KeyError as ex:
                raise RuntimeError(f"File did not provide a state dict for {ex}")

        # Create controlnet
        controlnet = ControlNetModel.from_unet(
            unet,
            **vidxtend_config["controlnet"]
        )

        # Load controlnet state dict
        controlnet.load_state_dict(state_dicts["controlnet"], strict=False)

        # Cleanup
        del state_dicts

        # Create the pipeline
        pipeline = cls(
            unet=unet,
            vae=vae,
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            scheduler=scheduler,
            controlnet=controlnet,
            resampler=resampler,
            tokenizer=tokenizer,
            noise_generator=noise_generator,
            mask_generator=mask_generator,
        )

        if torch_dtype is not None:
            pipeline.to(torch_dtype)

        if is_xformers_available():
            from vidxtend.models.processor import set_use_memory_efficient_attention_xformers
            set_use_memory_efficient_attention_xformers(
                unet,
                num_frames_conditioning=controlnet.config.num_frames_conditioning,
                num_frames=controlnet.config.num_frames,
            )
            set_use_memory_efficient_attention_xformers(
                controlnet,
                num_frames_conditioning=controlnet.config.num_frames_conditioning,
                num_frames=controlnet.config.num_frames,
            )
        else:
            logger.warning("XFormers is not available, falling back to PyTorch attention")

        return pipeline

    def __call__(
        self,
        images: List[Image.Image],
        prompt: str,
        num_frames: int=24,
        negative_prompt: Optional[str]=None,
        guidance_scale: float=7.5,
        num_inference_steps: int=25,
        anchor: IMAGE_ANCHOR_LITERAL="top-left",
        fit: IMAGE_FIT_LITERAL="cover",
    ) -> List[Image.Image]:
        """
        Extends a video.
        """
        image_size = self.unet.config.sample_size * self.vae_scale_factor
        images = fit_image(
            images,
            width=image_size,
            height=image_size,
            fit=fit,
            anchor=anchor
        )
        while len(images) < num_frames:
            result = super().__call__(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                image=images[-8:],
                input_frames_conditioning=images[:1],
                eta=1.0,
                output_type="pil"
            )
            images.extend(result.frames[8:])
            self.reset_noise_generator_state()
            torch.cuda.empty_cache()
            gc.collect()
        return images[:num_frames]
