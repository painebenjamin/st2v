from __future__ import annotations

import os
import gc
import re
import json
import torch

from typing import Optional, Union
from contextlib import ExitStack

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, KarrasDiffusionSchedulers
from diffusers.utils import is_accelerate_available, is_xformers_available

from accelerate import init_empty_weights, set_module_tensor_to_device
from huggingface_hub import hf_hub_download

from st2v.pipelines.text_to_video import TextToVideoSDPipeline
from st2v.models import (
    UNet3DConditionModel,
    ControlNetModel,
    FrozenOpenCLIPImageEmbedder,
    ImageEmbeddingContextResampler,
)
from st2v.utils import logger, iterate_state_dict

class StreamingTextToVideoPipeline(TextToVideoSDPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        controlnet: ControlNetModel,
        scheduler: KarrasDiffusionSchedulers,
        image_encoder: Optional[FrozenOpenCLIPImageEmbedder]=None,
        resampler: Optional[ImageEmbeddingContextResampler]=None,
    ):
        super(StreamingTextToVideoPipeline, self).__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
        )
        self.register_modules(
            image_encoder=image_encoder,
            resampler=resampler,
        )

    @classmethod
    def from_single_file(
        file_path_or_repository: str,
        filename: str="s2tv.safetensors",
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
            st2v_config = json.load(f)

        # Create the scheduler
        scheduler = DDIMScheduler(**st2v_config["scheduler"])

        # Create tokenizer (downloaded)
        tokenizer = CLIPTokenizer.from_pretrained(
            st2v_config["tokenizer"]["model"],
            subfolder=st2v_config["tokenizer"].get("subfolder", None),
            cache_dir=cache_dir,
        )

        # Create the base models
        context = ExitStack()
        if is_accelerate_available():
            context.enter_context(no_init_weights())
            context.enter_context(init_empty_weights())

        with context:
            # UNet
            unet = UNet3DConditionModel.from_config(st2v_config["unet"])

            # VAE
            vae = AutoencoderKL.from_config(st2v_config["vae"])

            # Text encoder
            text_encoder = CLIPTextModel.from_config(st2v_config["text_encoder"])

            # Image encoder
            image_encoder = FrozenOpenCLIPImageEmbedder.from_config(st2v_config["image_encoder"])

            # Resampler
            resampler = ImageEmbeddingContextResampler.from_config(st2v_config["resampler"])

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
                unet.load_state_dict(state_dicts["unet"])
                vae.load_state_dict(state_dicts["vae"])
                image_encoder.load_state_dict(state_dicts["image_encoder"], strict=False)
                text_encoder.load_state_dict(state_dicts["text_encoder"])
                resampler.load_state_dict(state_dicts["resampler"])
                del state_dicts
                gc.collect()
            except KeyError as ex:
                raise RuntimeError(f"File did not provide a state dict for {ex}")

        # Create controlnet
        controlnet = ControlNetModel.from_unet(
            unet,
            **st2v_config["controlnet"]
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
        )

        if torch_dtype is not None:
            pipeline.to(torch_dtype)

        pipeline.to(device)

        if is_xformers_available():
            from st2v.models.processor import set_use_memory_efficient_attention_xformers
            set_use_memory_efficient_attention_xformers(
                unet,
                num_frames_conditioning=unet.config.num_frames_conditioning,
                num_frames=unet.config.num_frames,
            )
            set_use_memory_efficient_attention_xformers(
                controlnet,
                num_frames_conditioning=unet.config.num_frames_conditioning,
                num_frames=unet.config.num_frames,
            )
        else:
            logger.warning("XFormers is not available, falling back to PyTorch attention")

        return pipeline
