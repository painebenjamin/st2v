from __future__ import annotations

import os
import sys
import click
import logging
import traceback

from PIL import Image

from typing import Optional

@click.command()
@click.argument("video", type=click.Path(exists=True, dir_okay=False))
@click.argument("prompt", type=str)
@click.option("--frame-rate", "-fps", type=int, default=None, help="Video FPS. Also controls the sampling rate of the audio. Will default to the video FPS if a video file is provided, or 30 if not.", show_default=True)
@click.option("--guidance-scale", "-cfg", type=float, default=7.5, help="Guidance scale for the diffusion process.", show_default=True)
@click.option("--num-inference-steps", "-ns", type=int, default=50, help="Number of diffusion steps.", show_default=True)
@click.option("--num-frames", "-nf", type=int, default=24, help="The total number of frames for the output video.", show_default=True)
@click.option("--seed", "-s", type=int, default=None, help="Random seed.")
@click.option("--model", "-m", type=str, default="benjamin-paine/vidxtend", help="HuggingFace model name.")
@click.option("--no-half", "-nh", is_flag=True, default=False, help="Do not use half precision.", show_default=True)
@click.option("--no-offload", "-no", is_flag=True, default=False, help="Do not offload to the CPU to preserve GPU memory.", show_default=True)
@click.option("--gpu-id", "-g", type=int, default=0, help="GPU ID to use.")
@click.option("--model-single-file", "-sf", is_flag=True, default=False, help="Download and use a single file instead of a directory.")
@click.option("--config-file", "-cf", type=str, default="config.json", help="Config file to use when using the model-single-file option. Accepts a path or a filename in the same directory as the single file. Will download from the repository passed in the model option if not provided.", show_default=True)
@click.option("--model-filename", "-mf", type=str, default="vidxtend.safetensors", help="The model file to download when using the model-single-file option.", show_default=True)
@click.option("--remote-subfolder", "-rs", type=str, default=None, help="Remote subfolder to download from when using the model-single-file option.")
@click.option("--cache-dir", "-cd", type=click.Path(exists=True, file_okay=False), help="Cache directory to download to. Default uses the huggingface cache.", default=None)
@click.option("--output", "-o", type=click.Path(exists=False, dir_okay=False), help="Output file.", default="output.mp4", show_default=True)
def main(
    video: str,
    prompt: str,
    frame_rate: Optional[int]=None,
    guidance_scale: float=7.5,
    num_inference_steps: int=50,
    num_frames: int=24,
    seed: Optional[int]=None,
    model: str="benjamin-paine/vidxtend",
    no_half: bool=False,
    no_offload: bool=False,
    gpu_id: int=0,
    model_single_file: bool=False,
    config_file: str="config.json",
    model_filename: str="vidxtend.safetensors",
    remote_subfolder: Optional[str]=None,
    cache_dir: Optional[str]=None,
    output: str="output.mp4",
) -> None:
    """
    Run VidXtend on a video file.
    """
    if os.path.exists(output):
        base, ext = os.path.splitext(os.path.basename(output))
        dirname = os.path.dirname(output)
        suffix = 1
        while os.path.exists(os.path.join(dirname, f"{base}-{suffix}{ext}")):
            suffix += 1
        new_output_filename = f"{base}-{suffix}{ext}"
        click.echo(f"Output file {output} already exists. Writing to {new_output_filename} instead.")
        output = os.path.join(dirname, new_output_filename)

    import torch
    from vidxtend.utils import Video
    from vidxtend.pipelines import VideoExtendPipeline

    device = (
        torch.device("cuda", index=gpu_id)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if no_half:
        variant = None
        torch_dtype = None
    else:
        variant = "fp16"
        torch_dtype = torch.float16

    if model_single_file:
        pipeline = VideoExtendPipeline.from_single_file(
            model,
            filename=model_filename,
            config_filename=config_file,
            variant=variant,
            subfolder=remote_subfolder,
            cache_dir=cache_dir,
            device=device,
            torch_dtype=torch_dtype,
        )
    else:
        pipeline = VideoExtendPipeline.from_pretrained(
            model,
            variant=variant,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
        )

    if torch_dtype is not None:
        pipeline.to(torch_dtype)
        pipeline.controlnet.to(torch_dtype)
        if pipeline.resampler is not None:
            pipeline.resampler.to(torch_dtype)
        if pipeline.image_encoder is not None:
            pipeline.image_encoder.to(torch_dtype)
    
    if no_offload:
        pipeline.to(device)
    else:
        pipeline.enable_model_cpu_offload(gpu_id=gpu_id)

    from vidxtend.utils import logger
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    pipeline.image_encoder.to(device)
    pipeline.resampler.to(device)

    images = Video.from_file(video).frames_as_list
    result = pipeline(
        images=images,
        prompt=prompt,
        negative_prompt="Bad quality",
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
    )

    Video(result).save("./result.mp4", rate=8, overwrite=True)

    """
    if frame_rate is None and video is not None:
        frame_rate = get_frame_rate(video)
    else:
        frame_rate = 8
    bytes_written = video_container.save(output, rate=frame_rate)
    click.echo(f"Wrote {len(result.videos)} frames to {output} ({human_size(bytes_written)})")
    """

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as ex:
        sys.stderr.write(f"{ex}\r\n")
        sys.stderr.write(traceback.format_exc())
        sys.stderr.flush()
        sys.exit(5)
