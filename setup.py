import os
import re
import sys

from setuptools import find_packages, setup

deps = [
    "click",
	"diffusers>=0.27.2",
	"einops>=0.4.1",
	"imageio>=2.33.0",
	"imageio[ffmpeg]",
	"numpy>=1.23.5",
	"omegaconf>=2.2.3",
	"Pillow>=9.5.0",
	"safetensors>=0.4.2",
	"torch>=2.0.1",
	"torchvision>=0.15.2",
	"tqdm>=4.66.1",
	"transformers>=4.39.3",
	"xformers>=0.0.22",
    "scikit-image>=0.21.0",
    "scikit-learn>=1.3.2",
    "scipy>=1.11.4",
    "moviepy>=1.0.3"
]

setup(
    name="vidxtend",
    version="0.1.0",  # expected format is one of x.y.z.dev0, or x.y.z.rc1 or x.y.z (no to dashes, yes to dots)
    description="Stage 2 of StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="BigScience Open RAIL-M License",
    author="Roberto Henschel, Levon Khachatryan, Daniil Hayrapetyan, Hayk Poghosyan, Vahram Tadevosyan, Zhangyang Wang, Shant Navasardyan, Humphrey Shi, Benjamin Paine",
    author_email="painebenjamin@gmail.com",
    url="https://streamingt2v.github.io/",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"vidxtend": ["py.typed", "data/*"]},
    include_package_data=True,
    python_requires=">=3.8.0",
    install_requires=deps,
    entry_points={
        "console_scripts": [
            "vidxtend = vidxtend.__main__:main"
        ]
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
    ]
    + [f"Programming Language :: Python :: 3.{i}" for i in range(8, 13)],
)
