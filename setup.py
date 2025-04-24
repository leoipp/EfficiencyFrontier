from setuptools import setup, find_packages
from pathlib import Path

# Caminho do README.md
readme_path = Path(__file__).parent / "README.md"

setup(
    name="EfficiencyFrontier",
    version="0.1b",
    author="Leonardo Ippolito Rodrigues",
    author_email="leoippef@gmail.com",
    description=(
        "Efficiency frontier simulation for environmental modelling, "
        "based on the Markowitz portfolio theory and temporal maximum values."
    ),
    long_description=readme_path.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/leoipp/EfficiencyFrontier",
    project_urls={
        "Source": "https://github.com/leoipp/EfficiencyFrontier",
        "Bug Tracker": "https://github.com/leoipp/EfficiencyFrontier/issues",
    },
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=[
        "numpy",
        "rasterio",
        "matplotlib",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    keywords="efficiency frontier markowitz raster simulation gis environmental-modelling",
    python_requires=">=3.7",
)
