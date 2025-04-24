from setuptools import setup, find_packages

setup(
    name="EfficiencyFrontier",
    version="0.1b",
    author="Leonardo Ippolito Rodrigues",
    author_email="leoippef@gmail.com",
    description="Efficiency frontier simulation for environmental modelling based on markowitz and temporal maximum "
                "values.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/leoipp/EfficiencyFrontier.git",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "rasterio",
        "matplotlib"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
