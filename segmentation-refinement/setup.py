import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="segmentation-refinement",
    version="0.6",
    author="Ho Kei Cheng, Jihoon Chung",
    author_email="hkchengrex@gmail.com",
    description="Deep learning based segmentation refinement system.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hkchengrex/CascadePSP",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['torch', 'torchvision', 'requests'],
)
