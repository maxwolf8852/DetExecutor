import os
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = ['pyyaml', 'opencv-python', 'torch', 'torchvision',
                'requests', 'wget', 'tqdm', 'ultralytics', 'black', 'colorama', 'pthflops']

setuptools.setup(
    name="det_executor",
    version="0.0.5",
    author="Maxim Volkovskiy",
    author_email="maxwolf8852@gmail.com",
    description="Python package with latest versions of YOLO architecture for training and inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maxwolf8852/DetExecutor.git",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
