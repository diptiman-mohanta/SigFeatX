from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="SigFeatX",
    version="0.1.0",
    author="Diptiman Mohanta",
    author_email="diptimanmohanta7@gmail.com",
    description="A comprehensive signal feature extraction library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/diptiman-mohanta/SigFeatX",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Signal Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "PyWavelets>=1.1.1",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
        "examples": [
            "pandas>=1.3.0",
            "matplotlib>=3.3.0",
        ],
    },
)


