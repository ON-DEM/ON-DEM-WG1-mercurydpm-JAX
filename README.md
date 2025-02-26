# MercuryDPM to JAX example

An example interface between MercuryDPM (Discrete Element Method engine) and JAX's XLA for machine learning applications

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

## 🌟 Motivation
Accelerate ML training pipelines by:
- Eliminating data I/O bottlenecks through direct in-process integration
- Enabling gradient-based optimization of particle dynamics parameters
- Providing a stateful foreign function interface (FFI) for [JAX/XLA](https://docs.jax.dev/en/latest/ffi.html)
- Packaging [MercuryDPM](https://bitbucket.org/mercurydpm/mercurydpm/src/master/) as a native Python extension via [nanobind](https://github.com/wjakob/nanobind)

**Key contribution** How to compile MercuryDPM into a shared library accessible in Python/JAX ecosystems while maintaining stateful DEM simulations. 

## 📋 Requirements
- [uv](https://github.com/astral-sh/uv) (Blazing-fast Python package manager)
- CMake 3.10+

### Core Dependencies
- [MercuryDPM](https://www.mercurydpm.org) (custom branch, feature/2JAX)
- Python 3.9+
- [JAX](https://github.com/google/jax) (with GPU/TPU support recommended)
- [Nanobind](https://github.com/wjakob/nanobind) (C++/Python binding)
- [scikit-build](https://scikit-build.readthedocs.io/)

## 🛠 Installation

```bash
# Clone with submodules
git clone --recurse-submodules -j8 https://github.com/Retiefasaurus/MercuryDPM2JAX.git
cd MercuryDPM2JAX

# Build & install
uv build
uv sync


uv run test/test_ffi.py

```
Interface is found in `test/test_ffi.py` and `mercurydpm/Drivers/2JAX/mdpm_jax.cpp`

## How we modify Mercury DPM ?
To interface MercuryDPM with JAX, we make the following modifications:
- Interface: The interface is found in the mercurydpm/Drivers/2JAX/mdpm_jax.cpp file.
- Driver CMakefile: The mercurydpm/Drivers/2JAX/CMakeLists.txt file specifies nanobind and installs the shared library.
- Modified CMakeLists.txt: The mercurydpm/CMakeLists.txt file is modified to include XLA and nanobind.
- Modified Kernel CMakeLists.txt: The mercurydpm/Kernel/CMakeLists.txt file is modified to compile with -fPIC.
- Commented out unused driver scripts: Unused driver scripts are commented out to avoid conflicts.


## Repo

This repo is maintained by Retief Lubbe.
