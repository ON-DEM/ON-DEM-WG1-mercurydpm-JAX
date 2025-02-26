# ⚽ MercuryDPM2 ⚽ JAX ⚽

This an example repo/proof of concept to interface MercuryDPM with JAX.

# Motivation
The repo interfaces [MercuryDPM](https://bitbucket.org/mercurydpm/mercurydpm/src/master/), a discrete element method code, with the [JAX foreign function interface](https://docs.jax.dev/en/latest/ffi.html). A motivation for this project is to enable faster training of machine learning algorithms by eliminating the need to load data. Nanobind allows for the compilation of MercuryDPM into a shared library, which can then be imported and used in Python.


# Requirements

The following requirements must be met for this example project:
- MercuryDPM (see [requirements](https://www.mercurydpm.org/documentation))
- [uv](https://github.com/astral-sh/uv) package manager

Note this module uses a branch of Mercury

## Installation
- `git clone --recurse-submodules -j8 git@github.com:Retiefasaurus/MercuryDPM2JAX.git`
- `uv build` (compile uv into site-packages)
- `uv sync`
- `uv run python test/test_ffi.py`

Interface is found in `test/test_ffi.py` and `mercurydpm/Drivers/2JAX/mdpm_jax.cpp`

# What is so cool about it?
- Packaging MercuryDPM as a shared library: Using scikit-build, we can package MercuryDPM as a shared library within Python.
- Interface with XLA: MercuryDPM is interfaced with XLA, allowing for efficient computation.
- Stateful foreign function interface: We can call stateful foreign functions with MercuryDPM using the JAX FFI.

# Technologies
The following technologies are used in MercuryDPM2JAX:
- nanobind: Used for binding MercuryDPM to Python.
- uv package manager: Used for managing dependencies.
- scikit-build: Used for packaging MercuryDPM as a shared library.
- XLA with JAX: ML etc.


# How we modify Mercury DPM 
To interface MercuryDPM with JAX, we make the following modifications:
- Interface: The interface is found in the mercurydpm/Drivers/2JAX/mdpm_jax.cpp file.
- Driver CMakefile: The mercurydpm/Drivers/2JAX/CMakeLists.txt file specifies nanobind and installs the shared library.
- Modified CMakeLists.txt: The mercurydpm/CMakeLists.txt file is modified to include XLA and nanobind.
- Modified Kernel CMakeLists.txt: The mercurydpm/Kernel/CMakeLists.txt file is modified to compile with -fPIC.
- Commented out unused driver scripts: Unused driver scripts are commented out to avoid conflicts.

