# NeuralAmpModelerCoreParametric

[![Build](https://github.com/phillipmself/NeuralAmpModelerCoreParametric/actions/workflows/build.yml/badge.svg)](https://github.com/phillipmself/NeuralAmpModelerCoreParametric/actions/workflows/build.yml)

Core C++ DSP library for NAM plugins.

This is a fork of [NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore) that adds
runtime support for **parametric** models (`HyperWaveNet` / `ConcatWaveNet`) — WaveNet architectures
whose weights are conditioned at runtime by a hypernetwork or concatenated control inputs, enabling
continuous, knob-controllable parameters on a loaded `.nam` model.

For an example of how to use, see [NamParametricPlugin](https://github.com/phillipmself/NamParametricPlugin),
since the upstream [NeuralAmpModelerPlugin](https://github.com/sdatkinson/NeuralAmpModelerPlugin) cannot load
this fork's parametric models.

## Included Tools

This repo includes a few helpful tools.
For guidance on building them, have a look at the workflow provided in `.github/workflows/build.yml`.

* [`run_tests`](https://github.com/sdatkinson/NeuralAmpModelerCore/blob/761fa968766bcf67d3035320c195969d9ba41fa1/tools/CMakeLists.txt#L15), which runs a suite of unit tests.
* [`loadmodel`](https://github.com/sdatkinson/NeuralAmpModelerCore/blob/761fa968766bcf67d3035320c195969d9ba41fa1/tools/CMakeLists.txt#L13), which allows you to test loading a `.nam` file.
* [`benchmodel`](https://github.com/sdatkinson/NeuralAmpModelerCore/blob/761fa968766bcf67d3035320c195969d9ba41fa1/tools/CMakeLists.txt#L14), which allows you to test how quickly a model runs in real time. _Note: For more granular profiling tools, check out the [`main-profiling`](https://github.com/sdatkinson/NeuralAmpModelerCore/tree/main-profiling) branch.

## Sharp edges
This library uses [Eigen](http://eigen.tuxfamily.org) to do the linear algebra routines that its neural networks require. Since these models hold their parameters as eigen object members, there is a risk with certain compilers and compiler optimizations that their memory is not aligned properly. This can be worked around by providing two preprocessor macros: `EIGEN_MAX_ALIGN_BYTES 0` and `EIGEN_DONT_VECTORIZE`, though this will probably harm performance. See [Structs Having Eigen Members](http://eigen.tuxfamily.org/dox-3.2/group__TopicStructHavingEigenMembers.html) for more information. This is being tracked as [Issue 67](https://github.com/sdatkinson/NeuralAmpModelerCore/issues/67).

## Sponsors

<div align="center">
  <img src="media/tone3000-logo.svg" alt="Tone3000 logo">
</div>

Development of version 0.4.0 of the upstream [NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore)
library (which this fork is based on) was generously supported by [TONE3000](https://tone3000.com). Note that
TONE3000 has no involvement in this fork or its parametric-model features — please direct any questions about
that work here, not to them.
**Thank you!**
