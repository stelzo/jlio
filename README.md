# jlio - Jetson LIO

A parallelized version of [FAST-LIO2](https://github.com/hku-mars/FAST_LIO) primarily for Jetson devices using CUDA with Unified Memory. If no CUDA device is available, it falls back to CPU threadpooling.

** Work in progress **

## Setup
There are frontends and backends. Backends are organized in libraries marking the actual processing, while frontends provide the data sources, like a ROS node or standalone executables for testing purposes.

### Linux

Tested on
- Ubuntu 18.04
- Ubuntu 20.04
- Ubuntu 22.04

- CUDA 10.2

```bash
# if gcc-11 is not available in your apt, add it via
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt-get install cmake libeigen3-dev libpcl-dev libomp-dev libtbb-dev gcc-11 g++-11
```

### Mac OS

Tested on
- Ventura 13.4
- Monterey 12.0

```bash
brew install cmake eigen pcl qt5 gcc@11 tbb
```

The cmake script depends on brew to install qt5. If you installed qt5 in some other way, head to the root `CMakeLists.txt` and remove the part where `Qt5_DIR` is set.

## Build
```bash
git clone git@github.com:stelzo/jlio.git
mkdir -p jlio/build && cd jlio/build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc-11 -DCMAKE_CXX_COMPILER=g++-11 ..
cmake --build .
```

## Acknowledgements
Thanks to the [Mechatronics and Robotic Systems (MaRS) Laboratory](https://mars.hku.hk) from the University of Hong Kong for FAST-LIO2.

Thanks to the Institute of Computer Science of the [Osnabrück University](https://www.uni-osnabrueck.de/en/home/) for providing the hardware.
