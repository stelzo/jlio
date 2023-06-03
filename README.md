# jlio - Jetson LIO

A parallelized version of [FAST-LIO2](https://github.com/hku-mars/FAST_LIO) primarily for Jetson devices using CUDA with Unified Memory. If no CUDA device is available, it falls back to CPU threadpooling.

** Work in progress **

## Setup
There are frontends and backends. Backends are organized in libraries marking the actual processing, while frontends provide the data sources, like a ROS node or standalone executables for testing purposes or use in HPC clusters for 3D reconstruction.

### Backends

### Ubuntu (L4T)
- 18.04
- 20.04
- 22.04

We require a newer cmake version than normally distributed on Ubuntu systems for proper CUDA detection.
```bash
# add the cmake apt repo
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"

# optional begin
sudo apt install kitware-archive-keyring # keeps the keys updated
sudo rm /etc/apt/trusted.gpg.d/kitware.gpg # delete the old one, will be auto replaced at next update
# optional end

sudo apt update # if this throws an error, "NO_PUBKEY", copy the key after and do sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key <KEY>

sudo apt-get install cmake libeigen3-dev libpcl-dev libomp-dev libtbb-dev libboost-dev clang clang++
```

- CUDA 10
- CUDA 11
- CUDA 12

If you have multiple CUDA versions installed, link `nvcc` to the version you want to use.

- CPU single/multi core

### Mac OS
- 12 Monterey
- 13 Ventura

```bash
brew install cmake eigen pcl qt5 gcc@11 tbb boost
```

- CPU single/multi core

The cmake script depends on brew to install qt5. If you installed qt5 in some other way, head to the root `CMakeLists.txt` and remove the part where `Qt5_DIR` is set.

### Frontends
- ROS1 melodic
- ROS1 noetic


## Build
The script will auto-detect CUDA and fall back to CPU threadpools.
```bash
git clone git@github.com:stelzo/jlio.git
cd jlio
make
cd build && ctest # check if everything works
```
If you wish to enforce CPU builds on CUDA enabled devices, use `make all-cpu`.

Alternatively you can choose the normal `ccmake` or `cmake` pipeline with flags or take a look at the targets in the `Makefile`.

## Acknowledgements
Thanks to the [Mechatronics and Robotic Systems (MaRS) Laboratory](https://mars.hku.hk) from the University of Hong Kong for FAST-LIO2.

Thanks to the Institute of Computer Science of the [Osnabrück University](https://www.uni-osnabrueck.de/en/home/) for providing the hardware.
