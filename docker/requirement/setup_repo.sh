#!/bin/sh

SUDO="sudo -H"

# install most dependencies via apt-get
${SUDO} apt-get -y update
${SUDO} apt-get -y upgrade
# We explicitly set the C++ compiler to g++, the default GNU g++ compiler. This is
# needed because we depend on system-installed libraries built with g++ and linked
# against libstdc++. In case `c++` corresponds to `clang++`, code will not build, even
# if we would pass the flag `-stdlib=libstdc++` to `clang++`.
${SUDO} apt-get -y install g++ cmake pkg-config libboost-serialization-dev libboost-filesystem-dev libboost-system-dev libboost-program-options-dev libboost-test-dev libeigen3-dev libode-dev wget libyaml-cpp-dev
export CXX=g++
export MAKEFLAGS="-j `nproc`"

${SUDO} apt-get -y install python3-dev python3-pip
# install additional python dependencies via pip
${SUDO} pip3 install -vU https://github.com/CastXML/pygccxml/archive/develop.zip pyplusplus
# install castxml
${SUDO} apt-get -y install castxml
${SUDO} apt-get -y install libboost-python-dev
${SUDO} apt-get -y install libboost-numpy-dev python${PYTHONV}-numpy
${SUDO} apt-get -y install pypy3

git clone https://github.com/ompl/ompl.git
cd ompl
mkdir -p build/Release
cd build/Release
cmake ../..
make -j 4 update_bindings
make -j 4
sudo make install
cp -r /usr/lib/python3.8/site-packages/* /usr/lib/python3/dist-packages/
