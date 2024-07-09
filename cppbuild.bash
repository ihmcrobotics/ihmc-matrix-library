#!/bin/bash
# This build script is designed to work on Linux and Windows. For Windows, run from a bash shell launched with launchBashWindows.bat

REPO_ROOT=$(pwd)

rm -rf cppbuild # Optional clean
mkdir cppbuild

#### Installing Eigen headers ####
cd cppbuild
EIGEN_VERSION=3.4.0
curl https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-$EIGEN_VERSION.zip -O
unzip -n eigen-$EIGEN_VERSION.zip
cd eigen-$EIGEN_VERSION
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=. ..
cmake --install .
cd $REPO_ROOT

#### Building NativeCommonOps ####
cd cppbuild
cp -r ../NativeCommonOps ./NativeCommonOps
cd NativeCommonOps
mkdir build
cd build
if [ "$MAC_CROSS_COMPILE_ARM" == "1" ]; then
  cmake -DEigen3_DIR=$(pwd)/../../eigen-$EIGEN_VERSION/build \
        -DCMAKE_OSX_ARCHITECTURES="arm64" \
        ..
elif [ "$LINUX_CROSS_COMPILE_ARM" == "1" ]; then
  cmake -DEigen3_DIR=$(pwd)/../../eigen-$EIGEN_VERSION/build \
        -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
        -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
        -DCMAKE_FIND_ROOT_PATH=/usr/aarch64-linux-gnu \
        -DCMAKE_PROGRAM_PATH=/usr/aarch64-linux-gnu/bin \
        ..
else
  cmake -DEigen3_DIR=$(pwd)/../../eigen-$EIGEN_VERSION/build \
        ..
fi
cmake --build . --config Release -j $(nproc)
ls -al # TODO DEBUG
ls -al Release # TODO DEBUG
cd $REPO_ROOT

#### Copy shared libs to resources ####
cd cppbuild
# Linux
mkdir -p ../src/main/resources/ihmc-matrix-library/native/linux-arm64
mkdir -p ../src/main/resources/ihmc-matrix-library/native/linux-x86_64
if [ -f "NativeCommonOps/build/libNativeCommonOps.so" ]; then
  if [ "$LINUX_CROSS_COMPILE_ARM" == "1" ]; then
    cp NativeCommonOps/build/libNativeCommonOps.so ../src/main/resources/ihmc-matrix-library/native/linux-arm64
  else
    cp NativeCommonOps/build/libNativeCommonOps.so ../src/main/resources/ihmc-matrix-library/native/linux-x86_64
  fi
fi
# Windows
mkdir -p ../src/main/resources/ihmc-matrix-library/native/windows-x86_64
if [ -f "NativeCommonOps/build/Release/NativeCommonOps.dll" ]; then
  cp NativeCommonOps/build/Release/NativeCommonOps.dll ../src/main/resources/ihmc-matrix-library/native/windows-x86_64
fi
# macOS
mkdir -p ../src/main/resources/ihmc-matrix-library/native/macos-arm64
mkdir -p ../src/main/resources/ihmc-matrix-library/native/macos-x86_64
if [ -f "NativeCommonOps/build/libNativeCommonOps.jnilib" ]; then
  if [ "$MAC_CROSS_COMPILE_ARM" == "1" ]; then
    cp NativeCommonOps/build/libNativeCommonOps.jnilib ../src/main/resources/ihmc-matrix-library/native/macos-arm64
  else
    cp NativeCommonOps/build/libNativeCommonOps.jnilib ../src/main/resources/ihmc-matrix-library/native/macos-x86_64
  fi
fi
cd $REPO_ROOT