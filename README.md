[ ![ihmc-matrix-library](https://api.bintray.com/packages/ihmcrobotics/maven-release/ihmc-matrix-library/images/download.svg) ](https://bintray.com/ihmcrobotics/maven-release/ihmc-matrix-library/_latestVersion)
[ ![buildstatus](https://bamboo.ihmc.us/plugins/servlet/wittified/build-status/LIBS-IHMCMATRIXLIBRARY)](https://bamboo.ihmc.us/plugins/servlet/wittified/build-status/LIBS-IHMCMATRIXLIBRARY)

IHMC Matrix Library gathers utilities to improve experience the matrix library EJML.

# Usage

## Windows

Requires the installation of Visual C++ 2019 Redistributable (https://aka.ms/vs/16/release/VC_redist.x64.exe).


# Development

To avoid conflicts, we rename the Eigen namespace to us_ihmc_matrix_library_vendor_matrix in "NativeMatrix.h". We then alias Eigen to us_ihmc_matrix_library_vendor_matrix (!). Do not include <Eigen/Dense> or any other Eigen libraries in any files other than the #define Eigen us_ihmc_matrix_library_vendor_matrix block in NativeMatrix.h.


# Compilation

## Linux

### Eigen 3 (Ubuntu 16.04)
Download the latest Eigen 3.3 release from http://eigen.tuxfamily.org/index.php?title=Main_Page

Unpack and run the following commands in the eigen-3.3.? directory

```
sudo apt remove libeigen3-dev
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr
make 
sudo make install
```

### Eigen 3 (Ubuntu 18.04+)
sudo apt install libeigen3-dev

### Swig ( Ubuntu 16.04)

Download swig 3.0.12 from https://sourceforge.net/projects/swig/files/swig/swig-3.0.12/

Unpack and run the following commands

```
./configure --prefix=/usr --without-pcre
make -j4
sudo make install
```

### Swig (Ubuntu 18.04+)

```
sudo apt install swig3.0
```

### Compile library

```
cd NativeCommonOps
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
make install
```

## Windows
Use CMake GUI to create the Visual Studio makefiles.
- Start the x64 Native Tools Command Prompt for VS 2019

```
cd [Source directory]\ihmc-matrix-library\NativeCommonOps
md buildc
cd buildc
"C:\Program Files\CMake\bin\cmake.exe" -G "Visual Studio 16 2019" -A x64 -DSWIG_EXECUTABLE="C:\swigwin-3.0.12\swig.exe"  -DSTANDALONE_PLUGIN=ON ..
"C:\Program Files\CMake\bin\cmake.exe" --build . --config Release --target install
```


Note: On Windows, only the Release configuration builds.

## Mac OS X

### Eigen 3
Download the latest Eigen 3.3 release from http://eigen.tuxfamily.org/index.php?title=Main_Page

Unpack and run the following commands in the eigen-3.3.? directory

```
sudo apt remove libeigen3-dev
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/Users/[username]/usr
make 
make install
```

### Swig


Download swig-3.0.12.tar.gz from https://sourceforge.net/projects/swig/files/swig/swig-3.0.12/
Unpack swig-3.0.12.tar.gz in ~/Downloads

```
cd ~/Downloads/swig-3.0.12
./configure --prefix=/Users/[username]/usr --without-pcre
make -j4
make install
```

## Compiling library

```
cd NativeCommonOps
mkdir build
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:/Users/[username]/usr"
cmake -DSWIG_EXECUTABLE=/Users/[username]/usr/bin/swig -DCMAKE_BUILD_TYPE=Release ..
make
make install
```
