[ ![ihmc-matrix-library](https://api.bintray.com/packages/ihmcrobotics/maven-release/ihmc-matrix-library/images/download.svg) ](https://bintray.com/ihmcrobotics/maven-release/ihmc-matrix-library/_latestVersion)
[ ![buildstatus](https://bamboo.ihmc.us/plugins/servlet/wittified/build-status/LIBS-IHMCMATRIXLIBRARY)](https://bamboo.ihmc.us/plugins/servlet/wittified/build-status/LIBS-IHMCMATRIXLIBRARY)

IHMC Matrix Library gathers utilities to improve experience the matrix library EJML.


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
