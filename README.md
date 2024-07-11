# ihmc-matrix-library
IHMC Matrix Library gathers utilities to improve experience the matrix library EJML.

# Usage
Supported platforms:
- Linux (Ubuntu 20.04+ or similar x86_64, arm64)
- Windows (Windows 10+ x86_64)
- macOS (macOS 12+ Intel, Apple Silicon)

Requires Java 17.

## Windows
Requires the installation of Visual C++ 2019 Redistributable (https://aka.ms/vs/16/release/VC_redist.x64.exe).

# Development
To avoid conflicts, we rename the Eigen namespace to us_ihmc_matrix_library_vendor_matrix in "NativeMatrix.h". We then alias Eigen to us_ihmc_matrix_library_vendor_matrix (!). Do not include <Eigen/Dense> or any other Eigen libraries in any files other than the #define Eigen us_ihmc_matrix_library_vendor_matrix block in NativeMatrix.h.

## Compilation
Use the `cppbuild.bash` script to compile for your local platform. This script is compatible with Linux, Windows, and macOS.

Compiling the native library locally should only be done during development of a new feature. For releases, use the GitHub action to build for all platforms.

### Required Tools for Compiling
Linux:
- GNU C Compiler (g++)
- swig
- CMake

Windows:
- Visual Studio 2019
- swig
- CMake

macOS:
- XCode 1.13.1+
- swig
- CMake