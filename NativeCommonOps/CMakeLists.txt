cmake_minimum_required(VERSION 3.5)

project(NativeCommonOps)

find_package(JNI REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(SWIG REQUIRED)

include_directories(${JNI_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
include(${SWIG_USE_FILE})

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -s -mno-avx")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set default visibility to hidden. This avoids conflicts when combining with libraries linked to other versions of Eigen.
# Instead of using a global symbol, the Eigen functions are internal to this library
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET hidden)

set(SOURCE_FILES NativeCommonOps.cpp NativeMatrix.cpp NativeNullspaceProjector.cpp NativeKalmanFilter.cpp)



set_source_files_properties(NativeMatrix.i PROPERTIES CPLUSPLUS ON)
set_property(SOURCE NativeMatrix.i PROPERTY SWIG_FLAGS "-package" "us.ihmc.matrixlib.jni")
set(SWIG_OPT_LINKER_FLAGS "")
swig_add_library(NativeCommonOps TYPE SHARED LANGUAGE java OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src/main/java/us/ihmc/matrixlib/jni" SOURCES NativeMatrix.i ${SOURCE_FILES})
set(CMAKE_SWIG_OUTDIR "${CMAKE_CURRENT_SOURCE_DIR}/../src/main/java/us/ihmc/matrixlib/jni")
#swig_add_module(NativeCommonOps java NativeMatrix.i ${SOURCE_FILES})
target_link_libraries(NativeCommonOps Eigen3::Eigen)





install(TARGETS NativeCommonOps RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/../src/main/resources LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/../src/main/resources)
