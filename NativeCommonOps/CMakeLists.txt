cmake_minimum_required(VERSION 3.1)

project(NativeCommonOps)

find_package(JNI REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(${JNI_INCLUDE_DIRS})

# This requires the build to be in release mode: Eigen does allocate on the heap unless optimized.
add_definitions(-DEIGEN_NO_MALLOC)

set(CMAKE_CXX_STANDARD 11)
set(SOURCE_FILES NativeCommonOps.cpp)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../src/main/resources)

add_library(NativeCommonOps SHARED ${SOURCE_FILES})
target_link_libraries (NativeCommonOps Eigen3::Eigen)

install(TARGETS NativeCommonOps RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/../src/main/resources LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/../src/main/resources)
