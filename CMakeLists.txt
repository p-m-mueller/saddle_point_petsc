cmake_minimum_required(VERSION 3.11)

project(saddle_point_petsc VERSION 1.0)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)

if(EXISTS ${PETSC_DIR})
	set(CMAKE_PREFIX_PATH "${PETSC_DIR}")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")
find_package(PETSC 3.7 REQUIRED)

find_package(MPI 3.0 REQUIRED)

find_package(HDF5 1.10 REQUIRED)

add_subdirectory(src)
