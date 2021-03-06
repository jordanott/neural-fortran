# cmake version, project name, language
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(neural-fortran Fortran)

# set output paths for modules, archives, and executables
set(CMAKE_Fortran_MODULE_DIRECTORY ${PROJECT_BINARY_DIR}/include)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# if build type not specified, default to release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "release")
endif()

# handle integer size
if(INT)
  message(STATUS "Configuring build for ${INT}-bit integers")
  add_definitions(-DINT${INT})
else()
  message(STATUS "Configuring build for 32-bit integers")
  add_definitions(-DINT32)
endif()

# handle real size
if(REAL)
  message(STATUS "Configuring build for ${REAL}-bit reals")
  add_definitions(-DREAL${REAL})
else()
  message(STATUS "Configuring build for 32-bit reals")
  add_definitions(-DREAL32)
endif()

if(SERIAL)
  message(STATUS "Configuring build for serial execution")
else()
  message(STATUS "Configuring build for parallel execution")
  add_definitions(-DCAF)
endif()

# compiler flags for gfortran
if(CMAKE_Fortran_COMPILER_ID MATCHES GNU)

  if(SERIAL)
    message(STATUS "Configuring to build with -fcoarray=single")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fcoarray=single")
  endif()

  if(BLAS)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fexternal-blas ${BLAS}")
    set(LIBS "${LIBS} blas")
    message(STATUS "Configuring build to use BLAS from ${BLAS}")
  endif()

  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -cpp -fopenmp")
  set(CMAKE_Fortran_FLAGS_DEBUG "-O0 -g -C -fbacktrace")
  set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -ffast-math")
endif()

# compiler flags for ifort
if(CMAKE_Fortran_COMPILER_ID MATCHES Intel)

  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fpp -assume byterecl,realloc_lhs -heap-arrays -qopenmp")
  set(CMAKE_Fortran_FLAGS_DEBUG "-O0 -g -C -traceback")
  set(CMAKE_Fortran_FLAGS_RELEASE "-O3")

  if(NOT SERIAL)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -coarray=shared")
  endif()

endif()

# compiler flags for Cray ftn
if(CMAKE_Fortran_COMPILER_ID MATCHES Cray)
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -h noomp")
  set(CMAKE_Fortran_FLAGS_DEBUG "-O0 -g")
  set(CMAKE_Fortran_FLAGS_RELEASE "-O3")
endif()

# library to archive (libneural.a)
add_library(neural src/lib/mod_activation.F90 src/lib/mod_io.F90 src/lib/mod_kinds.F90 src/lib/mod_layer.F90 src/lib/mod_dense_layer.F90 src/lib/mod_dropout_layer.F90 src/lib/mod_batchnorm_layer.F90 src/lib/mod_mnist.F90 src/lib/mod_network.F90 src/lib/mod_ensemble.F90 src/lib/mod_parallel.F90 src/lib/mod_random.F90)

# Remove leading or trailing whitespace
string(REGEX REPLACE "^ | $" "" LIBS "${LIBS}")

# tests
enable_testing()
#  mnist network_save network_sync set_activation_function
foreach(execid keras bulk ensembles training)
  add_executable(test_${execid} src/tests/test_${execid}.F90)
  target_link_libraries(test_${execid} neural ${LIBS})
  add_test(test_${execid} bin/test_${execid})
endforeach()

# foreach(execid mnist save_and_load simple sine)
#   add_executable(example_${execid} src/tests/example_${execid}.F90)
#   target_link_libraries(example_${execid} neural ${LIBS})
#   add_test(example_${execid} bin/example_${execid})
# endforeach()
