project(gAlpha2d)
cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)

#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE Debug)
set(CUDA_NVCC_FLAGS
    "${CUDA_NVCC_FLAGS};
    -gencode arch=compute_50,code=sm_50
    -gencode arch=compute_35,code=sm_35
    -std=c++11")
#add_definitions(-DREAL_TYPE_FP32)
#include_directories(
#    src
#    )
cuda_add_executable(
    ${PROJECT_NAME}
      gpudt/gpudt.cpp
      gpudt/predicates.cpp
      main.cu
      gpudt/Cuda/cudaMissing.cu
      gpudt/Cuda/cudaVoronoi.cu
      gpudt/Cuda/cudaConstraint.cu
      gpudt/Cuda/cudaMain.cu
      gpudt/Cuda/pba2DHost.cu
      gpudt/Cuda/cudaBoundary.cu
      gpudt/Cuda/cudaFlipping.cu
      gpudt/Cuda/cudaShifting.cu
      gpudt/Cuda/cudaReconstruction.cu
      gpudt/Cuda/cudaScanLargeArray.cu
gpudt/AlphaShape/headers.cu
gpudt/AlphaShape/kernel.cu
gpudt/AlphaShape/Alpha_Shapes_2D.cu  
#src/gpudt/AlphaShape/drawBasics.cu  
#src/gpudt/AlphaShape/Visualizer.cu     
   )
set(CMAKE_VERBOSE_MAKEFILE "ON")
target_link_libraries(${PROJECT_NAME} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLUT_LIBRARIES})

