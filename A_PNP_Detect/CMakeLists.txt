cmake_minimum_required(VERSION 3.16)
project(myTest)

set(CMAKE_CXX_STANDARD 20)

set(WITH_CUDA ON)
set(WITH_OPENVINO OFF)
set(GRAPHIC_DEBUG ON)
set(WITH_SERIAL OFF)
set(SAVE_VIDEO OFF)

# realsense
include_directories(/usr/include)

# opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#Eigen
include_directories(/usr/include/eigen3)

# 寻找 Ceres Solver
find_package(Ceres REQUIRED)

include_directories(/home/wenhan/ceres-solver/include)

include_directories(/home/wenhan/abseil_install/include)


find_package(absl REQUIRED PATHS /home/wenhan/abseil_install)

if (WITH_CUDA)
    # cuda & cudnn
    include_directories(/usr/local/cuda/include)
    # tensorrt
    include_directories(/usr/include/x86_64-linux-gnu)
    include_directories(/home/wenhan/TensorRT-8.6.1.6/include)
    file(GLOB TENSORRT_LIBS /home/wenhan/TensorRT-8.6.1.6/lib/*)
    # cvcuda & nvcv_types
    include_directories(/home/wenhan/cvcuda/include)
    file(GLOB CVCUDA_LIBS /home/wenhan/cvcuda/lib/x86_64-linux-gnu/*.so)
    # libtorch
    include_directories(/home/wenhan/libtorch/include)
    include_directories(/home/wenhan/libtorch/include/torch/csrc/api/include)
    file(GLOB TORCH_LIBS /home/wenhan/libtorch/lib/*.so)
    # torchvision
    # include_directories(/home/mei23/torchvision/include)
    # file(GLOB TORCHVISION_LIBS /home/mei23/torchvision/lib/*.so)
    # add definition
    add_definitions(-DWITH_CUDA)
    message("Using cuda")
elseif (WITH_OPENVINO)
    # openvino
    set(OpenVINO_DIR /home/mei23/openvino/runtime/cmake)
    find_package(OpenVINO REQUIRED)
    include_directories(/home/mei23/openvino/runtime/include)
    # add definition
    add_definitions(-DWITH_OPENVINO)
    message("Using openvino")
endif ()

# MEI
if (GRAPHIC_DEBUG)
    add_definitions(-DGRAPHIC_DEBUG)
endif ()

if (WITH_SERIAL)
    add_definitions(-DWITH_SERIAL)
endif ()

if (SAVE_VIDEO)
    add_definitions(-DSAVE_VIDEO)
endif ()

include_directories(include)
file(GLOB_RECURSE SRC_FILES src/*.cpp)
add_executable(${PROJECT_NAME}
        ${SRC_FILES}
        main.cpp
)

target_link_libraries(${PROJECT_NAME}
        ${CMAKE_DL_LIBS}
        ${OpenCV_LIBS}
        realsense2
        pthread
        Ceres::ceres
)


if (WITH_CUDA)
    target_link_libraries(${PROJECT_NAME}
            cudart
            nvinfer
            nvinfer_plugin
            ${CVCUDA_LIBS}
            ${TORCH_LIBS}
            ${TORCHVISION_LIBS}
    )
elseif (WITH_OPENVINO)
    target_link_libraries(${PROJECT_NAME}
            openvino::runtime
    )
endif ()
