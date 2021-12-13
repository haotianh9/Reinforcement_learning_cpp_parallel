cmake_minimum_required(VERSION 3.4...3.18)
project(ppo_fish)

#target_link_libraries(mymod PRIVATE ${MY_LIBRARIES})

add_subdirectory(pybind11)
pybind11_add_module(ppo_fish ppo_fish_pybind.cpp)

#set(CMAKE_PREFIX_PATH "./libtorch;./eigen_build")

find_package(MPI REQUIRED)
find_package(Torch REQUIRED)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package (Eigen3 REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

#add_executable(ppo_fish main.cpp)
#add_library(ppo_fish_lib SHARED )
#
#add_executable(main memory.cpp cart_pole.cpp)
include_directories(${EIGEN3_INCLUDE_DIR})

message("${CMAKE_MODULE_PATH}")
message("${EIGEN3_INCLUDE_DIR}")
target_link_libraries(ppo_fish PRIVATE MPI::MPI_CXX Eigen3::Eigen "${TORCH_LIBRARIES}")
set_property(TARGET ppo_fish PROPERTY CXX_STANDARD 14)

#find_package(MPI REQUIRED)
#find_package(Torch REQUIRED)
#add_library(ppo_class SHARED ../RL3linkFishCppPortMerge/cart_pole.cpp)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET ppo_fish
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:ppo_fish>)
endif (MSVC)