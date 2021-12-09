cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(Network)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# SET( EIGEN3_INCLUDE_DIR "$ENV{EIGEN3_INCLUDE_DIR}" )
# IF( NOT EIGEN3_INCLUDE_DIR )
#     MESSAGE( FATAL_ERROR "Please point the environment variable EIGEN3_INCLUDE_DIR to the include directory of your Eigen3 installation.")
# ENDIF()
# INCLUDE_DIRECTORIES ( "${EIGEN3_INCLUDE_DIR}" )

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(./)
add_executable(Network Network.cpp)
target_link_libraries(Network "${TORCH_LIBRARIES}")
target_link_libraries (Network Eigen3::Eigen)
set_property(TARGET Network PROPERTY CXX_STANDARD 14)


# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET Network
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:Network>)
endif (MSVC)