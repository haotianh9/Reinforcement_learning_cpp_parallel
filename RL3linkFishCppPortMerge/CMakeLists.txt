cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)

find_package(MPI REQUIRED) 	

find_package(Torch REQUIRED)


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
# find_package (Eigen3 REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(example-app ./cart_pole.cpp ./ppo_nn.cpp)
# include_directories(${EIGEN3_INCLUDE_DIR})

message("${CMAKE_MODULE_PATH}")
# message("${EIGEN3_INCLUDE_DIR}")
# target_link_libraries(example-app PRIVATE  MPI::MPI_CXX Eigen3::Eigen "${TORCH_LIBRARIES}" )
target_link_libraries(example-app PRIVATE  MPI::MPI_CXX  "${TORCH_LIBRARIES}" )
set_property(TARGET example-app PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
	  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
	    add_custom_command(TARGET example-app
		                         POST_BUILD
					                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
							                           ${TORCH_DLLS}
										                        $<TARGET_FILE_DIR:example-app>)
											endif (MSVC)
