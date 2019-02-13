cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(libtorch_c)

find_package(Torch REQUIRED)

add_library(torch_c SHARED torch_c.cpp)

target_include_directories(torch_c PRIVATE "${DLPACK_DIRECTORY}/include")
target_link_libraries(torch_c "${TORCH_LIBRARIES}")

set_property(TARGET torch_c PROPERTY CXX_STANDARD 11)

install(TARGETS torch_c DESTINATION lib)
install(FILES torch_c.h DESTINATION include)
