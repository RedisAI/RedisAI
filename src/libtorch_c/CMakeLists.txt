add_library(torch_c STATIC torch_c.cpp)
target_link_libraries(torch_c "${TORCH_LIBRARIES}")
set_property(TARGET torch_c PROPERTY CXX_STANDARD 11)
