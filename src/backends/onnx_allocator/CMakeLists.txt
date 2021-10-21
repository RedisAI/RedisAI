add_library(onnx_allocator STATIC onnx_allocator.cpp)
target_link_libraries(onnx_allocator "${ONNX_LIBRARIES}")
set_property(TARGET onnx_allocator PROPERTY CXX_STANDARD 14)