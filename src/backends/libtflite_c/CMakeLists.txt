add_library(tflite_c STATIC tflite_c.cpp)
message(${TFLITE_LIBRARIES})
target_link_libraries(tflite_c "${TFLITE_LIBRARIES}")
set_property(TARGET tflite_c PROPERTY CXX_STANDARD 11)
