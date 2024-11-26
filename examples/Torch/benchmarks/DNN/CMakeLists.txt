file(GLOB LIB_SOURCES *.cpp)
file(GLOB LIB_HEADERS *.h)

add_library(dnn SHARED ${LIB_SOURCES} ${LIB_HEADERS})
target_link_libraries(dnn "${TORCH_LIBRARIES}")
set_property(TARGET dnn PROPERTY POSITION_INDEPENDENT_CODE ON)