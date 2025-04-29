file(GLOB LIB_SOURCES *.cpp)
file(GLOB LIB_HEADERS *.h)

add_library(transformer SHARED ${LIB_SOURCES} ${LIB_HEADERS})
target_link_libraries(transformer "${TORCH_LIBRARIES}")
set_property(TARGET transformer PROPERTY POSITION_INDEPENDENT_CODE ON)