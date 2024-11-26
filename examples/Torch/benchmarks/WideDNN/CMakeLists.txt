file(GLOB LIB_SOURCES *.cpp)
file(GLOB LIB_HEADERS *.h)

add_library(widednn SHARED ${LIB_SOURCES} ${LIB_HEADERS})
target_link_libraries(widednn "${TORCH_LIBRARIES}")
set_property(TARGET widednn PROPERTY POSITION_INDEPENDENT_CODE ON)