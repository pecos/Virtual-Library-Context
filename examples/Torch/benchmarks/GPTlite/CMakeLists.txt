file(GLOB LIB_SOURCES *.cpp)
file(GLOB LIB_HEADERS *.h)

add_library(gptlite SHARED ${LIB_SOURCES} ${LIB_HEADERS})
target_link_libraries(gptlite "${TORCH_LIBRARIES}")
set_property(TARGET gptlite PROPERTY POSITION_INDEPENDENT_CODE ON)