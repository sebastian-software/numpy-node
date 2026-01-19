cmake_minimum_required(VERSION 3.15)
project(numpy_node_native)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Node.js and cmake-js
include_directories(${CMAKE_JS_INC})

# Platform detection
if(APPLE)
    message(STATUS "Building for macOS")
    set(USE_ACCELERATE ON)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
elseif(WIN32)
    message(STATUS "Building for Windows")
    set(USE_OPENBLAS ON)
else()
    message(STATUS "Building for Linux")
    set(USE_OPENBLAS ON)
endif()

# Source files
set(SOURCES
    src/binding.cc
    src/ndarray.cc
    src/linalg.cc
    src/math_ops.cc
    src/random.cc
    src/fft_ops.cc
)

# Header files
set(HEADERS
    src/ndarray.h
    src/linalg.h
    src/math_ops.h
    src/random.h
    src/fft_ops.h
)

# Create the native addon
add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS} ${CMAKE_JS_SRC})
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")

# Include node-addon-api
execute_process(COMMAND node -p "require('node-addon-api').include"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/..
    OUTPUT_VARIABLE NODE_ADDON_API_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REPLACE "\"" "" NODE_ADDON_API_DIR ${NODE_ADDON_API_DIR})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${NODE_ADDON_API_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

# Platform-specific configuration
if(USE_ACCELERATE)
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_ACCELERATE=1)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${ACCELERATE_FRAMEWORK})

    # macOS-specific flags
    target_compile_options(${PROJECT_NAME} PRIVATE
        -march=native
        -O3
        -ffast-math
    )
elseif(USE_OPENBLAS)
    find_package(BLAS)
    find_package(LAPACK)

    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_OPENBLAS=1)

    if(BLAS_FOUND)
        target_link_libraries(${PROJECT_NAME} PRIVATE ${BLAS_LIBRARIES})
    endif()
    if(LAPACK_FOUND)
        target_link_libraries(${PROJECT_NAME} PRIVATE ${LAPACK_LIBRARIES})
    endif()
endif()

# Link against cmake-js
target_link_libraries(${PROJECT_NAME} PRIVATE ${CMAKE_JS_LIB})

# N-API version
target_compile_definitions(${PROJECT_NAME} PRIVATE
    NAPI_VERSION=8
    NAPI_DISABLE_CPP_EXCEPTIONS
)

# Installation
install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
