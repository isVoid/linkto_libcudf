cmake_minimum_required(VERSION 3.18)

project(libcudf_example)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES "")


set(CPM_DOWNLOAD_VERSION 0.27.2) 
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")

if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(STATUS "Downloading CPM.cmake")
    file(DOWNLOAD https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake ${CPM_DOWNLOAD_LOCATION})
endif()

include(${CPM_DOWNLOAD_LOCATION})

set(CUDAToolkit_INCLUDE_DIR "/usr/local/cuda-10.2/include")

set(CUDF_VERSION "0.19")

function(find_and_configure_cudf VERSION)

    CPMAddPackage(NAME  cudf
        VERSION         ${VERSION}
        GIT_REPOSITORY  https://github.com/rapidsai/cudf
        GIT_TAG         branch-0.19
        GIT_SHALLOW     TRUE
        SOURCE_SUBDIR   cpp
        OPTIONS         "BUILD_TESTS OFF"
                        "BUILD_BENCHMARKS OFF"
                        "ARROW_STATIC_LIB ON"
                        "JITIFY_USE_CACHE ON"
                        "CUDA_STATIC_RUNTIME ON"
                        "DISABLE_DEPRECATION_WARNING ON"
                        # control where generated JIT headers get placed
                        # so we don't cache-bust ccache every clean build
                        "CUDF_GENERATED_INCLUDE_DIR ${CPM_SOURCE_CACHE}/cudf-build"
    )
endfunction()

find_and_configure_cudf(${CUDF_VERSION})

add_executable(${PROJECT_NAME} "src/test_cpp_grpby.cpp")
target_link_libraries(${PROJECT_NAME} rmm::rmm cudf::cudf)
