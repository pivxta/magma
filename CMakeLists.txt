cmake_minimum_required(VERSION 3.25)
project(vulkan)

if (NOT IS_DIRECTORY "$ENV{VULKAN_SDK}")
    message(WARNING "Vulkan SDK was not detected.")
endif ()

option(ENABLE_ASAN "Build with AddressSanitizer" OFF)
option(ENABLE_CLANG_TIDY "Run clang-tidy during build" OFF)
option(ENABLE_WERROR "Treat warnings as errors" OFF)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_USE_HYBRID_HPG ON CACHE BOOL "" FORCE)
set(KTX_FEATURE_TESTS OFF CACHE BOOL "" FORCE)
set(KTX_FEATURE_TOOLS OFF CACHE BOOL "" FORCE)
set(KTX_FEATURE_LOADTEST_APPS OFF CACHE BOOL "" FORCE)
set(KTX_FEATURE_DOC OFF CACHE BOOL "" FORCE)

include(FetchContent)
FetchContent_Declare(
    vmahpp
    URL https://github.com/YaaZ/VulkanMemoryAllocator-Hpp/releases/download/v3.3.0%2B3/VulkanMemoryAllocator-Hpp-3.3.0.tar.gz
    URL_HASH SHA256=8c2a0573babe1f86f3241d12b4c9df40bb944649b7f4239c1c611797f8af6cd5
    SYSTEM
)
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog
    GIT_TAG v1.17.0
    SYSTEM
)
FetchContent_Declare(
    tinyobjloader
    GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader
    GIT_TAG v2.0.0rc13
    SYSTEM
)
FetchContent_Declare(
    glm 
    GIT_REPOSITORY https://github.com/g-truc/glm.git 
    GIT_TAG 1.0.3
    SYSTEM
)
FetchContent_Declare(
    glfw 
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
    SYSTEM
)
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG master
)
FetchContent_MakeAvailable(glfw glm spdlog tinyobjloader vmahpp stb)

find_program(SLANGC_EXECUTABLE
    NAMES slangc
    HINTS "$ENV{VULKAN_SDK}/bin"
    REQUIRED
)

function(add_shaders TARGET)
    set(SHADER_TARGETS "")
    set(OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin/shaders")

    foreach(SHADER ${ARGN})
        get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
        set(OUTPUT_SPV "${OUTPUT_DIR}/${SHADER_NAME}.spv")

        add_custom_command(
            OUTPUT ${OUTPUT_SPV}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
            COMMAND ${SLANGC_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER}
                -target spirv -profile spirv_1_4
                -emit-spirv-directly -fvk-use-entrypoint-name
                -o ${OUTPUT_SPV}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER}
            COMMENT "Compiling Slang shader ${SHADER}"
            VERBATIM
        )

        set(SHADER_TARGET "${TARGET}_${SHADER_NAME}_shader")
        add_custom_target(${SHADER_TARGET} DEPENDS ${OUTPUT_SPV})
        list(APPEND SHADER_TARGETS ${SHADER_TARGET})
    endforeach()

    add_dependencies(${TARGET} ${SHADER_TARGETS})
endfunction()

find_package(Vulkan MODULE REQUIRED)

if (WIN32)
    add_compile_definitions(NOMINMAX _USE_MATH_DEFINES)
endif()

if (MSVC)
    add_compile_options(/EHsc)
endif ()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(
    ${PROJECT_NAME} 
    src/main.cpp
    src/window.cpp
    src/window.h
    src/input.h
    src/time.cpp
    src/time.h
    src/renderer.cpp
    src/renderer.h
    src/target.h
    src/buffer.h
    src/imagetexture.h
    src/swapchain.cpp
    src/swapchain.h
    src/performance.h
    src/transform.h
    src/camera.h
    src/image.h
    src/image.cpp
)
add_shaders(${PROJECT_NAME} shaders/shader.slang)
add_compile_definitions(VULKAN_HPP_NO_EXCEPTIONS GLFW_INCLUDE_VULKAN)

if (ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXECUTABLE NAMES clang-tidy REQUIRED)
    set(CLANG_TIDY_COMMAND "${CLANG_TIDY_EXECUTABLE}")
    set_target_properties(${PROJECT_NAME} PROPERTIES
        CXX_CLANG_TIDY "${CLANG_TIDY_COMMAND}")
endif()

if (ENABLE_ASAN)
    if (MSVC)
        target_compile_options(${PROJECT_NAME} PRIVATE /fsanitize=address)
        target_link_options(${PROJECT_NAME} PRIVATE /INCREMENTAL:NO)
    else()
        target_compile_options(${PROJECT_NAME} PRIVATE
            -fsanitize=address -fno-omit-frame-pointer -g)
        target_link_options(${PROJECT_NAME} PRIVATE -fsanitize=address)
    endif()
endif()

if (MSVC) 
    target_compile_options(
        ${PROJECT_NAME} PRIVATE 
        /W4 $<$<BOOL:${ENABLE_WERROR}>:/WX> /wd4458 /wd4324
    )
else ()
    target_compile_options(
        ${PROJECT_NAME} PRIVATE 
        -Wall -Wextra -Wno-missing-field-initializers
        $<$<BOOL:${ENABLE_WERROR}>:-Werror>
    )
endif ()

if (WIN32)
    target_compile_definitions(${PROJECT_NAME} PRIVATE _CRT_SECURE_NO_WARNINGS)
endif ()

if(NOT TARGET stb)
    add_library(stb INTERFACE)
    target_include_directories(stb INTERFACE ${stb_SOURCE_DIR})
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_link_libraries(
    ${PROJECT_NAME} PRIVATE 
    Vulkan::Vulkan
    glm::glm 
    spdlog
    glfw 
    stb
    tinyobjloader
    VulkanMemoryAllocator-Hpp::VulkanMemoryAllocator-Hpp
)