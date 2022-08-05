cmake_minimum_required(VERSION 3.15)

# Sets paths that CMake checks for cmake modules [include (...)]
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Output binary to predictable location.
set(BINARY_OUT_DIR ${CMAKE_BINARY_DIR}/bin)
set(LIB_OUT_DIR ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BINARY_OUT_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LIB_OUT_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${LIB_OUT_DIR})

# Set a default build type if none was specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Debug' as none was specified.")
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build." FORCE)

    # Set the possible values of build type for cmake-gui, ccmake
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

# Predeclared options used later. Prefer changing then in the build-type specific ifs.
option(BP_ENABLE_PCH "Enable Precompiled Headers" OFF) # @todo When to enable this? And what to put in here?
option(BP_USE_UBSAN "Enable the Undefined Behavior Sanitizers" OFF)
option(BP_USE_MSAN "Enable the Memory Sanitizers" OFF)
option(BP_USE_TSAN "Enable the Thread Sanitizers" OFF)
option(BP_USE_LSAN "Enable the Integer Overflow Sanitizers" OFF)
option(BP_USE_SAFE_STACK "Enable the Safe Stack Separation Sanitizers" OFF)
option(BP_USE_ASAN "Enable the Address Sanitizers" OFF)
option(BP_ENABLE_IPO "Enable Interprocedural optimization, aka Link Time Optimization (LTO)" OFF)
option(BP_ENABLE_CPPCHECK "Enable static analysis with cppcheck" OFF)
option(BP_ENABLE_CLANG_TIDY "Enable static analysis with clang-tidy" OFF)
option(BP_WARNING_AS_ERROR "Treats compiler warnings as errors" ON)
option(BP_USE_CONAN "Use conan to manage dependencies" OFF)
option(BP_FORCE_COLOR "Always produce ANSI-colored output (GNU/Clang only)." ON)
option(BP_USE_MOLD "Use the mold linker" OFF)
option(BP_DISABLE_ALL_WARNINGS "Disable all warnings regardless or prior options. For prototyping." OFF)

# ----------------------------------- Configurable stuff -----------------------------------

# @extension Enable static analyzers
# set(BP_ENABLE_CLANG_TIDY ON)
# set(BP_ENABLE_CPPCHECK ON)

# @extension Change the project name
project(pause_codegen LANGUAGES CXX)

# @extension Add options for different build types
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  # I think -Werror isn't that useful. I don't tend to ignore warnings anyway
  set(BP_WARNING_AS_ERROR OFF)

  # @todo Some of these say that you should use at leat -O1, so check perf
  # @note Sanitizers can only be combined in the following way:
  #   - address, undefined, integer can be combined. Integer only if no intentional overflow occur)
  #   - memory isn't compatible with the others. Enable separately, or use valgrind
  #   - thread isn't compatible either.
  set(BP_USE_ASAN  ON) # @note ASAN includes the leak sanitizer
  set(BP_USE_UBSAN ON)
  set(BP_USE_LSAN  ON)

  set(BP_USE_TSAN  OFF)
  set(BP_USE_MSAN  OFF)
  set(BP_USE_SAFE_TACK OFF)

  set(BP_ENABLE_IPO OFF)

  set(BP_DISABLE_ALL_WARNINGS ON)
endif()
if (CMAKE_BUILD_TYPE STREQUAL "Release") #@todo Enable the proper sanitizers for other build types. E.g. safe-stack always
  set(BP_ENABLE_IPO ON) # @todo Test if this works
endif()
if (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  set(BP_ENABLE_IPO ON)
endif()
if (CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
  set(BP_ENABLE_IPO OFF)
endif()

# @extension Add subdirectories here
set(JSON_BuildTests OFF CACHE INTERNAL "")
add_subdirectory(nlohmann_json)

# @extension Change target name
set (cmake_cpp_boilerplate_target_name "bp_app")

# @extension Change the main source file name
set (cmake_cpp_boilerplate_main_source "main.cpp")

# @extension Add libraries that should be linked against the target here
set (cmake_cpp_linked_libs "nlohmann_json::nlohmann_json")

# @exrension Add additional include paths here
set (cmake_cpp_boilerplate_include_dirs "include")

# # ----------------------------------- End of configurable -----------------------------------

# Where the linker checks for libraries
link_directories(${CMAKE_CURRENT_BINARY_DIR}/lib)

# Use Ccache for faster rebuilds
find_program(CCACHE ccache)
if (CCACHE)
    message("using ccache")
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
endif ()

# Generate compile_commands.json to make it easier to work with clang based
# tools
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# LTO option
if (BP_ENABLE_IPO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
    if (result)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else ()
        message(SEND_ERROR "IPO is not supported: ${output}")
    endif ()
endif ()

if(BP_ENABLE_CPPCHECK)
    find_program(CPPCHECK cppcheck)
    if(CPPCHECK)
        set(CMAKE_CXX_CPPCHECK ${CPPCHECK} --suppress=missingInclude --enable=all
                --inconclusive -i ${CMAKE_SOURCE_DIR}/imgui/lib)
    else()
        message(SEND_ERROR "cppcheck requested but executable not found")
    endif()
endif()

# Optionally use clang-tidy.
# @todo This is a bit finicky with older versions, so not sure how useful it is
# I mostly get this info anyway when programming with VSCode
if(BP_ENABLE_CLANG_TIDY)
    find_program(CLANGTIDY clang-tidy)
    if(CLANGTIDY)
        set(CMAKE_CXX_CLANG_TIDY ${CLANGTIDY})
    else()
        message(SEND_ERROR "clang-tidy requested but executable not found")
    endif()
endif()

# Conan configuration
macro(run_conan)
    if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
        message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
        file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/master/conan.cmake"
                "${CMAKE_BINARY_DIR}/conan.cmake")
    endif()

    include(${CMAKE_BINARY_DIR}/conan.cmake)

    conan_cmake_autodetect(settings)
    conan_cmake_install(PATH_OR_REFERENCE ${PROJECT_SOURCE_DIR}
            BUILD missing
            SETTINGS ${settings}
            OUTPUT_QUIET
            )
endmacro()

if(BP_USE_CONAN) 
	run_conan()
endif()

# ----------- Compiler options and configurations for different compilers -----------
# Link this 'library' to use the standard warnings
add_library(cmake_cpp_boilerplate_compiler_options INTERFACE)


if(MSVC)
  target_compile_options(cmake_cpp_boilerplate_compiler_options INTERFACE /W4 "/permissive-")
  if(BP_WARNING_AS_ERROR)
    target_compile_options(cmake_cpp_boilerplate_compiler_options INTERFACE /WX)
  endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")

target_compile_options(cmake_cpp_boilerplate_compiler_options INTERFACE      			   
                                   -g
								   -Wall
                                   -Wextra
                                   -Wshadow
                                   -Wnon-virtual-dtor
                                   -Wold-style-cast
                                   -Wcast-align
                                   -Wunused
                                   -Woverloaded-virtual
                                   -Wpedantic
                                   -Wconversion
                                   -Wsign-conversion
                                   -Wnull-dereference
                                   -Wdouble-promotion
                                   -Wformat=2
								   -Weffc++
								   -pedantic-errors
								   -Wcast-align
								   -Wcast-qual
								   -Wchar-subscripts  
								   -Wcomment 
								   -Wconversion
								   -Wdisabled-optimization 
								   -Wfloat-equal  
								   -Wformat  
								   -Wformat-nonliteral 
								   -Wformat-security 
								   -Wformat-y2k
								   -Wimport  
								   -Winit-self  
								   -Winline
								   -Winvalid-pch 
								   -Wmissing-braces
								   -Wmissing-field-initializers
								   -Wmissing-format-attribute 
								   -Wmissing-include-dirs
								   -Wmissing-noreturn 
								   -Wpacked
								   -Wparentheses
								   -Wpointer-arith
								   -Wredundant-decls
								   -Wreturn-type
								   -Wsequence-point
								   -Wshadow
								   -Wsign-compare
								   -Wstack-protector
								   -Wstrict-aliasing
								   -Wstrict-aliasing=2
								   -Wswitch
								   -Wswitch-default
								   -Wswitch-enum
								   -Wtrigraphs
								   -Wuninitialized
								   -Wunknown-pragmas
								   -Wunreachable-code
								   -Wunused
								   -Wunused-function
								   -Wunused-label
								   -Wunused-parameter
								   -Wunused-value
								   -Wunused-variable
								   -Wvariadic-macros
								   -Wvolatile-register-var
								   -Wwrite-strings)


  if(BP_WARNING_AS_ERROR)
    message("Using warnings as errors")
	  target_compile_options(cmake_cpp_boilerplate_compiler_options INTERFACE -Werror)
  endif()

  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  target_compile_options(cmake_cpp_boilerplate_compiler_options INTERFACE
                                  -Wmisleading-indentation
                                  -Wduplicated-cond
                                  -Wduplicated-branches
                                  -Wlogical-op
                                  -Wuseless-cast
								  -Wunsafe-loop-optimizations
                           )
  endif()
endif()


if(BP_USE_LSAN)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")  
    message("Enable Integer Overflow Sanitizer")
    target_compile_options(cmake_cpp_boilerplate_compiler_options INTERFACE -fsanitize=integer)
    target_link_libraries(cmake_cpp_boilerplate_compiler_options INTERFACE  -fsanitize=integer)
  else()
    message("Integer Overflow sanitizer enabled but compiler is not clang. Ignoring")
  endif()
endif()

if(BP_USE_ASAN)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")  
    message("Enable Address Sanitizer")
    target_compile_options(cmake_cpp_boilerplate_compiler_options INTERFACE -fsanitize=address -fno-omit-frame-pointer)
    target_link_libraries(cmake_cpp_boilerplate_compiler_options INTERFACE  -fsanitize=address)
  else()
    message("Address sanitizer enabled but compiler is not clang. Ignoring")
  endif()
endif()

if(BP_USE_TSAN)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")  
    message("Enable Thread Sanitizer")
    target_compile_options(cmake_cpp_boilerplate_compiler_options INTERFACE -fsanitize=thread)
    target_link_libraries(cmake_cpp_boilerplate_compiler_options INTERFACE  -fsanitize=thread)
  else()
    message("Thread sanitizer enabled but compiler is not clang. Ignoring")
  endif()
endif()

if(BP_USE_MSAN)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")  
    message("Enable Memory Sanitizer")
    target_compile_options(cmake_cpp_boilerplate_compiler_options INTERFACE -fsanitize=memory -fno-omit-frame-pointer)
    target_link_libraries(cmake_cpp_boilerplate_compiler_options INTERFACE  -fsanitize=memory)
  else()
    message("Memory sanitizer enabled but compiler is not clang. Ignoring")
  endif()
endif()

if(BP_USE_UBSAN)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")  
    message("Enable Undefined Behavior Sanitizer")
    target_compile_options(cmake_cpp_boilerplate_compiler_options INTERFACE -fsanitize=undefined)
    target_link_libraries(cmake_cpp_boilerplate_compiler_options INTERFACE  -fsanitize=undefined)
  else()
    message("UB sanitizer enabled but compiler is not clang. Ignoring")
  endif()
endif()

if(BP_USE_SAFE_STACK)
  # Splits stack into safe and unsafe stack, so overflows can't go between them (would segfault)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message("Enable Safe Stack Sanitizer")
    target_compile_options(cmake_cpp_boilerplate_compiler_options INTERFACE -fsanitize=safe-stack)
    target_link_libraries(cmake_cpp_boilerplate_compiler_options  INTERFACE -fsanitize=safe-stack)
  else()
    message("Safe Stack sanitizer enabled but compiler is not clang. Ignoring")
  endif()
endif()

find_program(MOLD mold)
if(MOLD AND BP_USE_MOLD)
    message("Using the mold linker")
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      target_link_libraries(cmake_cpp_boilerplate_compiler_options INTERFACE -fuse-ld=mold)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
      target_link_libraries(cmake_cpp_boilerplate_compiler_options INTERFACE -B/usr/libexec/mold)
    else()
      message(SEND_ERROR "Using a compiler that can't use the mold linker")
    endif()
endif()

include_directories(${cmake_cpp_boilerplate_include_dirs})

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

add_executable(${cmake_cpp_boilerplate_target_name} ${cmake_cpp_boilerplate_main_source})

# Find the libraries that correspond to the LLVM components
# that we wish to use
llvm_map_components_to_libnames(llvm_libs support core irreader)


target_link_libraries(${cmake_cpp_boilerplate_target_name} PUBLIC 
                      cmake_cpp_boilerplate_compiler_options
                      ${cmake_cpp_linked_libs} ${llvm_libs})

# Force the compiler to generate colored output (GNU/Clang only). Needed for ninja to show colors
if (BP_FORCE_COLOR)
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(${cmake_cpp_boilerplate_target_name} PRIVATE "-fdiagnostics-color=always")
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(${cmake_cpp_boilerplate_target_name} PRIVATE "-fcolor-diagnostics")
  endif ()
endif ()

# @todo When to activate these? And what to put in here?
if (BP_ENABLE_PCH)
	  target_precompile_headers(${cmake_cpp_boilerplate_target_name} INTERFACE
			  <algorithm>
			  <array>
			  <vector>
			  <string>
			  <utility>
			  <functional>
			  <memory>
			  <memory_resource>
			  <string_view>
			  <cmath>
			  <cstddef>
			  <type_traits>
			  )
endif ()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

# TODO:
# - Add ability to use a testing framework
# - Code coverage tool
# - CPack. I think that's a tool for generating packages
# - Support multiple targets

