
cmake_minimum_required(VERSION 3.5)

project(Fuzzy_Sets ${PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)


find_package(Boost 1.75.0 COMPONENTS date_time timer coroutine thread system unit_test_framework regex REQUIRED)

find_package (Eigen3 3.3 REQUIRED)

find_package (TBB REQUIRED tbb)

file(GLOB SOURCES
    *.hpp
    *.cpp)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -ltbb -Wall -Wextra")
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftemplate-depth=900")

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries (${PROJECT_NAME} Eigen3::Eigen TBB::tbb ${Boost_LIBRARIES})

