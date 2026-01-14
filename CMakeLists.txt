cmake_minimum_required(VERSION 3.10)
project(BitsetProject)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(bitset_app main.cpp Bitset.cpp)
