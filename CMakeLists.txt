cmake_minimum_required(VERSION 3.10)
project(GraphConverter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)
find_package(BZip2 REQUIRED)

# Add your source files
add_executable(graph_converter main.cpp)
add_executable(dijkstra dijkstra.cpp)
add_executable(verify_graph verify_graph.cpp)

# Include libosmium headers directly
target_include_directories(graph_converter PRIVATE 
    /opt/homebrew/include
    /opt/homebrew/Cellar/libosmium/2.22.0/include
    /opt/homebrew/opt/expat/include
)

# Link required libraries
target_link_libraries(graph_converter PRIVATE 
    Threads::Threads
    ZLIB::ZLIB
    BZip2::BZip2
    /opt/homebrew/lib/liblz4.dylib
    /opt/homebrew/opt/expat/lib/libexpat.dylib
)

# Dijkstra only needs threading support
target_link_libraries(dijkstra PRIVATE 
    Threads::Threads
)