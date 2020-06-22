cmake_minimum_required(VERSION 2.8)
project( ColorCorrection )

# Dependenices
find_package( OpenCV REQUIRED )

# Executables
add_executable( ColorCorrection color_correction.cpp )
add_executable( FrontCamera front_camera.cpp )
add_executable( WaterCorrection water_correction.cpp )

# Link targets

target_link_libraries( ColorCorrection ${OpenCV_LIBS} )
target_link_libraries( FrontCamera ${OpenCV_LIBS} )
target_link_libraries( WaterCorrection ${OpenCV_LIBS} )