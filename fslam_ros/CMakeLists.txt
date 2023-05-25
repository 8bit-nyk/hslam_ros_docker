cmake_minimum_required(VERSION 2.4.6)
project(fslam_ros)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  sensor_msgs
  cv_bridge
)

#SET PATH TO FSLAM

set(FSLAM_PATH /catkin_ws/src/FSLAM)
#set(FSLAM_LIBRARY_PATH ~/catkin_ws/src/FSLAM/lib)
set(FSLAM_LIBRARY_PATH /catkin_ws/src/FSLAM/build/lib)

# set values from environment variables
if (NOT FSLAM_PATH)
    set(FSLAM_PATH $ENV{FSLAM_PATH})
endif()
if (NOT FSLAM_LIBRARY_PATH)
    set(FSLAM_LIBRARY_PATH $ENV{FSLAM_LIBRARY_PATH})
    if (NOT FSLAM_LIBRARY_PATH AND FSLAM_PATH)
        set(FSLAM_LIBRARY_PATH "${FSLAM_PATH}/build/lib")
    endif()
endif()

if (FSLAM_PATH)
    # Manually set path to FSLAM source folder
    message("FSLAM_PATH set, trying to find library.")
    message("---- EXPECTING FSLAM sources at\n\"${FSLAM_PATH}\"")
    set(FSLAM_INCLUDE_DIRS "${FSLAM_PATH}/src" "${FSLAM_PATH}/Thirdparty/Sophus" "${FSLAM_PATH}/Thirdparty/g2o" "${FSLAM_PATH}/Thirdparty/DBow3" "${FSLAM_PATH}/Thirdparty/opencv-3.4.6")
    message("---- LOOKING FOR FSLAM library at\n\"${FSLAM_LIBRARY_PATH}\"")
    find_library(FSLAM_LIBRARY HSLAM_lib ${FSLAM_LIBRARY_PATH})
else()
    # Detect FSLAM install (also catkin devel folder)
    message("FSLAM_PATH not set yet, trying to find installed fslam headers and library.")
    find_path(FSLAM_INCLUDE_DIRS FSLAM)
    if (FSLAM_INCLUDE_DIRS)
        set(FSLAM_INCLUDE_DIRS "${FSLAM_INCLUDE_DIRS}/FSLAM")
        message("---- FOUND FSLAM headers at \"${FSLAM_INCLUDE_DIRS}\"")
    endif()
    find_library(FSLAM_LIBRARY fslam)
endif()


if (NOT FSLAM_INCLUDE_DIRS)
    message(FATAL_ERROR "FSLAM headers not found and/or FSLAM_PATH not set.")
endif()
if (NOT FSLAM_LIBRARY)
    message(FATAL_ERROR "FSLAM library not found and/or FSLAM_LIBRARY_PATH not set.")
endif()

message("---- Found FSLAM library at \"${FSLAM_LIBRARY}\"")


set(CMAKE_MODULE_PATH /catkin_ws/src/fslam_ros/cmake)
set(OpenCV_DIR /catkin_ws/src/FSLAM/Thirdparty/CompiledLibs/share/OpenCV)
set(Pangolin_DIR /catkin_ws/src/FSLAM/Thirdparty/CompiledLibs/lib/cmake)
set(DBoW3_DIR /catkin_ws/src/FSLAM/Thirdparty/CompiledLibs/lib/cmake/DBoW3)
set(G2O_DIR "/catkin_ws/src/FSLAM/Thirdparty/CompiledLibs/lib/cmake/g2o")


find_package(Pangolin 0.2 REQUIRED PATHS /catkin_ws/src/FSLAM/Thirdparty/CompiledLibs/lib/cmake)
message("\n\n---- FOUND Pangolin Headers at\n\"${Pangolin_INCLUDE_DIRS}\"")
message("---- FOUND Pangolin Libs at\n\"${Pangolin_LIBRARIES}\"")

find_package(OpenCV 3.4.6 REQUIRED PATHS /catkin_ws/src/FSLAM/Thirdparty/CompiledLibs/lib/cmake)
message("\n\n---- FOUND OpenCV Headers at\n\"${OpenCV_INCLUDE_DIRS}\"")
message("---- FOUND OpenCV Libs at\n\"${OpenCV_LIBS}\"")

find_package(DBoW3 REQUIRED)
message("\n\n---- FOUND DBoW3 Headers at\n\"${DBoW3_INCLUDE_DIRS}\"")
message("---- FOUND DBoW3 Libs at\n\"${DBoW3_LIBRARIES}\"")

find_package(SuiteParse REQUIRED)
message("\n\n---- FOUND CSPARSE at\n\"${CSPARSE_INCLUDE_DIR}\"")
message("\n\n---- FOUND CSPARSE LIBS at\n\"${CSPARSE_LIBRARY}\"")
message("\n\n---- FOUND CHOLMOD at\n\"${CHOLMOD_INCLUDE_DIR}\"")
message("\n\n---- FOUND CHOLMOD LIBS at\n\"${CHOLMOD_LIBRARIES}\"")

find_package(Eigen3 REQUIRED)
find_package(Boost COMPONENTS system thread)

#G2O not found troubleshooting
find_package(G2O COMPONENTS)
#message("-----G2O troubleshooting-----")
#message("The G2O directory is : ${G2O_DIR}" )
#message("The G2O include  directory is : ${G2O_INCLUDE_DIR}" )
#message("The G2O libraries is : ${G2O_LIBS}" )
#find_package(G2O REQUIRED COMPONENTS PATHS /catkin_ws/src/FSLAM/Thirdparty/CompiledLibs/lib/cmake/g2o)
message ("The following were consodered for g2o: ${G2O_CONSIDERED_CONFIGS}")
message("\n\n---- FOUND G2O Headers at\n\"${G2O_INCLUDE_DIR}\"")
message("---- FOUND G2O Libs at\n\"${G2O_LIBS}\"")

catkin_package(
  CATKIN_DEPENDS
    geometry_msgs
    roscpp
    sensor_msgs
    cv_bridge
)


set(LIBS
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
  ${G2O_LIBS}
  ${Pangolin_LIBRARIES}
  ${DBoW3_LIB_DIR}/libDBoW3.so
  ${BOOST_THREAD_LIBRARY}
  boost_system
  cxsparse
  cholmod
)
###########
## Build ##
###########

set(CMAKE_CXX_FLAGS "-std=c++14 -march=native"
#   "${SSE_FLAGS} -O2 -g -std=c++0x -march=native -fno-omit-frame-pointer"
)

include_directories(
  ${PROJECT_SOURCE_DIR}/src
  ${FSLAM_INCLUDE_DIRS}
  ${Pangolin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
  ${G2O_INCLUDE_DIR}
  ${DBoW3_INCLUDE_DIRS}
  ${CSPARSE_INCLUDE_DIR}
  ${CHOLMOD_INCLUDE_DIR}
  
)

link_directories(
  ${G2O_INCLUDE_DIR}
  ${DBoW3_INCLUDE_DIR}
  ${CSPARSE_INCLUDE_DIR}
  ${CHOLMOD_INCLUDE_DIR}
)
## Declare a C++ executable
add_executable(fslam_live src/main.cpp)

target_link_libraries(fslam_live
    ${FSLAM_LIBRARY}
    ${Pangolin_LIBRARIES}
    ${OpenCV_LIBS}
    ${G2O_LIBS}
    ${DBoW3_LIB_DIR}/libDBoW3.so
    ${catkin_LIBRARIES}
    ${CSPARSE_LIBRARY}
    ${CHOLMOD_LIBRARIES}
    
   )
