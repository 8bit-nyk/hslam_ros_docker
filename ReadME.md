# H-SLAM (Hybrid Simultaneous Localization and Mapping).
### This is an outdated version of the HSLAM project, for the most recent development repository please refer to [HSLAM Dev](https://github.com/8bit-nyk/hslam_ros2)
A containerized "ready-to-use" SLAM application that leverages both direct and indirect methods.

### Related Publications:

Please cite the paper if used in an academic context.

[H-SLAM: Hybrid direct--indirect visual SLAM](https://doi.org/10.1016/j.robot.2024.104729)
 ```bibtex
    @article{younes2024h,
    title={H-SLAM: Hybrid direct--indirect visual SLAM},
    author={Younes, Georges and Khalil, Douaa and Zelek, John and Asmar, Daniel},
    journal={Robotics and Autonomous Systems},
    volume={179},
    pages={104729},
    year={2024},
    publisher={Elsevier}
    }
 ```
#### Other Related Works

[Inline Photometrically Calibrated Hybrid Visual SLAM](https://doi.org/10.1109/IROS58592.2024.10802153)


```bibtex
@inproceedings{abboud2024inline,
  title={Inline Photometrically Calibrated Hybrid Visual SLAM},
  author={Abboud, Nicolas and Sayour, Malak and Elhajj, Imad H and Zelek, John and Asmar, Daniel},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={10089--10096},
  year={2024},
  organization={IEEE}
}
```

## Table of Contents

- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)

## Project Description

The H-SLAM project is an implementation of a visual simultaneous localization and mapping algorithm. 
It utilizes the ROS (Robot Operating System) Noetic and is designed for Ubuntu 20.04.



## Installation

To use this Dockerfile and build the H-SLAM project, follow these steps:

1. Install Docker on your machine.
2. Create a new directory and navigate to it in the terminal.
3. Clone this repository into the previously created directory.
4. Open a terminal and navigate to the directory containing the Dockerfile and project files.
```bash
    cd ~/<YOUR_PATH>/hslam_ros_docker
```
6. Run the following command to build the Docker image:

```bash
    docker build -t hslam .
```
The build will start using the Dockerfile structure, which is divided into multiple stages. 
In the first stage, it sets up the base image with Ubuntu 20.04 and ROS Noetic using OSRF official noetic image. It also installs the necessary dependencies for catkin tools.
In the second stage, it installs additional dependencies and the Realsense SDK. These dependencies include various libraries and tools required for camera support, graphics, system functionality, and GUI elements. Additionally, it downloads and extracts third-party libraries, including Ceres Solver and OpenCV with OpenCV Contrib.
The project files are then copied into the container, and the Thirdparty libraries are built using the provided `build.sh` script. After that, the H-SLAM project is built using CMake, and the `hslam_ros` wrapper is copied and built using catkin.
Finally, the container's entry point is specified, and calibration files are copied into the project directory.

## Usage

To run the H-SLAM project you will need to run two containers of the same image.
One to to publish images from the camera and the other to run the H-SLAM main application.

0. Allow access to containers:
``` bash
xhost +
```

1. Open two terminals and execute the following command in each:
``` bash
docker run -it --net=host --privileged -e DISPLAY=unix$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw --device /dev/video0:/dev/video0  hslam /bin/bash
```
This command starts the container and provides an interactive terminal within it.

2. In the first terminal run the following command to open a camera stream through ROS and publish the camera's images unto a ROS topic:
``` bash
roslaunch usb_cam usb_cam-test.launch
```
A display window will pop with the camera's stream.

3. In the second terminal, execute this command to run the H-SLAM algorithm on the image stream.
``` bash
rosrun hslam_ros hslam_live image:=/usb_cam/image_raw calib=/catkin_ws/src/res/camera.txt gamma=/catkin_ws/src/res/pcalib.txt vignette=/catkin_ws/src/res/vignette.png
```

Start moving the camera around and perform realtime Visual SLAM!

### Results:

The HSLAM system output two files when it exists:
1. **result.txt** file that contains corrected trajectory.
2. **map.pcd** file containing point cloud of the contrusted map. 

Additionally, the following data is published over the ROS network:
1. **/hslam_path**: publishes the current camera pose.
2. **/hslam_pose**: published the path tracked so far.
3. **/hslam_map**: publishes the map redered so far.

You should get an output similair to screenshot below showing the Visual Monocular SLAM process:
  ![hslam-output-screenshot](https://github.com/8bit-nyk/HSLAM/assets/49674476/ab3a1c94-8f38-41da-855d-c25566f720c4)
  
## Features

- Utilizes the H-SLAM algorithm for simultaneous localization and mapping.
- Integrates with ROS Noetic and utilizes various ROS functionalities.
- Supports camera integration, including Realsense cameras.
- Provides a wrapper for ROS integration and additional functionality.

## Contributing

Contributions to the H-SLAM project are welcome. If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make the necessary changes and commit them.
4. Push your changes to your forked repository.
5. Submit a pull request detailing the changes you have made.

## License 
This repository is licensed under the GNU General Public License version 3 [GPLv3](LICENSE).

This work is a joint collaborative effort between the:

- Vision and Robotics Lab at the American University of Beirut (AUB)
- Vision and Image Processing Group at the University of Waterloo (UW)
