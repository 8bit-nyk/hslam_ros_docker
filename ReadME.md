# Project Title

FSLAM (FastSLAM) Project

## Table of Contents

- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)

## Project Description

The FSLAM project is an implementation of a visual simultaneous localization and mapping algorithm. It utilizes the ROS (Robot Operating System) Noetic and is designed for Ubuntu 20.04.

This Dockerfile is divided into multiple stages. In the first stage, it sets up the base image with Ubuntu 20.04 and ROS Noetic. It also installs the necessary dependencies for catkin tools.

In the second stage, it installs additional dependencies and the Realsense SDK. These dependencies include various libraries and tools required for camera support, graphics, system functionality, and GUI elements. Additionally, it downloads and extracts third-party libraries, including Ceres Solver and OpenCV with OpenCV Contrib.

The project files are then copied into the container, and the Thirdparty libraries are built using the provided `build.sh` script. After that, the FSLAM project is built using CMake, and the `fslam_ros` wrapper is copied and built using catkin.

Finally, the container's entry point is specified, and calibration files are copied into the project directory.

## Installation

To use this Dockerfile and build the FSLAM project, follow these steps:

1. Install Docker on your machine.
2. Create a new directory and navigate to it in the terminal.
3. Copy the Dockerfile and the project files to the created directory.
4. Open a terminal and navigate to the directory containing the Dockerfile and project files.
5. Run the following command to build the Docker image:

```bash
    docker build -t fslam .
```
## Usage

To run the FSLAM project within the Docker container, execute the following command:

``` bash
    docker run -it fslam
```

This command starts the container and provides an interactive terminal within it.

## Features

- Utilizes the FastSLAM algorithm for simultaneous localization and mapping.
- Integrates with ROS Noetic and utilizes various ROS functionalities.
- Supports camera integration, including Realsense cameras.
- Provides a wrapper for ROS integration and additional functionality.

## Contributing

Contributions to the FSLAM project are welcome. If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make the necessary changes and commit them.
4. Push your changes to your forked repository.
5. Submit a pull request detailing the changes you have made.

## License

This project is licensed under the [MIT License](LICENSE).