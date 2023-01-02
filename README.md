# Udacity Sensor Fusion Nanodegree - Project 2: 3D Object Tracking

<img src="images/course_code_structure.png" width="779" height="414" />

This repository is a fork from [Udacity's repository for the project](https://github.com/udacity/SFND_3D_Object_Tracking) to be completed. 

## Tasks to carry out
In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks: 
1. Develop a way to match 3D objects over time by using keypoint correspondences. 
2. Compute the TTC based on Lidar measurements. 
3. Do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. Conduct various tests with the framework, to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. 


## Setting up the system
### Dependencies and required build tools on Windows Subsystem for Linux with Ubuntu 20.04 
#### [cmake](https://cmake.org/install/) 3.16.3
#### [build-essential](https://packages.ubuntu.com/focal/build-essential)
  * make 4.2.1 (Installed by default and upgraded.)
  * gcc/g++ 9.4.0 (Installed by default and upgraded.)
#### [Git LFS](https://git-lfs.github.com/) 3.3.0
In the [original repository](https://github.com/udacity/SFND_3D_Object_Tracking), it was recommended to prepare git for uploading large files before cloning the project repository. The instructions for installing Git Large File Storage extension can be found [here](https://git-lfs.com/). Install the extension and then clone the repository. 
```sh
sudo apt-get -y install git-lfs
```
#### [OpenCV](https://github.com/opencv/opencv) 4.7.0 and [Contrib Modules](https://github.com/opencv/opencv_contrib)
The following was used to build the latest version of OpenCV together with Contrib modules. At the time of executing, January 2nd, 2023, the latest version was 4.7.0. 
```sh
git clone https://github.com/opencv/opencv
git clone https://github.com/opencv/opencv_contrib
cd opencv
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=RELEASE -DOPENCV_ENABLE_NONFREE=ON -DENABLE_PRECOMPILED_HEADERS=OFF -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules ..
make -j`nproc`
sudo make install
```

### This repository

```sh
git clone git@github.com:SaidZahrai/SFND_3D_Object_Tracking.git
cd SFND_3D_Object_Tracking
mkdir build && cd build
cmake ..
make
./3D_Object_Tracking
```


## References

1. Instructions for setting up the system [Roman Smirnov](https://gist.github.com/roman-smirnov/efff8bb1db8a4063600a40c29a3a0874)

