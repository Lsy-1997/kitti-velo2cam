# Lidar to camera projection of KITTI
## Compare to Original Code
add comprehensable annotation and add reflectance projection
## Intro
[中文博客](https://www.cnblogs.com/azureology/p/14004131.html)

This is a Python implementation of how to project point cloud from Velodyne coordinates into the left color image with KITTI data set.
## Dependices
```
matplotlib == 3.1.3
numpy == 1.18.1
```

Also tested with
```
matplotlib == 3.4.3
numpy == 1.23.5
```
## Usage
Download KITTI dataset and place `proj_velo2cam.py` in the root path.
```
python3 proj_velo2cam.py
```
## Quick demo
Just clone the whole repo and run `proj_velo2cam.py`.
By default, run with frame 000007 with path below:
```
.
├── data_object_image_2
│   └── testing
│       ├── image_2
│       │   └── 000007.png
│       └── projection
│           └── 000007.png
├── data_object_velodyne
│   └── testing
│       └── velodyne
│           └── 000007.bin
├── proj_velo2cam.py
├── readme.md
└── testing
    └── calib
        └── 000007.txt
```
## Projection Result
![](./data_object_image_2/testing/projection/000001.png)
