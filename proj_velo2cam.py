import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

sn = int(sys.argv[1]) if len(sys.argv)>1 else 7 #default 0-7517
name = '%06d'%sn # 6 digit zeropadding
img = f'./data_object_image_2/testing/image_2/{name}.png'
binary = f'./data_object_velodyne/testing/velodyne/{name}.bin'
with open(f'./testing/calib/{name}.txt','r') as f:
    calib = f.readlines()

# camera2 相机内参 P2 (3 x 4) for left eye
# | fx 0 u0 |
# | 0 fy v0 |
# | 0  0  1 |
# example: 000007.txt
# 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 4.485728000000e+01 
# 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 2.163791000000e-01 
# 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 2.745884000000e-03
P2 = np.array([float(x) for x in calib[2].strip('\n').split(' ')[1:]]).reshape(3,4)

# 相机旋转矩阵
# | fx 0 u0 |
# | 0 fy v0 |
# | 0  0  1 |
# example: 000007.txt
# 9.999239000000e-01 9.837760000000e-03 -7.445048000000e-03 
# -9.869795000000e-03 9.999421000000e-01 -4.278459000000e-03 
# 7.402527000000e-03 4.351614000000e-03 9.999631000000e-01
R0_rect = np.array([float(x) for x in calib[4].strip('\n').split(' ')[1:]]).reshape(3,3)

# Add a 1 in bottom-right, reshape to 4 x 4
R0_rect = np.insert(R0_rect,3,values=[0,0,0],axis=0)
R0_rect = np.insert(R0_rect,3,values=[0,0,0,1],axis=1)

# 激光雷达到相机坐标变换矩阵
# | R  T |
# | 0  1 |
# example: 000007.txt
# 7.533745000000e-03 -9.999714000000e-01 -6.166020000000e-04 -4.069766000000e-03 
# 1.480249000000e-02 7.280733000000e-04 -9.998902000000e-01 -7.631618000000e-02 
# 9.998621000000e-01 7.523790000000e-03 1.480755000000e-02 -2.717806000000e-01
Tr_velo_to_cam = np.array([float(x) for x in calib[5].strip('\n').split(' ')[1:]]).reshape(3,4)
Tr_velo_to_cam = np.insert(Tr_velo_to_cam,3,values=[0,0,0,1],axis=0)

# point cloud
scan = np.fromfile(binary, dtype=np.float32).reshape((-1,4))
points = scan[:, 0:3] # lidar xyz (front, left, up)

# TODO: use fov filter? 
velo = np.insert(points,3,1,axis=1).T
velo = np.delete(velo,np.where(velo[0,:]<0),axis=1)
cam = P2.dot(R0_rect.dot(Tr_velo_to_cam.dot(velo)))
cam = np.delete(cam,np.where(cam[2,:]<0),axis=1)

# get u,v,z
cam[:2] /= cam[2,:]

# do projection staff
plt.figure(figsize=(12,5),dpi=96,tight_layout=True)
png = mpimg.imread(img)
IMG_H,IMG_W,_ = png.shape

# restrict canvas in range
plt.axis([0,IMG_W,IMG_H,0])
plt.imshow(png)

# filter point out of canvas
u,v,z = cam
u_out = np.logical_or(u<0, u>IMG_W)
v_out = np.logical_or(v<0, v>IMG_H)
outlier = np.logical_or(u_out, v_out)
cam = np.delete(cam,np.where(outlier),axis=1)

# generate color map from depth
u,v,z = cam
plt.scatter([u],[v],c=[z],cmap='rainbow_r',alpha=0.5,s=2)
plt.title(name)
plt.savefig(f'./data_object_image_2/testing/projection/{name}.png',bbox_inches='tight')
plt.show()
