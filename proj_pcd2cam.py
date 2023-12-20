import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import glob
from tqdm import tqdm
import yaml

def load_pcd_data(file_path):
    pts = []
    with open(file_path,'r') as f:
        data = f.readlines()
    line = data[9]
    line = line.strip('\n')
    i = line.split(' ')
    pts_num = eval(i[-1])
    for line in data[11:]:
        line = line.strip('\n')
        xyzi = line.split(' ')
        # print(line)
        if xyzi[0] != 'nan' and xyzi[1] != 'nan' and xyzi[2] != 'nan' and xyzi[3] != 'nan':
            x, y, z = [eval(i) for i in xyzi[0:3]]
            intensity = str(int(xyzi[-1])/255.0)
            pts.append([x, y, z, intensity])

    print(f"origin points: {pts_num}")
    print(f"valid points: {len(pts)}")
    print(f"invalid points: {pts_num - len(pts)}")
    res = np.zeros((len(pts), len(pts[0])), dtype=np.float32)
    for i in range(len(pts)):
        res[i] = pts[i]
    return res

def get_calib_param(cam_lidar_calib_file):
    with open(cam_lidar_calib_file, 'r') as file:
        cam_lidar_calib_data = yaml.safe_load(file)
    camera_mat = cam_lidar_calib_data.get('CameraMat')
    intrinsic = np.array(camera_mat.get('data')).reshape(3,3)
    intrinsic = np.insert(intrinsic,3,values=[0,0,0],axis=1)

    extrinsic = cam_lidar_calib_data.get('CameraExtrinsicMat')
    extrinsic = np.array(extrinsic.get('data')).reshape(4,4)
    return intrinsic, extrinsic

def get_pointcloud_on_image(intrinsic, extrinsic, pointcloud):
    # Autoware标定矩阵变换，跟普通变换矩阵不同, 若为普通变换矩阵，请注释下面这段代码
    extrinsic[:3,:3] = extrinsic[:3,:3].T
    x = extrinsic[0, 3]
    y = extrinsic[1, 3]
    z = extrinsic[2, 3]
    extrinsic[0, 3] = y
    extrinsic[1, 3] = z
    extrinsic[2, 3] = -x

    # lidar xyz (front, left, up)
    points = pointcloud[:, 0:3]

    # reflectance
    reflectance = pointcloud[:, 3].T

    # 补充一个维度，便于矩阵计算
    lidar = np.insert(points,3,1,axis=1).T

    # 删除距离为负的点云
    # reflectance = np.delete(reflectance, np.where(velo[0,:]<0))
    # velo = np.delete(velo,np.where(velo[0,:]<0),axis=1)

    cam = intrinsic.dot(extrinsic.dot(lidar))

    # 删除像方坐标z为负值的点
    reflectance = np.delete(reflectance, np.where(cam[2,:]<0))
    cam = np.delete(cam,np.where(cam[2,:]<0),axis=1)

    # get u,v,z
    cam[:2] /= cam[2,:]

    return cam, reflectance

def plt_init(img_file):
    fig, axes = plt.subplots(1, 3)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    img = mpimg.imread(img_file)
    IMG_H,IMG_W,_ = img.shape

    axes[0].imshow(img)
    axes[0].set_title('Image', fontsize=6)

    axes[1].imshow(img)
    axes[1].set_title('Depth Mix', fontsize=6)

    axes[2].imshow(img)
    axes[2].set_title('Reflectance Mix', fontsize=6)

    plt.tight_layout()
    axes[0].axis('off')
    axes[1].axis('off')
    axes[2].axis('off')
    return IMG_H, IMG_W, axes

def process_one_frame(number):
    # 读取标定得到的内外参
    cam_lidar_calib_file='ros_data/20231218_132035_autoware_lidar_camera_calibration.yaml'
    intrinsic, extrinsic = get_calib_param(cam_lidar_calib_file)

    # 读取激光点云数据
    point_cloud_file2 = 'ros_data/pointcloud/1702895061247132.pcd'
    scan = load_pcd_data(point_cloud_file2)

    cam, reflectance = get_pointcloud_on_image(intrinsic, extrinsic, scan)

    # plt init
    img_file = 'ros_data/image/1702895061262535.jpg'
    img_name = os.path.splitext(os.path.basename(img_file))[0]
    IMG_H, IMG_W, axes = plt_init(img_file)

    # filter point out of canvas 删除相机取景框以外的点云
    u,v,z = cam
    u_out = np.logical_or(u<0, u>IMG_W)
    v_out = np.logical_or(v<0, v>IMG_H)
    outlier = np.logical_or(u_out, v_out)

    reflectance = np.delete(reflectance,np.where(outlier))
    cam = np.delete(cam,np.where(outlier),axis=1)

    # 根据 u, v 将点云画到图像上 (s可调整点云像素大小)
    u,v,z = cam
    axes[1].scatter([u],[v],c=[z],cmap='rainbow_r',alpha=0.5,s=5)
    axes[2].scatter([u],[v],c=[reflectance],cmap='rainbow_r',alpha=0.5,s=5)

    projection_save_dir = 'ros_data/projection/'
    os.makedirs(projection_save_dir, exist_ok=True)
    # plt.savefig(f'./data_object_image_2/testing/projection/{number}.png',dpi=300,bbox_inches='tight')
    plt.savefig(os.path.join(projection_save_dir, img_name),dpi=300,bbox_inches='tight')

    # plt.show()

def main():
    process_one_frame('000003')
    print("finished!")

if __name__ == '__main__':
    main()