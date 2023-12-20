import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np
import yaml
import datetime
import proj_pcd2cam
import glob
import os
from tqdm import tqdm

class ExtrinsicAdjuster:
    def __init__(self, window, img_dir, pointcloud_dir, calib_path):
        self.window = window
        self.window.title("Lidar2Image Extrinsic Mat Adjuster")

        self.img_ext = '.jpg'
        self.pt_ext = '.pcd'
        self.images_file = []
        self.img_dir = img_dir
        self.pointcloud_dir = pointcloud_dir
        for file_path in glob.glob(os.path.join(img_dir, f'*{self.img_ext}')):
            self.images_file.append(os.path.splitext(os.path.basename(file_path))[0])

        self.pointcloud_file = []
        for file_path in glob.glob(os.path.join(pointcloud_dir, f'*{self.pt_ext}')):
            self.pointcloud_file.append(os.path.splitext(os.path.basename(file_path))[0])

        self.images_file.sort(key=lambda x: int(x))
        self.pointcloud_file.sort(key=lambda x: int(x))

        self.file_num = 0
        image_path = os.path.join(img_dir, self.images_file[self.file_num] + self.img_ext)
        pointcloud_path = os.path.join(pointcloud_dir, self.pointcloud_file[self.file_num] + self.pt_ext)

        # Load the image using OpenCV
        self.original_image = cv2.imread(image_path)
        self.image = self.original_image.copy()

        def create_slider(label_text, from_, to_, command):
            frame = ttk.Frame(window)
            label = ttk.Label(frame, text=label_text)
            label.pack(side=tk.LEFT)
            scale = ttk.Scale(frame, from_=from_, to=to_, orient="horizontal", command=command)
            scale.set(0)
            scale.pack(side=tk.LEFT)
            frame.pack()
            return scale

        # Create sliders with labels for Alpha, Beta, Gamma, X, Y, Z
        self.alpha_scale = create_slider("Alpha", -1, 1, self.update_extrinsic)
        self.beta_scale = create_slider("Beta", -1, 1, self.update_extrinsic)
        self.gamma_scale = create_slider("Gamma", -1, 1, self.update_extrinsic)
        self.x_scale = create_slider("X", -1, 1, self.update_extrinsic)
        self.y_scale = create_slider("Y", -1, 1, self.update_extrinsic)
        self.z_scale = create_slider("Z", -1, 1, self.update_extrinsic)
        self.progress_bar = create_slider("Progress Bar", 0, len(self.images_file)-1, self.move_progress_bar)

        # Set default value of sliders to 1 (no change)
        self.alpha_scale.set(0)
        self.beta_scale.set(0)
        self.gamma_scale.set(0)
        self.x_scale.set(0)
        self.y_scale.set(0)
        self.z_scale.set(0)
        self.progress_bar.set(0)

        # Place the sliders
        self.alpha_scale.pack()
        self.beta_scale.pack()
        self.gamma_scale.pack()
        self.x_scale.pack()
        self.y_scale.pack()
        self.z_scale.pack()
        self.progress_bar.pack()

        # Create a label to display the image
        self.image_label = ttk.Label(window)
        self.image_label.pack()

        # Create a save button
        self.save_button = ttk.Button(window, text="Save Extrinsic", command=self.save_extrinsic)
        self.save_button.pack()
        self.save_button = ttk.Button(window, text="Refresh Parameter", command=self.refresh_extrinsic)
        self.save_button.pack()
        self.save_button = ttk.Button(window, text="Next Image", command=self.next_img)
        self.save_button.pack()
        self.save_button = ttk.Button(window, text="Prev Image", command=self.prev_img)
        self.save_button.pack()

        # Initialize the image on the label
        self.intrisic, self.extrinsic= proj_pcd2cam.get_calib_param(calib_path)
        self.new_extrinsic = self.extrinsic.copy()
        self.pointcloud = proj_pcd2cam.load_pcd_data(pointcloud_path)
        self.update_extrinsic()
    
    def next_img(self):
        if(self.file_num+1<len(self.images_file)):
            self.file_num += 1
        self.refresh_image()
        self.progress_bar.set(self.file_num)

    def prev_img(self):
        if(self.file_num-1>=0):
            self.file_num -= 1
        self.refresh_image()
        self.progress_bar.set(self.file_num)

    def move_progress_bar(self, event=None):
        self.file_num = int(self.progress_bar.get())
        self.refresh_image()

    def refresh_image(self):
        image_path = os.path.join(self.img_dir, self.images_file[self.file_num] + self.img_ext)
        pointcloud_path = os.path.join(self.pointcloud_dir, self.pointcloud_file[self.file_num] + self.pt_ext)
        self.pointcloud = proj_pcd2cam.load_pcd_data(pointcloud_path)
        self.original_image = cv2.imread(image_path)
        self.image = self.original_image.copy()
        self.update_extrinsic()

    def refresh_extrinsic(self):
        self.alpha_scale.set(0)
        self.beta_scale.set(0)
        self.gamma_scale.set(0)
        self.x_scale.set(0)
        self.y_scale.set(0)
        self.z_scale.set(0)
        self.update_extrinsic()
    
    def save_extrinsic(self):
        data_to_save = {
            "CameraMat": {
                "rows": 3,
                "cols": 3,
                "dt": "d",  # 数据类型, 这里假设是 'd' (double)
                "data": self.new_extrinsic.tolist()  # 将 NumPy 数组转换为列表
            }
        }
        current_time = datetime.datetime.now()
        # 将数组写入 YAML 文件
        with open(f'{current_time}extrinsic.yaml', 'w') as file:
            yaml.dump(data_to_save, file)
        print("Extrinsic save successfully!")

    def update_extrinsic(self, _=None):
        # Adjust the rotation parameters
        alpha, beta, gamma = self.alpha_scale.get(), self.beta_scale.get(), self.gamma_scale.get()
        x, y, z = self.x_scale.get(), self.y_scale.get(), self.z_scale.get()
        
        alpha = (alpha / 3.14) * (1 / 360.0)
        beta = (beta / 3.14) * (1 / 360.0)
        gamma = (gamma / 3.14) * (1 / 360.0)
        # 3个旋转矩阵
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha), np.cos(alpha)]
        ])
        Ry = np.array([
            [np.cos(beta), 0, np.sin(beta)],
            [0, 1, 0],
            [-np.sin(beta), 0, np.cos(beta)]
        ])
        Rz = np.array([
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma),  np.cos(gamma), 0],
            [0, 0, 1]
        ])
        rotation_mat = self.extrinsic[:3,:3]
        rotation_mat = Rx.dot(Ry.dot(Rz.dot(rotation_mat)))
        self.new_extrinsic[:3, :3] = rotation_mat
        self.new_extrinsic[0, 3] = self.extrinsic[0, 3] + x
        self.new_extrinsic[1, 3] = self.extrinsic[1, 3] + y
        self.new_extrinsic[2, 3] = self.extrinsic[2, 3] + z

        point_on_img, reflectance = proj_pcd2cam.get_pointcloud_on_image(self.intrisic, self.new_extrinsic, self.pointcloud)
        
        u, v, z = point_on_img
        
        # adjusted_image = cv2.merge([self.image[:,:,0] * b, self.image[:,:,1] * g, self.image[:,:,2] * r])

        adjusted_image = draw_circle(self.image, u, v, z)
        # Convert to PIL format and update the label
        pil_image = Image.fromarray(cv2.cvtColor(adjusted_image.astype('uint8'), cv2.COLOR_BGR2RGB))
        resize_pil_image = pil_image.resize((pil_image.size[0]//2, pil_image.size[1]//2))
        tk_image = ImageTk.PhotoImage(resize_pil_image)
        self.image_label.configure(image=tk_image)
        self.image_label.image = tk_image

def draw_circle(image, u, v, z):
    adjusted_image = image.copy()
    max_in_z = np.max(z)
    z = z/max_in_z
    for i in range(len(u)):
        color = (255 * z[i], 0, 0)
        cv2.circle(adjusted_image, (int(u[i]), int(v[i])), radius=10, color=color, thickness=-1)
    return adjusted_image

# Main function to create the GUI
def main():

    # img_dir = 'ros_data/image'
    # pointcloud_dir = 'ros_data/pointcloud/'

    img_dir = 'correspond_data/image'
    pointcloud_dir = 'correspond_data/pointcloud'
    calib_data_path = 'ros_data/20231218_132035_autoware_lidar_camera_calibration.yaml'
    
    root = tk.Tk()
    
    app = ExtrinsicAdjuster(root, img_dir, pointcloud_dir, calib_data_path)
    root.mainloop()

if __name__ == '__main__':
    main()