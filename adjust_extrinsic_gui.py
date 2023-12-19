import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np
import proj_pcd2cam

class ExtrinsicAdjuster:
    def __init__(self, window, path, pt_path, calib_path):
        self.window = window
        self.window.title("Lidar2Image Extrinsic Mat Adjuster")

        # Load the image using OpenCV
        self.original_image = cv2.imread(path)
        self.image = self.original_image.copy()

        # Create sliders for R, G, B
        self.alpha_scale = ttk.Scale(window, from_=0, to=2, orient="horizontal", command=self.update_extrinsic)
        self.beta_scale = ttk.Scale(window, from_=0, to=2, orient="horizontal", command=self.update_extrinsic)
        self.gamma_scale = ttk.Scale(window, from_=0, to=2, orient="horizontal", command=self.update_extrinsic)
        self.x_scale = ttk.Scale(window, from_=0, to=2, orient="horizontal", command=self.update_extrinsic)
        self.y_scale = ttk.Scale(window, from_=0, to=2, orient="horizontal", command=self.update_extrinsic)
        self.z_scale = ttk.Scale(window, from_=0, to=2, orient="horizontal", command=self.update_extrinsic)

        # Set default value of sliders to 1 (no change)
        self.alpha_scale.set(1)
        self.beta_scale.set(1)
        self.gamma_scale.set(1)
        self.x_scale.set(1)
        self.y_scale.set(1)
        self.z_scale.set(1)

        # Place the sliders
        self.alpha_scale.pack()
        self.beta_scale.pack()
        self.gamma_scale.pack()
        self.x_scale.pack()
        self.y_scale.pack()
        self.z_scale.pack()

        # Create a label to display the image
        self.image_label = ttk.Label(window)
        self.image_label.pack()

        # Initialize the image on the label
        self.update_extrinsic()

        self.pointcloud = proj_pcd2cam.load_pcd_data(pt_path)
        self.intrisic, self.extrinsic= proj_pcd2cam.get_extrinsic(calib_path)

    def update_image(self, _=None):
        # Adjust the image color
        r, g, b = self.r_scale.get(), self.g_scale.get(), self.b_scale.get()
        adjusted_image = cv2.merge([self.image[:,:,0] * b, self.image[:,:,1] * g, self.image[:,:,2] * r])

        # Convert to PIL format and update the label
        pil_image = Image.fromarray(cv2.cvtColor(adjusted_image.astype('uint8'), cv2.COLOR_BGR2RGB))
        tk_image = ImageTk.PhotoImage(pil_image)
        self.image_label.configure(image=tk_image)
        self.image_label.image = tk_image

    def update_extrinsic(self, _=None):
        # Adjust the rotation parameters
        alpha, beta, gamma = self.alpha_scale.get(), self.beta_scale.get(), self.gamma_scale.get()
        x, y, z = self.alpha_scale.get(), self.beta_scale.get(), self.gamma_scale.get()
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha), np.cos(alpha)]
        ])

        # 绕 Y 轴旋转的旋转矩阵
        Ry = np.array([
            [np.cos(beta), 0, np.sin(beta)],
            [0, 1, 0],
            [-np.sin(beta), 0, np.cos(beta)]
        ])
        Rz = np.array([
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma),  np.cos(gamma), 0],
            [0,                     0,                      1]
        ])
        self.extrinsic = Rx*Ry*Rz*(self.extrinsic)

        u, v, z = get_pointcloud_on_image(self.extrinsic, self.pointcloud)
        
        adjusted_image = cv2.merge([self.image[:,:,0] * b, self.image[:,:,1] * g, self.image[:,:,2] * r])

        # Convert to PIL format and update the label
        pil_image = Image.fromarray(cv2.cvtColor(adjusted_image.astype('uint8'), cv2.COLOR_BGR2RGB))
        tk_image = ImageTk.PhotoImage(pil_image)
        self.image_label.configure(image=tk_image)
        self.image_label.image = tk_image

def get_color(z):
    # 确保 z 值在合理范围内
    z = max(0, min(z, 1))
    # 将 z 值映射到 0 到 255 范围的蓝色通道
    return (255 * z, 0, 0)

def draw_circle(image, u, v, z):
    color = get_color(z)
    cv2.circle(image, (u, v), radius=10, color=color, thickness=-1)

# Main function to create the GUI
def main():
    image_path = 'correspond_data/image/1702895061262535.jpg'
    pointcloud_path = 'correspond_data/pointcloud/1702895061247132.pcd'
    calib_data_path = 'self_data/avpslam/calibration/20231218_192345_autoware_lidar_camera_calibration.yaml'
    root = tk.Tk()
    app = ExtrinsicAdjuster(root, image_path, pointcloud_path, calib_data_path)  # Replace with your image path
    root.mainloop()

if __name__ == '__main__':
    main()