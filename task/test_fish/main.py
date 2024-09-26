import os
import tifffile as tiff
import numpy as np

# 读取文件夹中的所有tif文件
folder_path = '/home/disks/sde/wwh/wuwh/Dataset/Fish_for_Weihang/5_min_bessel_low_power/Bliq_VMS/C2'
tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]

# 初始化一个列表来存储所有的图片数据
images = []

# 逐个读取tif文件并存储到列表中
for tif_file in tif_files:
    image = tiff.imread(os.path.join(folder_path, tif_file))
    images.append(image)

# 将列表转换为一个numpy数组
images_array = np.array(images)

# 叠加每6帧成1帧
new_frames = []
for i in range(0, len(images_array), 4):
    combined_frame = np.sum(images_array[i:i+4], axis=0)  # 叠加操作
    new_frames.append(combined_frame)

# 将新帧转换为numpy数组
new_frames_array = np.array(new_frames)

# 保存为新的tif文件
output_file = os.path.join(folder_path, 'combined_video.tif')
tiff.imwrite(output_file, new_frames_array.squeeze().astype(np.uint16))

# 计算最大值、最小值、均值和方差
max_value = np.max(new_frames_array)
min_value = np.min(new_frames_array)
mean_value = np.mean(new_frames_array)
std_value = np.std(new_frames_array)

print(f'最大灰度值: {max_value}')
print(f'最小灰度值: {min_value}')
print(f'均值: {mean_value}')
print(f'方差: {std_value}')
