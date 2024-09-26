import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


def generate_gcamp6_signal(time, peak_times, decay_constant=0.1, noise_level=0.05):
    """
    生成模拟GCaMP6钙离子信号。
    :param time: 时间数组。
    :param peak_times: 活动峰值时间点数组。
    :param decay_constant: 衰减常数。
    :param noise_level: 噪声水平。
    :return: 模拟的钙离子信号。
    """
    signal = np.zeros_like(time)
    for peak in peak_times:
        peak_index = np.argmin(np.abs(time - peak))
        signal[peak_index:] += np.exp(-decay_constant * (time[peak_index:] - peak))
    signal += np.random.normal(0, noise_level, signal.shape)
    return signal

# 更新模拟参数
num_neurons = 3
num_frames = 1000       # 时间点数量
signal_frequency = 6   # 信号频率
duration = num_frames / signal_frequency
time = np.linspace(0, duration, num_frames)
#
# np.random.seed(0)
# signals = []
# for i in range(3):
#     # 完全随机生成活动峰值的时间和强度
#     num_peaks = np.random.randint(5, 10)
#     peak_times = np.sort(np.random.uniform(0, duration, num_peaks))
#     peak_amplitudes = np.random.uniform(0.5, 1.5, num_peaks)  # 随机峰值强度
#
#     # 生成信号
#     signal = np.zeros_like(time)
#     for peak_time, amplitude in zip(peak_times, peak_amplitudes):
#         peak_index = np.argmin(np.abs(time - peak_time))
#         signal[peak_index:] += amplitude * np.exp(-0.2 * (time[peak_index:] - peak_time))
#     signal += np.random.normal(0, 0.05, signal.shape)  # 添加噪声
#     signals.append(signal)
#

signals = sio.loadmat('signals.mat')['signals'][:3000, :]
## 绘制这些信号
# plt.figure(figsize=(15, 5))
# for i, signal in enumerate(signals):
#     plt.plot(time, signal, label=f'神经元 {i+1}')
# plt.xlabel('时间 (秒)')
# plt.ylabel('活动强度')
# plt.title('发放时间和峰值强度随机的三个神经元的模拟GCaMP6钙离子信号')
# plt.legend()
# plt.show()


# 更新神经元区域为类椭圆形，面积大约在10*10
def create_ellipse_mask(center, axes, shape):
    """
    创建一个椭圆形的掩码。
    :param center: 椭圆中心点 (x, y)。
    :param axes: 椭圆的半轴长度 (a, b)。
    :param shape: 掩码的形状。
    :return: 椭圆形掩码。
    """
    x, y = np.ogrid[:shape[0], :shape[1]]
    h, k = center
    a, b = axes
    mask = ((x - h)**2 / a**2) + ((y - k)**2 / b**2) <= 1
    return mask


def calculate_iou(mask1, mask2):
    """
    计算两个掩码之间的IOU。
    :param mask1: 第一个掩码。
    :param mask2: 第二个掩码。
    :return: IOU值。
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    else:
        return intersection / union

# 定义三个神经元的不同形状和位置
neuron_shapes = [(6, 4), (5, 5), (4, 6)]
neuron_positions = [(5, 5), (8, 8), (11, 11)]

# 确保至少两个神经元的IOU超过0.3
masks = [create_ellipse_mask(neuron_positions[i], neuron_shapes[i], (20, 20)) for i in range(3)]
ious = [[calculate_iou(masks[i], masks[j]) for j in range(3)] for i in range(3)]

# 更新视频数据
video = np.zeros((500, 20, 20))
for i, signal in enumerate(signals):
    mask = masks[i]
    for t in range(500):
        video[t][mask] += signal[t]

# 规范化视频数据到合理的范围
video = np.clip(video, 0, 255)

# 转换颜色值为适合matplotlib处理的格式（归一化到[0, 1]范围）
colors_normalized = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # 红色、绿色、蓝色

# 绘制带有不同颜色边界的帧
frame_with_colored_boundaries_adjusted = np.zeros((20, 20, 3))  # 创建三通道图像
for i in range(3):
    mask = masks[i]
    for c in range(3):  # 应用颜色到相应通道
        frame_with_colored_boundaries_adjusted[:, :, c] += mask * colors_normalized[i][c]
frame_with_colored_boundaries_adjusted = np.clip(frame_with_colored_boundaries_adjusted * 255, 0, 255).astype(np.uint8)

# 显示结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(frame_with_colored_boundaries_adjusted)
plt.colorbar(label='像素强度')
plt.title('不同颜色表示不同神经元的模拟视频帧')

# 显示每个神经元的信号
plt.subplot(1, 2, 2)
for i, signal in enumerate(signals):
    plt.plot(signal, label=f'神经元 {i+1}', color=colors_normalized[i])
plt.xlabel('时间 (帧)')
plt.ylabel('活动强度')
plt.title('三个神经元的活动信号')
plt.legend()

plt.tight_layout()
plt.show()