import cv2
import os

# 输入和输出文件夹路径
input_folder = '/home/honor410/Disk4T/thj/InSPyReNet/results/kendo'  # 输入灰度图像文件夹路径
output_folder = '/home/honor410/Disk4T/thj/InSPyReNet/results_otsu/kendo'  # 输出二值图像文件夹路径

# 如果输出文件夹不存在，则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.bmp'):
        # 构建完整的文件路径
        img_path = os.path.join(input_folder, filename)
        
        # 读取图像
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # 使用 Otsu 算法进行阈值处理
        _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 保存二值图像
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, binary_img)

print("所有图像已处理并保存。")
