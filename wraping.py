import cv2
import numpy as np
import os
from tqdm import tqdm
import time

def compute_homography(image1_path, image2_path):
    # 读取图像
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # 初始化SIFT检测器
    sift = cv2.SIFT_create()

    # 检测并计算描述子
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # 使用BFMatcher进行描述子匹配
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # 按距离排序匹配结果
    matches = sorted(matches, key=lambda x: x.distance)

    # 提取匹配点坐标
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # 计算变换矩阵
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    
    return h

def transform_mask(mask_path, homography_matrix, reference_image_path):
    # 读取掩码和参考图像
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    reference_image = cv2.imread(reference_image_path)

    # 检查是否成功读取参考图像
    if reference_image is None:
        raise ValueError(f"Failed to read reference image from {reference_image_path}")
    
    # 检查是否成功读取掩码
    if mask is None:
        print(f"Failed to read mask from {mask_path}, using a black mask instead")
        mask = np.zeros((reference_image.shape[0], reference_image.shape[1]), dtype=np.uint8)
    else:
        # 调整掩码大小以匹配参考图像
        mask = cv2.resize(mask, (reference_image.shape[1], reference_image.shape[0]))

    # 获取参考图像尺寸
    h, w = reference_image.shape[:2]
    
    # 应用透视变换
    transformed_mask = cv2.warpPerspective(mask, homography_matrix, (w, h))
    
    return transformed_mask

def apply_threshold(mask, threshold=128):
    # 应用阈值，将灰度值高于阈值的部分设为255，其余部分设为0
    _, binary_mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    return binary_mask

def keep_largest_connected_component(mask):
    # 查找所有的连通组件
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # 如果没有连通组件，直接返回原始掩码
    if num_labels <= 1:
        return mask
    
    # 找到面积最大的连通组件，排除背景
    largest_component_index = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    
    # 创建新的掩码，只保留最大的连通组件
    largest_component_mask = np.zeros_like(mask)
    largest_component_mask[labels == largest_component_index] = 255
    
    return largest_component_mask

def calculate_intersection(masks):
    # 计算多个掩码的交集
    intersection = masks[0]
    for mask in masks[1:]:
        intersection = cv2.bitwise_and(intersection, mask)
    return intersection

def calculate_maximum_intersection(masks):
    """
    计算所有掩码中交集最多的部分。

    参数:
    masks (list of numpy.ndarray): 掩码列表。

    返回:
    numpy.ndarray: 最终的最大交集掩码。
    """
    # 将掩码堆叠在一起，逐像素累加
    combined_mask = np.sum(np.array(masks), axis=0)
    
    # 找到所有掩码中交集最多的值
    max_intersection_value = np.max(combined_mask)
    
    # 生成最大交集掩码
    max_intersection_mask = (combined_mask == max_intersection_value).astype(np.uint8) * 255
    
    return max_intersection_mask

def main():
    image_dir = '/home/honor410/Disk4T/thj/3DGS/gaussian-splatting/output/kendo/train/ours_30000/renders'
    mask_dir = '/home/honor410/Disk4T/thj/InSPyReNet/results_otsu/kendo'
    output_dir = '/home/honor410/Disk4T/thj/wrapping_result/kendo_for_otsu_n_8'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取所有图像文件，按文件名排序
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')], key=lambda x: int(x.split('.')[0]))
    # print(image_files)

    # 记录开始时间
    start_time = time.time()

    for i in tqdm(range(len(image_files) - 1), desc="Processing images"):
        homographies = []
        # for j in range(-3, 4):  # 扩展范围，从 -3 到 3
        # for j in range(-3, 2):  # 扩展范围，从 -3 到 1
        # for j in range(-3, 6):  # 扩展范围，从 -3 到 5
        for j in range(-3, 8):  # 扩展范围，从 -3 到 7
            if i + j < 0 or i + j >= len(image_files):
                continue  # 跳过越界的情况

            image1_path = os.path.join(image_dir, image_files[i+j])
            image2_path = os.path.join(image_dir, image_files[i])
            mask_path = os.path.join(mask_dir, image_files[i+j])

            # 计算变换矩阵
            homography_matrix = compute_homography(image1_path, image2_path)
            homographies.append((homography_matrix, mask_path))
            print(f"Computed Homography Matrix for {image_files[i]} and {image_files[i+j]}")

        # 对所有掩码进行透视变换
        transformed_masks = []
        for homography_matrix, mask_path in homographies:
            transformed_mask = transform_mask(mask_path, homography_matrix, image2_path)
            transformed_masks.append(transformed_mask)

        # 计算所有掩码的最大交集
        max_intersection_mask = calculate_maximum_intersection(transformed_masks)
        final_mask = max_intersection_mask

        # 应用阈值去除阴影
        final_mask = apply_threshold(final_mask, threshold=200)
        
        output_path = os.path.join(output_dir, f'{image_files[i].split(".")[0]}.jpg')
        cv2.imwrite(output_path, final_mask)
        if os.path.exists(output_path):
            print(f"Final combined mask saved as '{output_path}'")
        else:
            raise ValueError(f"Failed to write mask to: {output_path}")
        
     # 记录结束时间
    end_time = time.time()

    # 计算总时间并转换为时分秒格式
    total_time = end_time - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total processing time: {int(hours)}h {int(minutes)}m {int(seconds)}s")


if __name__ == "__main__":
    main()
