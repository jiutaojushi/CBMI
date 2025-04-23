from PIL import Image
import torch
import numpy as np
import os
import cv2
from sklearn.cluster import KMeans
import time  # 用于记录时间

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

image = Image.open('/home/honor410/Disk4T/thj/3DGS/gaussian-splatting/output/camera/train/ours_30000/renders/00000.jpg')
image = np.array(image.convert("RGB"))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_checkpoint = "/home/honor410/Disk4T/thj/segment-anything-2/checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)

predictor.set_image(image)

binary_image_path = "/home/honor410/Disk4T/thj/wrapping_result/kendo_for_otsu_n_8/00000.jpg"
# 读取二值mask
mask = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)
_, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

if mask is None or len(np.unique(mask)) > 2:
    raise ValueError("图像读取失败或不是二值图像")


# # 记录开始时间
# start_time = time.time()

# 获取白色区域的坐标
white_coords = np.argwhere(mask == 255)

# 设置 K-means 聚类的参数
n_clusters = 15  # 你可以根据需要设置聚类的数量


# 执行 K-means 聚类
kmeans = KMeans(n_clusters=n_clusters, n_init=n_clusters, random_state=0).fit(white_coords)


# 获取聚类结果
centroids = kmeans.cluster_centers_
labels = kmeans.labels_


# 将聚类中心坐标四舍五入为整数
centroids_int = np.round(centroids).astype(int)


# # [y, x] -> [x, y]
# input_point = np.array([[y, x] for x, y in centroids_int])
# input_label = np.array([1 for _ in range(len(centroids))])

# # User Light
# input_point = [[596, 566], [378, 454], [182, 650], [428, 680], [412, 588]] 
# input_label = [1, 1, 1, 1, 1]

# User Film
input_point = [[1048, 774], [1066, 868], [1228, 870], [1160, 742], [1202, 712]]
input_label = [1, 1, 1, 1, 1]

# # 假设 mask 是你的原图像
# height, width = mask.shape[:2]

# # 创建一张与原图一样的副本
# new_image = mask.copy()

# # 把选取点标记出来
# for point in input_point:
#     print(point)
#     cv2.circle(new_image, point, radius=1, color=(0, 0, 0), thickness=-1)  # 在新图上标记，黑色圆点

# # 保存标记后的图像
# cv2.imwrite('/home/wrf/Disk2/thj/0_sample.png', new_image)

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

# 再分割
mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    mask_input=mask_input[None, :, :],
    multimask_output=False,
)

# # 再再分割
# mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
# masks, scores, _ = predictor.predict(
#     point_coords=input_point,
#     point_labels=input_label,
#     mask_input=mask_input[None, :, :],
#     multimask_output=False,
# )

output_dir = f"/home/honor410/Disk4T/thj/segment-anything-2/output/3dgs_solo"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
for i in range(masks.shape[0]):
    mask = masks[i] * 255
    mask = mask.astype(np.uint8)  # 确保数据类型为 uint8
    output_path = os.path.join(output_dir, f'camera_for_user_film.png')
    cv2.imwrite(output_path, mask)
    print(f'Saved mask {i+1} to {output_path}')
print("OK!")

# # 记录结束时间
# end_time = time.time()
# elapsed_time = end_time - start_time

# # 转换为时分秒格式
# hours, rem = divmod(elapsed_time, 3600)
# minutes, seconds = divmod(rem, 60)
# print(f"Process completed in {int(hours):02}:{int(minutes):02}:{seconds:.2f}")
