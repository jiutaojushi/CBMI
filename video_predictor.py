import os
import numpy as np
import torch
import cv2
from sklearn.cluster import KMeans
from sam2.build_sam import build_sam2_video_predictor
import time

# 选择运行设备
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

# 加载预训练模型及其配置
sam2_checkpoint = "/home/honor410/Disk4T/thj/segment-anything-2/checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

# 初始化predictor和读取帧的name
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
video_dir = "/home/honor410/Disk4T/thj/3DGS/gaussian-splatting/output/camera/train/ours_30000/renders"
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# 获取图像的shape
h, w, _= cv2.imread(video_dir + "/" + frame_names[0]).shape
img_shape = (h, w)

# 初始化state，把帧load进去
inference_state = predictor.init_state(video_path=video_dir)

# 帧索引，单个追踪物体索引
ann_frame_idx = 0
ann_obj_id = 0


binary_image_path = "/home/honor410/Disk4T/thj/segment-anything-2/output/3dgs_solo/camera_for_user_light.png"
# 读取二值mask
mask = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)

if mask is None or len(np.unique(mask)) > 2:
    raise ValueError("图像读取失败或不是二值图像")

# 记录传播开始时间
start_time = time.time()

# 获取白色区域的坐标
white_coords = np.argwhere(mask == 255)

# 设置 K-means 聚类的参数
n_clusters = 30  # 你可以根据需要设置聚类的数量

# 执行 K-means 聚类
kmeans = KMeans(n_clusters=n_clusters, n_init=n_clusters, random_state=0).fit(white_coords)

# 获取聚类结果
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# 将聚类中心坐标四舍五入为整数
centroids_int = np.round(centroids).astype(int)

# [y, x] -> [x, y]
points = np.array([[y, x] for x, y in centroids_int], dtype=np.float32)
labels = np.array([1 for _ in range(len(centroids_int))], np.int32)

# # 假设 mask 是你的原图像
# height, width = mask.shape[:2]

# # 创建一张与原图一样的副本
# new_image = mask.copy()

# # 把选取点标记出来
# for point in points:
#     print(point)
#     point = [int(i) for i in point]
#     cv2.circle(new_image, point, radius=1, color=(0, 0, 0), thickness=-1)  # 在新图上标记，黑色圆点

# # 保存标记后的图像
# cv2.imwrite('/home/wrf/Disk2/thj/0_sample.png', new_image)

predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# 对第一帧增加mask，以优化其表现
predictor.add_new_mask(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    mask=mask,
)

# 把mask和prompts传播到后续所有帧，进行追踪
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# 记录传播结束时间
end_time = time.time()
elapsed_time = end_time - start_time

# 转换为时分秒格式
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Propagation completed in {int(hours):02}:{int(minutes):02}:{seconds:.2f}")


# 保存每一帧的跟踪结果
output_dir = "/home/honor410/Disk4T/thj/segment-anything-2/output/camera_for_user_light" 
os.makedirs(output_dir, exist_ok=True)

for out_frame_idx, segment in video_segments.items():
    frame_name = frame_names[out_frame_idx].split(".")[0]  # 根据帧索引获取文件名
     # 初始化一个空的掩码
    combined_mask = np.zeros(img_shape, dtype=np.uint8)

    for out_obj_id, mask in segment.items():
        # 去掉多余的维度并将布尔掩码转换为 uint8 图像（0 或 255）
        mask_image = (mask.squeeze() * 255).astype(np.uint8)
        
        # 合并到总掩码中
        combined_mask = cv2.bitwise_or(combined_mask, mask_image)
    mask_filename = os.path.join(output_dir, f"{frame_name}.png")
    cv2.imwrite(mask_filename, combined_mask)
