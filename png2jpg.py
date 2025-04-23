from PIL import Image
import os

def convert_png_to_jpg(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            img = Image.open(os.path.join(folder_path, filename))
            new_filename = os.path.splitext(filename)[0] + ".jpg"
            img.save(os.path.join(folder_path, new_filename), "JPEG")
            os.remove(os.path.join(folder_path, filename))  # 删除原png文件

# folder_path = "/home/honor410/Disk4T/thj/3DGS/gaussian-splatting/output/kendo/train/ours_30000/renders"  # 请替换为实际的文件夹路径
folder_path = "/home/honor410/Disk4T/thj/3DGS/gaussian-splatting/output/camera/train/ours_30000/renders"
convert_png_to_jpg(folder_path)
