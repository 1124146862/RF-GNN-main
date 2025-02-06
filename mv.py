import shutil
import os

# 定义源目录和目标目录
source_dir = '/home/gehongfei/RecInRec'
target_dir = '/Data/gehongfei'

# 构造目标文件夹的路径
target_dir_path = os.path.join(target_dir, os.path.basename(source_dir))

# 移动整个文件夹
try:
    shutil.move(source_dir, target_dir_path)
    print(f"文件夹 {source_dir} 已成功移动到 {target_dir_path}")
except Exception as e:
    print(f"无法移动文件夹 {source_dir}: {e}")
