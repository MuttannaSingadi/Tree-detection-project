import os
import shutil
import random

image_dir = "dataset/images"
mask_dir = "dataset/masks"

train_img = "dataset/train/images"
train_mask = "dataset/train/masks"
test_img = "dataset/test/images"
test_mask = "dataset/test/masks"
val_img = "dataset/val/images"
val_mask = "dataset/val/masks"

folders = [train_img, train_mask, test_img, test_mask, val_img, val_mask]
for folder in folders:
    os.makedirs(folder, exist_ok=True)

files = os.listdir(image_dir)
files = [f for f in files if f.endswith((".jpg", ".png"))]

random.shuffle(files)

train_split = int(0.7 * len(files))
test_split = int(0.9 * len(files))

train_files = files[:train_split]
test_files = files[train_split:test_split]
val_files = files[test_split:]

def copy_files(file_list, src_img, src_mask, dst_img, dst_mask):
    for file in file_list:
        shutil.copy(os.path.join(src_img, file), os.path.join(dst_img, file))
        shutil.copy(os.path.join(src_mask, file), os.path.join(dst_mask, file))

copy_files(train_files, image_dir, mask_dir, train_img, train_mask)
copy_files(test_files, image_dir, mask_dir, test_img, test_mask)
copy_files(val_files, image_dir, mask_dir, val_img, val_mask)

print("Data split completed!")