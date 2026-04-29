import os

image_folder = "dataset/images/train"
label_folder = "dataset/labels/train"

prefix = "old_"   # change if needed

images = sorted(os.listdir(image_folder))

for i, img_name in enumerate(images):
    ext = img_name.split('.')[-1]
    
    new_name = f"{prefix}{i:04d}.{ext}"
    
    # Rename image
    os.rename(
        os.path.join(image_folder, img_name),
        os.path.join(image_folder, new_name)
    )
    
    # Rename label
    label_name = img_name.rsplit('.', 1)[0] + ".txt"
    new_label = new_name.rsplit('.', 1)[0] + ".txt"
    
    if os.path.exists(os.path.join(label_folder, label_name)):
        os.rename(
            os.path.join(label_folder, label_name),
            os.path.join(label_folder, new_label)
        )

print("✅ Renaming completed!")