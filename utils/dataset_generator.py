import cv2
import numpy as np
import os
import glob

def generate_dataset(image_dir='raw_images', save_dir='dataset', patch_size=224):
    os.makedirs(os.path.join(save_dir, 'low'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'medium'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'high'), exist_ok=True)
    
    if not os.path.exists(image_dir):
        os.makedirs(image_dir, exist_ok=True)
        print(f"Place original satellite images in '{image_dir}' directory and run again to generate dataset.")
        return

    image_paths = glob.glob(os.path.join(image_dir, '*.png')) + glob.glob(os.path.join(image_dir, '*.jpg'))
    count = 0
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None: continue
        h, w, _ = img.shape
        for y in range(0, h - patch_size + 1, patch_size):
            for x in range(0, w - patch_size + 1, patch_size):
                patch = img[y:y+patch_size, x:x+patch_size]
                
                patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                r = patch_rgb[:,:,0].astype(np.float32)
                g = patch_rgb[:,:,1].astype(np.float32)
                
                veg_index = (g - r) / (g + r + 1e-5)
                mean_vi = np.mean(veg_index)
                
                if mean_vi > 0.2:
                    label = 'low'
                elif 0.05 <= mean_vi <= 0.2:
                    label = 'medium'
                else:
                    label = 'high'
                
                save_path = os.path.join(save_dir, label, f'patch_{count}.png')
                cv2.imwrite(save_path, patch)
                count += 1
    print(f"Dataset generated. {count} patches created in '{save_dir}'.")

if __name__ == '__main__':
    generate_dataset()
