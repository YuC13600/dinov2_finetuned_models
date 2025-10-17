#!/usr/bin/env python3
"""
Extract DINOv2 ViT-B/14 features from coral images in 2022sample and 2023sample directories
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import h5py
from tqdm import tqdm
import cv2

def load_dinov2_model():
    """Load DINOv2 ViT-B/14 model"""
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model.eval()
    return model

def get_image_transform():
    """Get image preprocessing transform for DINOv2"""
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform

def read_bbox_from_exif(img_path):
    """Read bounding box from EXIF data (adapted from train_efficientNet.py)"""
    try:
        import piexif
        exif_dict = piexif.load(img_path)

        if piexif.ImageIFD.XPComment in exif_dict["0th"]:
            comment_bytes = exif_dict["0th"][piexif.ImageIFD.XPComment]

            if isinstance(comment_bytes, bytes):
                try:
                    comment = comment_bytes.decode('utf-16le').rstrip('\x00')
                except UnicodeDecodeError:
                    comment = str(comment_bytes)
            elif isinstance(comment_bytes, tuple) or isinstance(comment_bytes, list):
                comment = "".join([chr(b) for b in comment_bytes if b != 0])
            else:
                comment = str(comment_bytes)

            if comment.startswith("bbox(") and comment.endswith(")"):
                bbox_str = comment[5:-1]
                try:
                    bbox = tuple(map(int, bbox_str.split(',')))
                    if len(bbox) == 4:
                        return bbox
                except ValueError:
                    pass
            else:
                import re
                bbox_pattern = r"bbox\((\d+),(\d+),(\d+),(\d+)\)"
                match = re.search(bbox_pattern, comment)
                if match:
                    bbox = tuple(map(int, match.groups()))
                    return bbox
    except Exception:
        pass
    return None

def crop_image_with_bbox(img, bbox):
    """Crop image using bounding box"""
    if bbox is None:
        return img
    
    x1, y1, x2, y2 = bbox
    width, height = img.size
    
    # Ensure bbox is within image bounds
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(x1, min(x2, width))
    y2 = max(y1, min(y2, height))
    
    if x2 > x1 and y2 > y1:
        return img.crop((x1, y1, x2, y2))
    return img

def extract_features(model, img_path, transform, device):
    """Extract DINOv2 features from a single image"""
    try:
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Try to read bbox from EXIF and crop if available
        bbox = read_bbox_from_exif(img_path)
        if bbox is not None:
            img = crop_image_with_bbox(img, bbox)
        
        # Apply transforms
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Extract features
        with torch.no_grad():
            features = model(img_tensor)
        
        return features.cpu().numpy().flatten()
    
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def collect_images_from_area(base_dir, area_id):
    """Collect all JPG images from a specific area directory"""
    images = []
    coral_names = []
    
    area_dir = os.path.join(base_dir, area_id)
    if os.path.isdir(area_dir):
        for img_file in os.listdir(area_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg')):
                img_path = os.path.join(area_dir, img_file)
                images.append(img_path)
                # Extract coral name from filename (e.g., "37.1.JPG" -> "37.1")
                coral_name = os.path.splitext(img_file)[0]
                coral_names.append(coral_name)
    
    return images, coral_names

def get_area_directories(base_dir):
    """Get all area directories (e.g., 37, 38, 39, 40)"""
    areas = []
    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path) and item.isdigit():
                areas.append(item)
    return sorted(areas)

def extract_features_from_area(base_dir, area_id, year, model, transform, device):
    """Extract DINOv2 features from all images in a specific area"""
    print(f"Processing area {area_id} from {year}")
    
    # Collect images from this area
    image_paths, coral_names = collect_images_from_area(base_dir, area_id)
    print(f"Found {len(image_paths)} images in area {area_id}")
    
    if len(image_paths) == 0:
        print(f"No images found in area {area_id}!")
        return None
    
    # Extract features
    features_list = []
    valid_coral_names = []
    
    for img_path, coral_name in tqdm(zip(image_paths, coral_names), total=len(image_paths), desc=f"Area {area_id}"):
        features = extract_features(model, img_path, transform, device)
        if features is not None:
            features_list.append(features)
            valid_coral_names.append(coral_name)
    
    if len(features_list) == 0:
        print(f"No valid features extracted from area {area_id}!")
        return None
    
    # Convert to numpy array
    features_array = np.array(features_list)
    print(f"Area {area_id} extracted features shape: {features_array.shape}")
    
    # Generate output filename
    output_file = f"dinov2_{year}_{area_id}_features.h5"
    
    # Save to HDF5 file
    print(f"Saving features to {output_file}")
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('features', data=features_array)
        f.create_dataset('coral_names', data=[name.encode('utf-8') for name in valid_coral_names])
        f.attrs['model'] = 'dinov2_vitb14'
        f.attrs['feature_dim'] = features_array.shape[1]
        f.attrs['num_samples'] = features_array.shape[0]
        f.attrs['year'] = year
        f.attrs['area_id'] = area_id
    
    print(f"Successfully saved {len(features_list)} features to {output_file}")
    return output_file

def main():
    """Main function to extract features from both directories, separated by area"""
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model once
    print("Loading DINOv2 ViT-B/14 model...")
    model = load_dinov2_model()
    model.to(device)
    
    # Get transform
    transform = get_image_transform()
    
    processed_files = []
    
    # Process 2022sample by area
    print("\n" + "="*50)
    print("Processing 2022sample")
    print("="*50)
    areas_2022 = get_area_directories('2022sample')
    print(f"Found areas in 2022sample: {areas_2022}")
    
    for area_id in areas_2022:
        output_file = extract_features_from_area('2022sample', area_id, '2022', model, transform, device)
        if output_file:
            processed_files.append(output_file)
    
    # Process 2023sample by area
    print("\n" + "="*50)
    print("Processing 2023sample")
    print("="*50)
    areas_2023 = get_area_directories('2023sample')
    print(f"Found areas in 2023sample: {areas_2023}")
    
    for area_id in areas_2023:
        output_file = extract_features_from_area('2023sample', area_id, '2023', model, transform, device)
        if output_file:
            processed_files.append(output_file)
    
    print("\n" + "="*50)
    print("Feature extraction complete!")
    print("="*50)
    print("Generated files:")
    for file in processed_files:
        print(f"  - {file}")
    print(f"Total files generated: {len(processed_files)}")

if __name__ == "__main__":
    main()