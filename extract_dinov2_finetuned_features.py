#!/usr/bin/env python3
"""
Extract fine-tuned DINOv2 features from coral images in 2022sample and 2023sample directories
Uses the coral-specific fine-tuned DINOv2 model (dinov2_coral_finetuned.pt)

Copyright (C) 2025 YuC13600
This source code is licensed under the GPL-3.0 license.
You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

Description:
    This script extracts 1280-dimensional features from coral images using a fine-tuned
    DINOv2 model. It processes images from test directories organized by area and year,
    with support for both bbox cropping and whole-image modes.

    The extracted features can be compared across years to evaluate coral re-identification
    performance. Feature extraction mode (bbox vs whole image) must match the mode used
    during model training.

Usage:
    # Extract features using bbox cropping (for models trained with bbox)
    python extract_dinov2_finetuned_features.py

    # Extract features using whole images (for models trained with --use_whole_image)
    python extract_dinov2_finetuned_features.py --use_whole_image

    # IMPORTANT: Run from the coralscop conda environment
    zsh -c "conda activate coralscop && python extract_dinov2_finetuned_features.py"

Input:
    - Fine-tuned model: dinov2_coral_finetuned_final_{timestamp}.pt
    - 2022sample/ directory: Test images from 2022 organized by area (37, 38, 39, 40)
    - 2023sample/ directory: Test images from 2023 organized by area (37, 38, 39, 40)
    - For bbox mode: Images must have bounding box data in EXIF XPComment field

Output:
    - dinov2_finetuned_{year}_{area_id}_{cropped|whole}_features.h5
    - Each file contains: 1280-dim features, coral_names, and metadata
    - Feature files can be used with compare_dinov2_features.py for evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import h5py
from tqdm import tqdm
import cv2
import argparse

class FineTunedDINOv2(nn.Module):
    """DINOv2 with coral-specific projection head"""
    
    def __init__(self, embedding_size=1280, freeze_backbone=True):
        super(FineTunedDINOv2, self).__init__()
        
        # Load DINOv2 backbone
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        
        # Freeze backbone initially
        if freeze_backbone:
            for param in self.dinov2.parameters():
                param.requires_grad = False
        
        # Coral-specific projection head
        self.projection_head = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )
        
        # Feature processor for stability
        self.feature_processor = nn.Sequential(
            nn.BatchNorm1d(768),
            nn.Dropout(0.2),
        )
    
    def forward(self, x):
        # Extract DINOv2 features
        if self.training and not any(p.requires_grad for p in self.dinov2.parameters()):
            # Use no_grad for frozen backbone to save memory
            with torch.no_grad():
                dinov2_features = self.dinov2(x)
        else:
            dinov2_features = self.dinov2(x)
        
        # Process and project features
        processed_features = self.feature_processor(dinov2_features)
        projected_features = self.projection_head(processed_features)
        
        # L2 normalize for cosine similarity
        normalized_features = F.normalize(projected_features, p=2, dim=1)
        return normalized_features

def load_finetuned_dinov2_model(model_path="dinov2_coral_finetuned_final_20250812_152526.pt"):
    """Load fine-tuned DINOv2 model"""
    print(f"Loading fine-tuned DINOv2 model from {model_path}")
    
    # Create model
    model = FineTunedDINOv2(embedding_size=1280, freeze_backbone=True)
    
    # Load state dict
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print("Fine-tuned model loaded successfully")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model.eval()
    return model

def get_image_transform():
    """Get image preprocessing transform for DINOv2 (same as original)"""
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform

def read_bbox_from_exif(img_path):
    """Read bounding box from EXIF data (same as original)"""
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
    """Crop image using bounding box (same as original)"""
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

def extract_features(model, img_path, transform, device, use_whole_image=False):
    """Extract fine-tuned DINOv2 features from a single image"""
    try:
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Try to read bbox from EXIF and crop if available (only if not using whole image)
        if not use_whole_image:
            bbox = read_bbox_from_exif(img_path)
            if bbox is not None:
                img = crop_image_with_bbox(img, bbox)
        
        # Apply transforms
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Extract features using fine-tuned model
        with torch.no_grad():
            features = model(img_tensor)
        
        return features.cpu().numpy().flatten()
    
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def collect_images_from_area(base_dir, area_id):
    """Collect all JPG images from a specific area directory (same as original)"""
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
    """Get all area directories (e.g., 37, 38, 39, 40) (same as original)"""
    areas = []
    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path) and item.isdigit():
                areas.append(item)
    return sorted(areas)

def extract_features_from_area(base_dir, area_id, year, model, transform, device, use_whole_image=False):
    """Extract fine-tuned DINOv2 features from all images in a specific area"""
    print(f"Processing area {area_id} from {year}")
    if use_whole_image:
        print(f"Using whole images (no bbox cropping)")
    else:
        print(f"Using bbox cropping from EXIF data")
    
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
        features = extract_features(model, img_path, transform, device, use_whole_image)
        if features is not None:
            features_list.append(features)
            valid_coral_names.append(coral_name)
    
    if len(features_list) == 0:
        print(f"No valid features extracted from area {area_id}!")
        return None
    
    # Convert to numpy array
    features_array = np.array(features_list)
    print(f"Area {area_id} extracted features shape: {features_array.shape}")
    
    # Generate output filename with "finetuned" indicator and image type
    image_type = "whole" if use_whole_image else "cropped"
    output_file = f"dinov2_finetuned_{year}_{area_id}_{image_type}_features.h5"
    
    # Save to HDF5 file
    print(f"Saving features to {output_file}")
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('features', data=features_array)
        f.create_dataset('coral_names', data=[name.encode('utf-8') for name in valid_coral_names])
        f.attrs['model'] = 'dinov2_finetuned_coral'
        f.attrs['feature_dim'] = features_array.shape[1]
        f.attrs['num_samples'] = features_array.shape[0]
        f.attrs['year'] = year
        f.attrs['area_id'] = area_id
        f.attrs['model_file'] = 'dinov2_coral_finetuned_final_20250812_152526.pt'
        f.attrs['embedding_size'] = 1280
        f.attrs['use_whole_image'] = use_whole_image
    
    print(f"Successfully saved {len(features_list)} features to {output_file}")
    return output_file

def main():
    """Main function to extract features from both directories using fine-tuned model"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract fine-tuned DINOv2 features from coral images')
    parser.add_argument('--use_whole_image', action='store_true', 
                        help='Use whole images instead of cropping to bounding boxes')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Using whole image: {args.use_whole_image}")
    
    # Load fine-tuned model
    print("Loading fine-tuned DINOv2 model...")
    model = load_finetuned_dinov2_model("dinov2_coral_finetuned_final_20250812_152526.pt")
    model.to(device)
    
    # Get transform
    transform = get_image_transform()
    
    processed_files = []
    
    # Process 2022sample by area
    print("\n" + "="*60)
    print("Processing 2022sample with fine-tuned DINOv2")
    print("="*60)
    areas_2022 = get_area_directories('2022sample')
    print(f"Found areas in 2022sample: {areas_2022}")
    
    for area_id in areas_2022:
        output_file = extract_features_from_area('2022sample', area_id, '2022', model, transform, device, args.use_whole_image)
        if output_file:
            processed_files.append(output_file)
    
    # Process 2023sample by area
    print("\n" + "="*60)
    print("Processing 2023sample with fine-tuned DINOv2")
    print("="*60)
    areas_2023 = get_area_directories('2023sample')
    print(f"Found areas in 2023sample: {areas_2023}")
    
    for area_id in areas_2023:
        output_file = extract_features_from_area('2023sample', area_id, '2023', model, transform, device, args.use_whole_image)
        if output_file:
            processed_files.append(output_file)
    
    print("\n" + "="*60)
    print("Fine-tuned DINOv2 feature extraction complete!")
    print("="*60)
    print("Generated files:")
    for file in processed_files:
        print(f"  - {file}")
    print(f"Total files generated: {len(processed_files)}")
    print(f"Feature dimension: 1280 (vs 768 for original DINOv2)")

if __name__ == "__main__":
    main()
