#!/usr/bin/env python3
"""
Fine-tune DINOv2 for coral identification using triplet loss
Progressive fine-tuning approach with loss visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import logging
from tqdm import tqdm
from PIL import Image
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dinov2_coral_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping utility to prevent overfitting"""
    
    def __init__(self, patience=5, delta=0.001, verbose=True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = float('inf')
        self.counter = 0
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.verbose and self.counter >= self.patience:
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
            return self.counter >= self.patience

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
    
    def unfreeze_backbone(self, unfreeze_layers=4):
        """Unfreeze last N transformer blocks for fine-tuning"""
        logger.info(f"Unfreezing last {unfreeze_layers} transformer blocks")
        
        # Get all transformer blocks
        blocks = list(self.dinov2.blocks)
        
        # Unfreeze last N blocks
        for block in blocks[-unfreeze_layers:]:
            for param in block.parameters():
                param.requires_grad = True
        
        # Also unfreeze norm layer
        for param in self.dinov2.norm.parameters():
            param.requires_grad = True

def get_coral_id_from_path(image_path):
    """Extract coral ID from image path following train_efficientNet.py logic"""
    from pathlib import Path
    
    path = Path(image_path)
    parts = path.parts
    file_name = path.name
    
    year_idx = -1
    for i, part in enumerate(parts):
        if part in ['2021', '2022', '2023', '2024']:
            year_idx = i
            break

    if year_idx == -1:
        return None

    if len(parts) < year_idx + 5:
        return None

    file_base = os.path.splitext(file_name)[0]
    coral_id = f"{parts[year_idx + 1]}/{parts[year_idx + 2]}/{parts[year_idx + 3]}/{file_base}"
    return coral_id

def extract_info_from_image_path(image_path):
    """Extract info from image path following train_efficientNet.py logic"""
    from pathlib import Path
    
    path = Path(image_path)
    parts = path.parts
    file_name = path.name
    
    info = {
        'full_path': image_path,
        'file_name': path.name
    }
    
    year_idx = -1
    for i, part in enumerate(parts):
        if part in ['2021', '2022', '2023', '2024']:
            year_idx = i
            break

    if year_idx != -1:
        info['year'] = parts[year_idx]

        if len(parts) > year_idx + 1:
            info['site'] = parts[year_idx + 1]
        if len(parts) > year_idx + 2:
            info['location'] = parts[year_idx + 2]
        if len(parts) > year_idx + 3:
            info['area'] = parts[year_idx + 3]

        file_base = os.path.splitext(file_name)[0]
        info['coral_number'] = file_base
    else:
        info['year'] = 'unknown'

    info['coral_id'] = get_coral_id_from_path(image_path)
    return info

class CoralTripletDataset(Dataset):
    """Dataset for triplet training with coral images following train_efficientNet.py logic"""
    
    def __init__(self, root_dirs, transform=None, coral_ids=None, use_whole_image=False, same_area_negatives=False):
        self.root_dirs = root_dirs if isinstance(root_dirs, list) else [root_dirs]
        self.transform = transform
        self.coral_ids = coral_ids  # Specific coral IDs to include (for train/val/test split)
        self.use_whole_image = use_whole_image
        self.same_area_negatives = same_area_negatives
        
        # Load and filter images following train_efficientNet.py rules
        self.image_paths = []
        self.valid_image_paths = []
        self.image_bbox_dict = {}
        
        # Collect all image paths
        for root_dir in self.root_dirs:
            for root, _, files in os.walk(root_dir):
                # Skip Pocillopora folders and other non-coral directories
                if any(skip_dir in root.lower() for skip_dir in ['pocillopora', 'orthomosaic']):
                    continue
                    
                for file in files:
                    if (file.lower().endswith(('.jpg', '.jpeg', '.png')) and 
                        not file.lower().startswith('p')):  # Skip P*.JPG files
                        self.image_paths.append(os.path.join(root, file))
        
        # Filter images with valid EXIF bbox and specific coral IDs if provided
        for img_path in self.image_paths:
            if self.use_whole_image:
                # When using whole images, accept all images
                bbox = None
                if self.coral_ids is not None:
                    coral_id = get_coral_id_from_path(img_path)
                    if coral_id not in self.coral_ids:
                        continue
                self.valid_image_paths.append(img_path)
                self.image_bbox_dict[img_path] = bbox
            else:
                # Original behavior: only include images with valid bbox
                bbox = self._read_bbox_from_exif(img_path)
                if bbox is not None:
                    # If coral_ids specified, only include images from those corals
                    if self.coral_ids is not None:
                        coral_id = get_coral_id_from_path(img_path)
                        if coral_id not in self.coral_ids:
                            continue
                    self.valid_image_paths.append(img_path)
                    self.image_bbox_dict[img_path] = bbox
        
        if self.use_whole_image:
            logger.info(f'Found {len(self.image_paths)} images, using whole images (no bbox required)')
        else:
            logger.info(f'Found {len(self.image_paths)} images, {len(self.valid_image_paths)} have bbox')
        if self.coral_ids is not None:
            logger.info(f'Filtered to {len(self.coral_ids)} specific coral IDs')
        self.image_paths = self.valid_image_paths
        
        # Generate triplets using multi-year coral logic
        self.triplets = self._generate_triplets()
        logger.info(f"Generated {len(self.triplets)} triplets")
    
    def _generate_triplets(self, max_triplets_per_coral=50, max_negatives_per_pair=5):
        """Generate exhaustive triplets with multiple negatives per anchor-positive pair"""
        from collections import defaultdict

        triplets = []

        if self.same_area_negatives:
            # Use same-area negatives approach
            return self._generate_triplets_same_area(max_triplets_per_coral, max_negatives_per_pair)

        # Original approach: site-level negatives
        year_coral_images = defaultdict(list)

        # Group images by coral_id and year
        for img_path in self.image_paths:
            info = extract_info_from_image_path(img_path)
            if 'year' in info and info['coral_id']:
                key = info['coral_id']
                year_coral_images[key].append((info['year'], img_path))

        # Find corals with photos from multiple years
        multi_year_corals = {}
        for coral_id, year_img_pairs in year_coral_images.items():
            years = set(year for year, _ in year_img_pairs)
            if len(years) >= 2:
                # Organize by year for easier access
                year_dict = defaultdict(list)
                for year, img_path in year_img_pairs:
                    year_dict[year].append(img_path)
                multi_year_corals[coral_id] = year_dict

        logger.info(f'Found {len(multi_year_corals)} corals with photos from multiple years')

        if len(multi_year_corals) < 2:
            logger.warning("Not enough corals with photos from multiple years for triplet")
            return triplets

        coral_ids = list(multi_year_corals.keys())

        # EXHAUSTIVE TRIPLET GENERATION - Multiple negatives per anchor-positive pair
        for coral_id in coral_ids:
            year_images = multi_year_corals[coral_id]
            years = list(year_images.keys())
            triplet_count = 0

            # Generate all anchor-positive combinations from different years
            for i, year1 in enumerate(years):
                for year2 in years[i+1:]:
                    for anchor_img in year_images[year1]:
                        for positive_img in year_images[year2]:
                            if triplet_count >= max_triplets_per_coral:
                                break

                            # Smart geographic-aware negative selection
                            negative_coral_ids = [cid for cid in coral_ids if cid != coral_id]
                            if negative_coral_ids:
                                num_negatives = min(max_negatives_per_pair, len(negative_coral_ids))
                                selected_neg_corals = self._select_smart_negatives(
                                    coral_id, negative_coral_ids, num_negatives
                                )

                                for neg_coral_id in selected_neg_corals:
                                    neg_year_dict = multi_year_corals[neg_coral_id]
                                    neg_year = random.choice(list(neg_year_dict.keys()))
                                    negative_img = random.choice(neg_year_dict[neg_year])
                                    triplets.append((anchor_img, positive_img, negative_img))
                                    triplet_count += 1

                                    if triplet_count >= max_triplets_per_coral:
                                        break

                            if triplet_count >= max_triplets_per_coral:
                                break
                        if triplet_count >= max_triplets_per_coral:
                            break
                    if triplet_count >= max_triplets_per_coral:
                        break
                if triplet_count >= max_triplets_per_coral:
                    break
        
        # Smart triplet cap - allow more triplets for better training
        max_total_triplets = 50000 if self.coral_ids is None or len(self.coral_ids) > 400 else 20000
        
        if len(triplets) > max_total_triplets:
            logger.info(f'Generated {len(triplets)} triplets, limiting to {max_total_triplets} for memory efficiency')
            random.shuffle(triplets)
            triplets = triplets[:max_total_triplets]

        # Log example triplets
        for i, (anchor, positive, negative) in enumerate(triplets[:3]):
            anchor_info = extract_info_from_image_path(anchor)
            positive_info = extract_info_from_image_path(positive)
            negative_info = extract_info_from_image_path(negative)
            
            logger.info(f"Example triplet {i+1}:")
            logger.info(f"  Anchor: {os.path.basename(anchor)}, Year: {anchor_info.get('year', 'unknown')}, Coral ID: {anchor_info['coral_id']}")
            logger.info(f"  Positive: {os.path.basename(positive)}, Year: {positive_info.get('year', 'unknown')}, Coral ID: {positive_info['coral_id']}")
            logger.info(f"  Negative: {os.path.basename(negative)}, Year: {negative_info.get('year', 'unknown')}, Coral ID: {negative_info['coral_id']}")
        
        return triplets
    
    def _select_smart_negatives(self, anchor_coral_id, negative_coral_ids, num_negatives):
        """Select negatives with geographic awareness for harder mining"""
        # Extract site information from coral ID
        anchor_parts = anchor_coral_id.split('/')
        anchor_site = anchor_parts[0] if len(anchor_parts) > 0 else None
        
        if anchor_site is None:
            return random.sample(negative_coral_ids, num_negatives)
        
        # Categorize negatives by site relationship
        same_site_negatives = []
        different_site_negatives = []
        
        for neg_coral_id in negative_coral_ids:
            neg_parts = neg_coral_id.split('/')
            neg_site = neg_parts[0] if len(neg_parts) > 0 else None
            
            if neg_site == anchor_site:
                same_site_negatives.append(neg_coral_id)
            else:
                different_site_negatives.append(neg_coral_id)
        
        # Smart selection strategy: 60% same site (harder), 40% different site (easier)
        selected_negatives = []
        num_hard = int(num_negatives * 0.6)
        num_easy = num_negatives - num_hard
        
        # Select hard negatives (same site)
        if same_site_negatives and num_hard > 0:
            num_hard_available = min(num_hard, len(same_site_negatives))
            selected_negatives.extend(random.sample(same_site_negatives, num_hard_available))
            num_easy += num_hard - num_hard_available  # Add remainder to easy
        
        # Select easy negatives (different site)
        if different_site_negatives and num_easy > 0:
            num_easy_available = min(num_easy, len(different_site_negatives))
            selected_negatives.extend(random.sample(different_site_negatives, num_easy_available))
        
        # If we still need more negatives, fill from any remaining
        remaining_negatives = [cid for cid in negative_coral_ids if cid not in selected_negatives]
        if len(selected_negatives) < num_negatives and remaining_negatives:
            additional_needed = num_negatives - len(selected_negatives)
            additional_count = min(additional_needed, len(remaining_negatives))
            selected_negatives.extend(random.sample(remaining_negatives, additional_count))
        
        return selected_negatives[:num_negatives]

    def _generate_triplets_same_area(self, max_triplets_per_coral=50, max_negatives_per_pair=5):
        """Generate triplets with negatives restricted to the same area (Tag)"""
        from collections import defaultdict

        triplets = []

        # Group images by area and coral
        area_coral_years = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for img_path in self.image_paths:
            info = extract_info_from_image_path(img_path)
            if 'year' in info and info['coral_id']:
                # Extract area key (site/location/area)
                coral_parts = info['coral_id'].split('/')
                if len(coral_parts) >= 4:
                    area_key = '/'.join(coral_parts[:3])  # site/location/area
                    coral_name = coral_parts[3]  # individual coral filename
                    year = info['year']
                    area_coral_years[area_key][coral_name][year].append(img_path)

        logger.info(f'Found {len(area_coral_years)} areas (Tags)')

        # For each area, find multi-year corals
        viable_areas = 0
        excluded_corals = 0

        for area_key, coral_dict in area_coral_years.items():
            # Find multi-year corals in this area
            multi_year_corals = {}
            for coral_name, year_dict in coral_dict.items():
                if len(year_dict) >= 2:
                    multi_year_corals[coral_name] = year_dict

            # Need at least 2 multi-year corals in same area for same-area negatives
            if len(multi_year_corals) < 2:
                excluded_corals += len(multi_year_corals)
                continue

            viable_areas += 1

            # Generate triplets for this area
            coral_names = list(multi_year_corals.keys())

            for anchor_coral_name in coral_names:
                year_images = multi_year_corals[anchor_coral_name]
                years = list(year_images.keys())
                triplet_count = 0

                # Generate all anchor-positive combinations from different years
                for i, year1 in enumerate(years):
                    for year2 in years[i+1:]:
                        for anchor_img in year_images[year1]:
                            for positive_img in year_images[year2]:
                                if triplet_count >= max_triplets_per_coral:
                                    break

                                # Select negatives from OTHER corals in SAME area
                                negative_coral_names = [c for c in coral_names if c != anchor_coral_name]
                                if negative_coral_names:
                                    num_negatives = min(max_negatives_per_pair, len(negative_coral_names))
                                    selected_neg_corals = random.sample(negative_coral_names, num_negatives)

                                    for neg_coral_name in selected_neg_corals:
                                        neg_year_dict = multi_year_corals[neg_coral_name]
                                        neg_year = random.choice(list(neg_year_dict.keys()))
                                        negative_img = random.choice(neg_year_dict[neg_year])
                                        triplets.append((anchor_img, positive_img, negative_img))
                                        triplet_count += 1

                                        if triplet_count >= max_triplets_per_coral:
                                            break

                                if triplet_count >= max_triplets_per_coral:
                                    break
                            if triplet_count >= max_triplets_per_coral:
                                break
                        if triplet_count >= max_triplets_per_coral:
                            break
                    if triplet_count >= max_triplets_per_coral:
                        break

        logger.info(f'Same-area negatives: {viable_areas} viable areas, {excluded_corals} corals excluded (only 1 multi-year coral in their area)')

        # Smart triplet cap - allow more triplets for better training
        max_total_triplets = 50000 if self.coral_ids is None or len(self.coral_ids) > 400 else 20000

        if len(triplets) > max_total_triplets:
            logger.info(f'Generated {len(triplets)} triplets, limiting to {max_total_triplets} for memory efficiency')
            random.shuffle(triplets)
            triplets = triplets[:max_total_triplets]

        # Log example triplets
        for i, (anchor, positive, negative) in enumerate(triplets[:3]):
            anchor_info = extract_info_from_image_path(anchor)
            positive_info = extract_info_from_image_path(positive)
            negative_info = extract_info_from_image_path(negative)

            logger.info(f"Example triplet {i+1} (same-area negatives):")
            logger.info(f"  Anchor: {os.path.basename(anchor)}, Year: {anchor_info.get('year', 'unknown')}, Coral ID: {anchor_info['coral_id']}")
            logger.info(f"  Positive: {os.path.basename(positive)}, Year: {positive_info.get('year', 'unknown')}, Coral ID: {positive_info['coral_id']}")
            logger.info(f"  Negative: {os.path.basename(negative)}, Year: {negative_info.get('year', 'unknown')}, Coral ID: {negative_info['coral_id']}")

        return triplets

    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        anchor_path, positive_path, negative_path = self.triplets[idx]

        # Get bbox for each image
        anchor_bbox = self.image_bbox_dict.get(anchor_path)
        positive_bbox = self.image_bbox_dict.get(positive_path)
        negative_bbox = self.image_bbox_dict.get(negative_path)
        
        # Load and transform images with bbox cropping
        anchor_img = self._load_image_with_bbox(anchor_path, anchor_bbox)
        positive_img = self._load_image_with_bbox(positive_path, positive_bbox)
        negative_img = self._load_image_with_bbox(negative_path, negative_bbox)
        
        return (anchor_img, positive_img, negative_img), (anchor_path, positive_path, negative_path)
    
    def _load_image_with_bbox(self, img_path, bbox, enable_multi_crop=True):
        """Load and transform image with bbox cropping and optional multi-crop augmentation"""
        img = Image.open(img_path).convert('RGB')
        
        # Crop using provided bbox only if not using whole image
        if not self.use_whole_image and bbox is not None:
            img = self._crop_image_with_bbox(img, bbox)
            
            # Multi-crop augmentation during training (30% chance)
            if enable_multi_crop and self.transform and random.random() < 0.3:
                img = self._apply_multi_crop_augmentation(img, bbox)
        
        if self.transform:
            img = self.transform(img)
        return img
    
    def _apply_multi_crop_augmentation(self, img, original_bbox):
        """Apply multi-crop augmentation around the coral region"""
        width, height = img.size
        
        # Create slightly varied crops around the original region
        crop_variations = [
            (0.85, 0.85),  # Tighter crop
            (1.0, 1.0),    # Original size  
            (1.15, 1.15),  # Looser crop
        ]
        
        # Randomly select a crop variation
        scale_x, scale_y = random.choice(crop_variations)
        
        # Calculate new crop dimensions
        new_width = min(width, int(width * scale_x))
        new_height = min(height, int(height * scale_y))
        
        # Random offset within reasonable bounds
        max_offset_x = max(0, (width - new_width) // 4)
        max_offset_y = max(0, (height - new_height) // 4)
        
        offset_x = random.randint(-max_offset_x, max_offset_x) if max_offset_x > 0 else 0
        offset_y = random.randint(-max_offset_y, max_offset_y) if max_offset_y > 0 else 0
        
        # Calculate crop bounds
        left = max(0, (width - new_width) // 2 + offset_x)
        top = max(0, (height - new_height) // 2 + offset_y)
        right = min(width, left + new_width)
        bottom = min(height, top + new_height)
        
        # Apply crop if valid
        if right > left and bottom > top:
            img = img.crop((left, top, right, bottom))
        
        return img
    
    def _read_bbox_from_exif(self, img_path):
        """Read bounding box from EXIF data"""
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
    
    def _crop_image_with_bbox(self, img, bbox):
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

def split_coral_ids(image_paths, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """Split coral IDs into train/validation/test sets with stratification"""
    random.seed(random_seed)
    
    # Group images by coral_id and collect metadata
    coral_data = defaultdict(lambda: {'images': [], 'years': set(), 'sites': set()})
    
    for img_path in image_paths:
        info = extract_info_from_image_path(img_path)
        coral_id = info.get('coral_id')
        if coral_id:
            coral_data[coral_id]['images'].append(img_path)
            if 'year' in info:
                coral_data[coral_id]['years'].add(info['year'])
            if 'site' in info:
                coral_data[coral_id]['sites'].add(info['site'])
    
    # Filter multi-year corals for triplet training
    multi_year_corals = {
        coral_id: data for coral_id, data in coral_data.items() 
        if len(data['years']) >= 2
    }
    
    logger.info(f'Found {len(multi_year_corals)} corals with photos from multiple years')
    
    if len(multi_year_corals) < 3:
        logger.warning("Not enough multi-year corals for proper train/val/test split")
        return list(multi_year_corals.keys()), [], []
    
    # Create stratification keys (year_site combinations)
    coral_strata = {}
    for coral_id, data in multi_year_corals.items():
        # Use primary year and site for stratification
        primary_year = sorted(data['years'])[0]
        primary_site = sorted(data['sites'])[0] if data['sites'] else 'unknown'
        strata_key = f"{primary_year}_{primary_site}"
        coral_strata[coral_id] = strata_key
    
    # Group corals by strata
    strata_groups = defaultdict(list)
    for coral_id, strata_key in coral_strata.items():
        strata_groups[strata_key].append(coral_id)
    
    # Split each stratum proportionally
    train_corals, val_corals, test_corals = [], [], []
    
    for strata_key, coral_list in strata_groups.items():
        random.shuffle(coral_list)
        n_corals = len(coral_list)
        
        # Calculate splits
        n_train = max(1, int(n_corals * train_ratio))
        n_val = max(1, int(n_corals * val_ratio)) if n_corals > 2 else 0
        n_test = n_corals - n_train - n_val
        
        # Ensure at least one coral in each split if possible
        if n_corals >= 3:
            if n_val == 0:
                n_val = 1
                n_train -= 1
            if n_test == 0:
                n_test = 1
                n_train -= 1
        
        # Assign corals to splits
        train_corals.extend(coral_list[:n_train])
        if n_val > 0:
            val_corals.extend(coral_list[n_train:n_train + n_val])
        if n_test > 0:
            test_corals.extend(coral_list[n_train + n_val:])
    
    logger.info(f'Split: {len(train_corals)} train, {len(val_corals)} val, {len(test_corals)} test corals')
    
    # Log distribution by strata
    for strata_key, coral_list in strata_groups.items():
        train_count = len([c for c in coral_list if c in train_corals])
        val_count = len([c for c in coral_list if c in val_corals])
        test_count = len([c for c in coral_list if c in test_corals])
        logger.info(f'Strata {strata_key}: {train_count} train, {val_count} val, {test_count} test')
    
    return train_corals, val_corals, test_corals

class TripletLossWithMining(nn.Module):
    """Enhanced triplet loss with hard negative mining"""
    
    def __init__(self, margin=0.3, mining_strategy='hard'):
        super(TripletLossWithMining, self).__init__()
        self.margin = margin
        self.mining_strategy = mining_strategy
        
    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        if self.mining_strategy == 'hard' and len(neg_dist) > 1:
            # Select hardest negatives (smallest negative distances)
            hard_negatives = torch.topk(neg_dist, k=min(len(neg_dist), 3), largest=False)[0]
            neg_dist = hard_negatives.mean()
        else:
            neg_dist = neg_dist.mean()
            
        pos_dist = pos_dist.mean()
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss

def compute_triplet_accuracy(anchor_emb, positive_emb, negative_emb):
    """Compute triplet accuracy metric beyond loss"""
    pos_dist = F.pairwise_distance(anchor_emb, positive_emb, p=2)
    neg_dist = F.pairwise_distance(anchor_emb, negative_emb, p=2)
    accuracy = (pos_dist < neg_dist).float().mean()
    return accuracy.item()

def get_coral_transforms():
    """Enhanced data augmentation transforms for coral domain"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),  # Add vertical flip
        transforms.RandomRotation(30),  # Increase rotation range
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
        transforms.RandomGrayscale(p=0.1),  # Add grayscale augmentation
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Add blur
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),  # Random erasing
    ])

def train_phase(model, dataloader, optimizer, criterion, device, phase_name, scheduler=None, accumulation_steps=8):
    """Train model for one phase with gradient accumulation, clipping and accuracy tracking"""
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"{phase_name} Training")
    
    # Initialize gradient accumulation
    optimizer.zero_grad()
    accumulated_loss = 0.0
    accumulated_accuracy = 0.0
    
    for batch_idx, ((anchors, positives, negatives), _) in enumerate(pbar):
        anchors = anchors.to(device)
        positives = positives.to(device)
        negatives = negatives.to(device)
        
        # Forward pass
        anchor_embeddings = model(anchors)
        positive_embeddings = model(positives)
        negative_embeddings = model(negatives)
        
        # Compute loss and normalize by accumulation steps
        loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
        loss = loss / accumulation_steps
        
        # Compute accuracy
        accuracy = compute_triplet_accuracy(anchor_embeddings, positive_embeddings, negative_embeddings)
        
        # Backward pass
        loss.backward()
        
        # Accumulate metrics
        accumulated_loss += loss.item() * accumulation_steps  # Restore original scale for logging
        accumulated_accuracy += accuracy
        
        # Perform optimizer step every accumulation_steps batches
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Update totals with accumulated values
            num_accumulated = min(accumulation_steps, (batch_idx + 1) % accumulation_steps or accumulation_steps)
            total_loss += accumulated_loss / num_accumulated
            total_accuracy += accumulated_accuracy / num_accumulated
            num_batches += 1
            
            # Update progress bar with effective batch metrics
            pbar.set_postfix({
                'Loss': f'{accumulated_loss / num_accumulated:.4f}', 
                'Avg Loss': f'{total_loss/num_batches:.4f}',
                'Acc': f'{accumulated_accuracy / num_accumulated:.3f}',
                'Avg Acc': f'{total_accuracy/num_batches:.3f}',
                'Eff BS': f'{32 * accumulation_steps}'  # Show effective batch size
            })
            
            # Reset accumulation
            accumulated_loss = 0.0
            accumulated_accuracy = 0.0
    
    return total_loss / num_batches, total_accuracy / num_batches

def validate_model(model, dataloader, criterion, device):
    """Validate model performance with accuracy tracking"""
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        
        for batch_idx, ((anchors, positives, negatives), _) in enumerate(pbar):
            anchors = anchors.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)
            
            # Forward pass
            anchor_embeddings = model(anchors)
            positive_embeddings = model(positives)
            negative_embeddings = model(negatives)
            
            # Compute loss and accuracy
            loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
            accuracy = compute_triplet_accuracy(anchor_embeddings, positive_embeddings, negative_embeddings)
            
            total_loss += loss.item()
            total_accuracy += accuracy
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Val Loss': f'{loss.item():.4f}', 
                'Avg Val Loss': f'{total_loss/num_batches:.4f}',
                'Val Acc': f'{accuracy:.3f}',
                'Avg Val Acc': f'{total_accuracy/num_batches:.3f}'
            })
    
    return total_loss / num_batches, total_accuracy / num_batches

def main():
    """Main training function with proper train/validation/test splitting"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fine-tune DINOv2 for coral identification')
    parser.add_argument('--use_whole_image', action='store_true',
                        help='Use whole images instead of cropping to bounding boxes')
    parser.add_argument('--same_area_negatives', action='store_true',
                        help='Use same-area (same Tag) negatives for harder training task')
    args = parser.parse_args()
    
    # Configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 16  # Increased batch size for stability
    EMBEDDING_SIZE = 1280
    MARGIN = 0.3  # Reduced margin for better convergence
    
    # Generate timestamp for this training session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Training session timestamp: {timestamp}")

    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Using whole image: {args.use_whole_image}")
    logger.info(f"Using same-area negatives: {args.same_area_negatives}")
    
    # First, collect all valid image paths to determine coral-level splits
    root_dirs = ['annotated_imgs']  # This contains 2021/, 2022/, 2023/, 2024/ subdirectories with professional annotations
    
    # Collect all image paths with valid bboxes
    all_image_paths = []
    temp_dataset = CoralTripletDataset(root_dirs, transform=None)  # Just to collect paths
    all_image_paths = temp_dataset.image_paths
    
    logger.info(f"Found {len(all_image_paths)} total images with bboxes")
    
    # Split coral IDs into train/validation/test sets
    train_coral_ids, val_coral_ids, test_coral_ids = split_coral_ids(all_image_paths)
    
    # Create enhanced transforms
    train_transform = get_coral_transforms()
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create datasets for each split
    train_dataset = CoralTripletDataset(root_dirs, transform=train_transform, coral_ids=set(train_coral_ids),
                                       use_whole_image=args.use_whole_image, same_area_negatives=args.same_area_negatives)
    val_dataset = CoralTripletDataset(root_dirs, transform=val_transform, coral_ids=set(val_coral_ids),
                                     use_whole_image=args.use_whole_image, same_area_negatives=args.same_area_negatives)
    test_dataset = CoralTripletDataset(root_dirs, transform=val_transform, coral_ids=set(test_coral_ids),
                                      use_whole_image=args.use_whole_image, same_area_negatives=args.same_area_negatives)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True)
    
    logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    logger.info(f"Triplet counts - Train: {len(train_dataset.triplets)}, Val: {len(val_dataset.triplets)}, Test: {len(test_dataset.triplets)}")
    
    # Initialize model
    model = FineTunedDINOv2(embedding_size=EMBEDDING_SIZE, freeze_backbone=True)
    model.to(DEVICE)
    
    # Enhanced loss function with hard negative mining
    criterion = TripletLossWithMining(margin=MARGIN, mining_strategy='hard')
    
    # Progressive training phases with conservative learning rates for stability
    phases = [
        {
            'name': 'Phase 1: Head Only',
            'epochs': 20,  # More epochs for better convergence
            'lr': 3e-4,    # Further reduced for stability
            'unfreeze_backbone': False,
            'unfreeze_layers': 0
        },
        {
            'name': 'Phase 2: Head + Last 2 Blocks',
            'epochs': 15,  # More epochs
            'lr': 8e-5,    # More conservative
            'unfreeze_backbone': True,
            'unfreeze_layers': 2
        },
        {
            'name': 'Phase 3: Head + Last 4 Blocks',
            'epochs': 12,  # More epochs
            'lr': 3e-5,    # Very conservative
            'unfreeze_backbone': True,
            'unfreeze_layers': 4
        }
    ]
    
    # Training loop with enhanced tracking
    all_train_losses = []
    all_val_losses = []
    all_train_accuracies = []
    all_val_accuracies = []
    epoch_labels = []
    phase_boundaries = []
    
    total_epochs = 0
    best_val_loss = float('inf')
    
    for phase_idx, phase in enumerate(phases):
        logger.info(f"\nStarting {phase['name']}")
        
        # Update model freezing
        if phase['unfreeze_backbone']:
            model.unfreeze_backbone(phase['unfreeze_layers'])
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], 
            lr=phase['lr'],
            weight_decay=1e-4
        )
        
        # Setup learning rate scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                    patience=3, verbose=True, min_lr=1e-6)
        
        # Setup early stopping with increased patience for better convergence
        early_stopping = EarlyStopping(patience=6, delta=0.0005)
        
        # Mark phase boundary
        if phase_idx > 0:
            phase_boundaries.append(total_epochs)
        
        # Train for specified epochs
        for epoch in range(phase['epochs']):
            logger.info(f"Epoch {epoch+1}/{phase['epochs']}")
            
            # Training phase with gradient accumulation (effective batch size = 32 * 8 = 256)
            avg_train_loss, avg_train_acc = train_phase(model, train_loader, optimizer, criterion, DEVICE, 
                                       f"{phase['name']} Epoch {epoch+1}", scheduler, accumulation_steps=8)
            
            # Validation phase
            avg_val_loss, avg_val_acc = validate_model(model, val_loader, criterion, DEVICE)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            logger.info(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.3f}, "
                       f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.3f}")
            
            # Track losses, accuracies and epoch info
            all_train_losses.append(avg_train_loss)
            all_val_losses.append(avg_val_loss)
            all_train_accuracies.append(avg_train_acc)
            all_val_accuracies.append(avg_val_acc)
            epoch_labels.append(f"P{phase_idx+1}E{epoch+1}")
            total_epochs += 1
            
            # Check for early stopping
            if early_stopping(avg_val_loss):
                logger.info(f"Early stopping triggered in {phase['name']} at epoch {epoch+1}")
                break
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_path = f"dinov2_coral_best_model_{timestamp}.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'phase': phase['name'],
                    'epoch': epoch + 1,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'train_acc': avg_train_acc,
                    'val_acc': avg_val_acc
                }, best_model_path)
                logger.info(f"New best model saved: {best_model_path} (Val Loss: {avg_val_loss:.4f})")
            
            # Save checkpoint
            if (epoch + 1) % 3 == 0:
                checkpoint_path = f"dinov2_coral_checkpoint_phase{phase_idx+1}_epoch{epoch+1}_{timestamp}.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'phase': phase['name'],
                    'epoch': epoch + 1,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'train_acc': avg_train_acc,
                    'val_acc': avg_val_acc
                }, checkpoint_path)
                logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    # Final test evaluation
    logger.info("\nEvaluating on test set...")
    model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
    test_loss, test_acc = validate_model(model, test_loader, criterion, DEVICE)
    logger.info(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.3f}")
    
    # Plot and save loss and accuracy curves with timestamp
    plt.figure(figsize=(20, 8))
    
    # Training and validation loss curves
    plt.subplot(2, 2, 1)
    plt.plot(range(1, total_epochs + 1), all_train_losses, 'b-', linewidth=2, marker='o', markersize=4, label='Train Loss')
    plt.plot(range(1, total_epochs + 1), all_val_losses, 'r-', linewidth=2, marker='s', markersize=4, label='Val Loss')
    
    # Add phase boundaries
    for boundary in phase_boundaries:
        plt.axvline(x=boundary + 0.5, color='gray', linestyle='--', alpha=0.7)
    
    plt.xlabel('Epoch')
    plt.ylabel('Average Triplet Loss')
    plt.title('DINOv2 Fine-tuning: Train vs Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Training and validation accuracy curves  
    plt.subplot(2, 2, 2)
    plt.plot(range(1, total_epochs + 1), all_train_accuracies, 'b-', linewidth=2, marker='o', markersize=4, label='Train Accuracy')
    plt.plot(range(1, total_epochs + 1), all_val_accuracies, 'r-', linewidth=2, marker='s', markersize=4, label='Val Accuracy')
    
    # Add phase boundaries
    for boundary in phase_boundaries:
        plt.axvline(x=boundary + 0.5, color='gray', linestyle='--', alpha=0.7)
    
    plt.xlabel('Epoch')
    plt.ylabel('Triplet Accuracy')
    plt.title('DINOv2 Fine-tuning: Train vs Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Phase breakdown - Loss
    plt.subplot(2, 2, 3)
    phase_starts = [0] + phase_boundaries + [total_epochs]
    for i, phase in enumerate(phases):
        start = phase_starts[i]
        end = phase_starts[i + 1]
        epochs_range = range(start + 1, end + 1)
        train_subset = all_train_losses[start:end]
        val_subset = all_val_losses[start:end]
        
        plt.plot(epochs_range, train_subset, 'o-', label=f'{phase["name"]} Train', alpha=0.8)
        plt.plot(epochs_range, val_subset, 's-', label=f'{phase["name"]} Val', alpha=0.8)
    
    plt.xlabel('Epoch')
    plt.ylabel('Average Triplet Loss')
    plt.title('Loss by Training Phase')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Phase breakdown - Accuracy
    plt.subplot(2, 2, 4)
    phase_starts = [0] + phase_boundaries + [total_epochs]
    for i, phase in enumerate(phases):
        start = phase_starts[i]
        end = phase_starts[i + 1]
        epochs_range = range(start + 1, end + 1)
        train_acc_subset = all_train_accuracies[start:end]
        val_acc_subset = all_val_accuracies[start:end]
        
        plt.plot(epochs_range, train_acc_subset, 'o-', label=f'{phase["name"]} Train Acc', alpha=0.8)
        plt.plot(epochs_range, val_acc_subset, 's-', label=f'{phase["name"]} Val Acc', alpha=0.8)
    
    plt.xlabel('Epoch')
    plt.ylabel('Triplet Accuracy')
    plt.title('Accuracy by Training Phase')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot with timestamp
    loss_plot_path = f'dinov2_coral_training_loss_{timestamp}.png'
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Loss curves saved: {loss_plot_path}")
    plt.close()
    
    # Save loss and accuracy data as numpy arrays with timestamp
    np.save(f'dinov2_coral_train_losses_{timestamp}.npy', np.array(all_train_losses))
    np.save(f'dinov2_coral_val_losses_{timestamp}.npy', np.array(all_val_losses))
    np.save(f'dinov2_coral_train_accuracies_{timestamp}.npy', np.array(all_train_accuracies))
    np.save(f'dinov2_coral_val_accuracies_{timestamp}.npy', np.array(all_val_accuracies))
    logger.info(f"Training data saved with timestamp: {timestamp}")
    
    # Save final model with timestamp
    final_model_path = f"dinov2_coral_finetuned_final_{timestamp}.pt"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved: {final_model_path}")
    
    # Save training summary
    summary = {
        'coral_splits': {
            'train_corals': len(train_coral_ids),
            'val_corals': len(val_coral_ids),
            'test_corals': len(test_coral_ids)
        },
        'dataset_sizes': {
            'train_triplets': len(train_dataset.triplets),
            'val_triplets': len(val_dataset.triplets),
            'test_triplets': len(test_dataset.triplets)
        },
        'final_performance': {
            'best_val_loss': best_val_loss,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'best_model_path': best_model_path
        }
    }
    
    import json
    summary_path = f'dinov2_coral_training_summary_{timestamp}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Training summary saved: {summary_path}")
    
    logger.info("Fine-tuning completed with proper train/validation/test evaluation!")

if __name__ == "__main__":
    main()
