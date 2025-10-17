#!/usr/bin/env python3
"""
Compare DINOv2 features using the same logic as multiple_features_comparison.py
"""

import h5py
import numpy as np
import argparse
import os
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
import pandas as pd
from tqdm import tqdm

def compare_dinov2_features_batch(file1_path, file2_path, use_euclidean=False):
    """
    Compare two files containing multiple DINOv2 feature vectors, find similarity ranking for corals with same ID
    
    Args:
        file1_path: Path to first h5 file containing multiple feature vectors (reference set)
        file2_path: Path to second h5 file containing multiple feature vectors (query set)
        use_euclidean: Whether to use Euclidean distance instead of cosine similarity
    """
    try:
        # Check if files exist
        if not os.path.exists(file1_path):
            raise FileNotFoundError(f"Reference feature file not found: {file1_path}")
        if not os.path.exists(file2_path):
            raise FileNotFoundError(f"Query feature file not found: {file2_path}")
        
        # Load first file (reference set)
        with h5py.File(file1_path, 'r') as f:
            # Check file format
            if 'features' not in f or 'coral_names' not in f:
                available_keys = list(f.keys())
                raise KeyError(f"Invalid file format, required datasets not found. Available datasets: {available_keys}")
            
            file1_features = f['features'][:]
            file1_coral_names = f['coral_names'][:]
            
            # Process coral names (from bytes to string)
            if isinstance(file1_coral_names[0], bytes):
                file1_coral_names = [name.decode('utf-8') for name in file1_coral_names]
            
            # Check model info if available
            model_info = f.attrs.get('model', 'unknown')
            feature_dim = f.attrs.get('feature_dim', file1_features.shape[1])
            print(f"Reference set model: {model_info}, feature dim: {feature_dim}")
            
        # Load second file (query set)
        with h5py.File(file2_path, 'r') as f:
            if 'features' not in f or 'coral_names' not in f:
                available_keys = list(f.keys())
                raise KeyError(f"Invalid file format, required datasets not found. Available datasets: {available_keys}")
            
            file2_features = f['features'][:]
            file2_coral_names = f['coral_names'][:]
            
            # Process coral names (from bytes to string)
            if isinstance(file2_coral_names[0], bytes):
                file2_coral_names = [name.decode('utf-8') for name in file2_coral_names]
            
            # Check model info if available
            model_info = f.attrs.get('model', 'unknown')
            feature_dim = f.attrs.get('feature_dim', file2_features.shape[1])
            print(f"Query set model: {model_info}, feature dim: {feature_dim}")
        
        print(f"Reference set contains {len(file1_coral_names)} coral feature vectors")
        print(f"Query set contains {len(file2_coral_names)} coral feature vectors")
        
        # Check if feature vector dimensions match
        if file1_features.shape[1] != file2_features.shape[1]:
            raise ValueError(f"Feature vector dimensions don't match: reference {file1_features.shape[1]}, query {file2_features.shape[1]}")
        
        # Result storage
        results = []
        
        # Setup progress bar
        progress_bar = tqdm(enumerate(zip(file2_coral_names, file2_features)), 
                           desc="Processing", total=len(file2_coral_names))
        
        # Calculate ranking for each feature vector in query set
        for i, (query_name, query_feature) in progress_bar:
            # Calculate similarity or distance with reference set
            if use_euclidean:
                # Calculate Euclidean distance
                distances = cdist(query_feature.reshape(1, -1), file1_features, 'euclidean')[0]
                # Convert distance to ranking (smaller distance = higher rank)
                similarity_indices = np.argsort(distances)
                metric_name = "Euclidean Distance"
                metric_values = distances[similarity_indices][:5]  # Top 5 values
            else:
                # Calculate cosine similarity
                similarities = cosine_similarity(query_feature.reshape(1, -1), file1_features)[0]
                # Convert similarity to ranking (higher similarity = higher rank)
                similarity_indices = np.argsort(-similarities)
                metric_name = "Cosine Similarity"
                metric_values = similarities[similarity_indices][:5]  # Top 5 values
            
            # Get ranking order of all corals in reference set
            ranked_names = [file1_coral_names[idx] for idx in similarity_indices]
            
            # Find position of same ID in ranking
            try:
                rank = ranked_names.index(query_name) + 1  # +1 because index starts from 0, but ranking starts from 1
                results.append((query_name, rank, metric_values[rank-1] if rank <= 5 else None))
                progress_bar.set_postfix({"current": f"{query_name} Rank: {rank}"})
            except ValueError:
                # If no same ID found in reference set
                results.append((query_name, None, None))
                progress_bar.set_postfix({"current": f"{query_name} Not Found"})
        
        # Output detailed results
        print(f"\n--------- Detailed Ranking Results ({metric_name}) ---------")
        for name, rank, metric_value in results:
            if rank is not None:
                if metric_value is not None:
                    print(f"{name} Ranking: {rank} ({metric_name}: {metric_value:.4f})")
                else:
                    print(f"{name} Ranking: {rank}")
            else:
                print(f"{name} - No matching ID found in reference set")
        
        # Calculate statistics
        valid_ranks = [rank for _, rank, _ in results if rank is not None]
        if valid_ranks:
            print("\n--------- Statistical Analysis ---------")
            # Basic statistics
            avg_rank = sum(valid_ranks) / len(valid_ranks)
            median_rank = np.median(valid_ranks)
            print(f"Average rank: {avg_rank:.2f}")
            print(f"Median rank: {median_rank:.1f}")
            
            # Calculate top ranking ratios
            top1_count = sum(1 for rank in valid_ranks if rank == 1)
            top1_ratio = top1_count / len(valid_ranks)
            print(f"Rank 1 ratio: {top1_ratio:.2%} ({top1_count}/{len(valid_ranks)})")
            
            top3_count = sum(1 for rank in valid_ranks if rank <= 3)
            top3_ratio = top3_count / len(valid_ranks)
            print(f"Top 3 ratio: {top3_ratio:.2%} ({top3_count}/{len(valid_ranks)})")
            
            top5_count = sum(1 for rank in valid_ranks if rank <= 5)
            top5_ratio = top5_count / len(valid_ranks)
            print(f"Top 5 ratio: {top5_ratio:.2%} ({top5_count}/{len(valid_ranks)})")
            
            top10_count = sum(1 for rank in valid_ranks if rank <= 10)
            top10_ratio = top10_count / len(valid_ranks)
            print(f"Top 10 ratio: {top10_ratio:.2%} ({top10_count}/{len(valid_ranks)})")
            
            # Ranking distribution
            print("\nRanking distribution:")
            rank_counts = {}
            for rank in valid_ranks:
                rank_counts[rank] = rank_counts.get(rank, 0) + 1
            
            # Show all ranks, but limit output if there are too many
            sorted_ranks = sorted(rank_counts.keys())
            if len(sorted_ranks) <= 20:
                # Show all ranks if <= 20
                ranks_to_show = sorted_ranks
            else:
                # Show first 10 and last 5 ranks if > 20
                ranks_to_show = sorted_ranks[:10] + ['...'] + sorted_ranks[-5:]
            
            for rank in ranks_to_show:
                if rank == '...':
                    print(f"  ...")
                else:
                    count = rank_counts[rank]
                    ratio = count / len(valid_ranks)
                    print(f"  Rank {rank}: {count} samples ({ratio:.2%})")
            
            # Additional statistics for DINOv2 evaluation
            print(f"\nWorst rank: {max(valid_ranks)}")
            print(f"Best rank: {min(valid_ranks)}")
            print(f"Standard deviation: {np.std(valid_ranks):.2f}")
        
        return results
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def interactive_mode():
    """Interactive mode: get file paths through user input"""
    print("\nDINOv2 Coral Feature Vector Similarity Comparison Tool")
    print("-" * 60)
    
    file1 = input("Enter reference set feature vector h5 file path: ")
    file2 = input("Enter query set feature vector h5 file path: ")
    
    method = input("Select comparison method (1: Cosine similarity, 2: Euclidean distance, default is 1): ")
    use_euclidean = (method == '2')
    
    compare_dinov2_features_batch(file1, file2, use_euclidean)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='DINOv2 Coral Feature Vector Similarity Batch Comparison Tool')
    parser.add_argument('file1', nargs='?', help='Reference set feature vector h5 file path')
    parser.add_argument('file2', nargs='?', help='Query set feature vector h5 file path')
    parser.add_argument('--euclidean', action='store_true', help='Use Euclidean distance calculation (default is cosine similarity)')    
    args = parser.parse_args()
    
    if args.file1 and args.file2:
        # Command line mode
        compare_dinov2_features_batch(args.file1, args.file2, args.euclidean)
    else:
        # Interactive mode
        interactive_mode()