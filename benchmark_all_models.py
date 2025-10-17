#!/usr/bin/env python3
"""
Automated benchmarking script for all 6 DINOv2 fine-tuned models.
This script will:
1. For each model, update extract_dinov2_finetuned_features.py to load the correct model
2. Run feature extraction (with or without --use_whole_image)
3. Run comparisons for all 4 areas (37, 38, 39, 40) comparing 2022 vs 2023
4. Save all output to {model_timestamp}_nearest_benchmark.txt
"""

import subprocess
import os
import sys
import shutil
from datetime import datetime

# Model configurations: (timestamp, use_whole_image)
MODELS = [
    ("20250812_152526", False),  # Batch 32, no same-area neg, bbox
    ("20251007_133126", False),  # Batch 32, same-area neg, bbox
    ("20251008_094017", False),  # Batch 16, same-area neg, bbox
    ("20251014_183603", False),  # Batch 16, no same-area neg, bbox
    ("20251015_165008", True),   # Batch 16, same-area neg, whole image
    ("20251016_133229", True),   # Batch 16, no same-area neg, whole image
]

# Areas to benchmark
AREAS = ["37", "38", "39", "40"]
YEARS = ["2022", "2023"]

EXTRACTION_SCRIPT = "extract_dinov2_finetuned_features.py"
COMPARISON_SCRIPT = "compare_dinov2_features.py"
BACKUP_SCRIPT = "extract_dinov2_finetuned_features.py.backup"


def backup_extraction_script():
    """Backup the original extraction script"""
    if not os.path.exists(BACKUP_SCRIPT):
        print(f"Creating backup of {EXTRACTION_SCRIPT}...")
        shutil.copy(EXTRACTION_SCRIPT, BACKUP_SCRIPT)
        print(f"Backup created: {BACKUP_SCRIPT}")


def restore_extraction_script():
    """Restore the original extraction script"""
    if os.path.exists(BACKUP_SCRIPT):
        print(f"Restoring original {EXTRACTION_SCRIPT}...")
        shutil.copy(BACKUP_SCRIPT, EXTRACTION_SCRIPT)
        print("Restoration complete.")


def update_model_path_in_script(model_timestamp):
    """Update the model path in extract_dinov2_finetuned_features.py"""
    model_filename = f"dinov2_coral_finetuned_final_{model_timestamp}.pt"

    # Check if model file exists
    if not os.path.exists(model_filename):
        print(f"ERROR: Model file not found: {model_filename}")
        return False

    print(f"Updating {EXTRACTION_SCRIPT} to use model: {model_filename}")

    # Read the extraction script
    with open(EXTRACTION_SCRIPT, 'r') as f:
        content = f.read()

    # Replace the model path in load_finetuned_dinov2_model function
    # Find the default parameter value
    old_pattern = 'model_path="dinov2_coral_finetuned_final_'

    # Find where the pattern occurs and replace the entire default value
    lines = content.split('\n')
    updated_lines = []
    for line in lines:
        if old_pattern in line and 'def load_finetuned_dinov2_model' in content[max(0, content.index(line)-200):content.index(line)+200]:
            # Replace the default model path
            import re
            line = re.sub(
                r'model_path="dinov2_coral_finetuned_final_\d+_\d+\.pt"',
                f'model_path="{model_filename}"',
                line
            )
        updated_lines.append(line)

    # Also update the model path at line 270
    updated_lines_final = []
    for line in updated_lines:
        if 'model = load_finetuned_dinov2_model("dinov2_coral_finetuned_final_' in line:
            import re
            line = re.sub(
                r'load_finetuned_dinov2_model\("dinov2_coral_finetuned_final_\d+_\d+\.pt"\)',
                f'load_finetuned_dinov2_model("{model_filename}")',
                line
            )
        updated_lines_final.append(line)

    # Write back
    with open(EXTRACTION_SCRIPT, 'w') as f:
        f.write('\n'.join(updated_lines_final))

    print(f"Updated {EXTRACTION_SCRIPT} to use {model_filename}")
    return True


def run_feature_extraction(model_timestamp, use_whole_image):
    """Run feature extraction for a specific model"""
    print("\n" + "="*80)
    print(f"EXTRACTING FEATURES FOR MODEL: {model_timestamp}")
    print(f"Mode: {'Whole Image' if use_whole_image else 'BBox Cropping'}")
    print("="*80 + "\n")

    # Build command
    cmd = ["python", EXTRACTION_SCRIPT]
    if use_whole_image:
        cmd.append("--use_whole_image")

    print(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR during feature extraction: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def run_comparison(ref_file, query_file, output_file):
    """Run comparison between two feature files and append to output file"""
    print(f"\nComparing: {ref_file} vs {query_file}")

    if not os.path.exists(ref_file):
        print(f"WARNING: Reference file not found: {ref_file}")
        return False
    if not os.path.exists(query_file):
        print(f"WARNING: Query file not found: {query_file}")
        return False

    cmd = ["python", COMPARISON_SCRIPT, ref_file, query_file]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Append to output file
        with open(output_file, 'a') as f:
            f.write("\n" + "="*80 + "\n")
            f.write(f"Comparison: {os.path.basename(ref_file)} vs {os.path.basename(query_file)}\n")
            f.write("="*80 + "\n")
            f.write(result.stdout)
            if result.stderr:
                f.write("\nSTDERR:\n")
                f.write(result.stderr)

        print(f"Results appended to {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR during comparison: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def benchmark_model(model_timestamp, use_whole_image):
    """Benchmark a single model"""
    print("\n" + "#"*80)
    print(f"# BENCHMARKING MODEL: {model_timestamp}")
    print(f"# Mode: {'Whole Image' if use_whole_image else 'BBox Cropping'}")
    print("#"*80 + "\n")

    # Update extraction script to use this model
    if not update_model_path_in_script(model_timestamp):
        print(f"Skipping model {model_timestamp} due to missing model file")
        return False

    # Run feature extraction
    if not run_feature_extraction(model_timestamp, use_whole_image):
        print(f"Feature extraction failed for model {model_timestamp}")
        return False

    # Prepare output file
    output_file = f"{model_timestamp}_nearest_benchmark.txt"

    # Write header
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"BENCHMARK RESULTS FOR MODEL: {model_timestamp}\n")
        f.write(f"Mode: {'Whole Image' if use_whole_image else 'BBox Cropping'}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")

    # Determine feature file prefix based on mode
    if use_whole_image:
        feature_prefix = "dinov2_finetuned_{year}_{area}_whole_features.h5"
    else:
        feature_prefix = "dinov2_finetuned_{year}_{area}_cropped_features.h5"

    # Run comparisons for all areas
    success_count = 0
    for area in AREAS:
        ref_file = feature_prefix.format(year="2022", area=area)
        query_file = feature_prefix.format(year="2023", area=area)

        if run_comparison(ref_file, query_file, output_file):
            success_count += 1

    print(f"\nCompleted {success_count}/{len(AREAS)} comparisons for model {model_timestamp}")
    print(f"Results saved to: {output_file}")

    return True


def cleanup_feature_files(use_whole_image):
    """Clean up generated feature files to save space"""
    if use_whole_image:
        pattern = "dinov2_finetuned_*_whole_features.h5"
    else:
        pattern = "dinov2_finetuned_*_cropped_features.h5"

    print(f"\nCleaning up feature files matching: {pattern}")

    import glob
    files = glob.glob(pattern)
    for f in files:
        try:
            os.remove(f)
            print(f"Deleted: {f}")
        except Exception as e:
            print(f"Failed to delete {f}: {e}")


def main():
    print("="*80)
    print("AUTOMATED BENCHMARKING SCRIPT FOR 6 DINOV2 MODELS")
    print("="*80)
    print(f"\nModels to benchmark: {len(MODELS)}")
    for timestamp, use_whole in MODELS:
        mode = "Whole Image" if use_whole else "BBox"
        print(f"  - {timestamp} ({mode})")
    print(f"\nAreas to compare: {AREAS}")
    print(f"Years: {YEARS[0]} vs {YEARS[1]}")
    print("\n" + "="*80 + "\n")

    # Check if scripts exist
    if not os.path.exists(EXTRACTION_SCRIPT):
        print(f"ERROR: {EXTRACTION_SCRIPT} not found!")
        return 1

    if not os.path.exists(COMPARISON_SCRIPT):
        print(f"ERROR: {COMPARISON_SCRIPT} not found!")
        return 1

    # Backup original extraction script
    backup_extraction_script()

    try:
        # Benchmark each model
        for i, (model_timestamp, use_whole_image) in enumerate(MODELS, 1):
            print(f"\n\n{'#'*80}")
            print(f"# PROCESSING MODEL {i}/{len(MODELS)}: {model_timestamp}")
            print(f"{'#'*80}\n")

            success = benchmark_model(model_timestamp, use_whole_image)

            if success:
                print(f"\n✓ Model {model_timestamp} benchmarking completed successfully!")

                # Optional: Clean up feature files after each model to save space
                # Uncomment the line below if you want to delete h5 files after each model
                # cleanup_feature_files(use_whole_image)
            else:
                print(f"\n✗ Model {model_timestamp} benchmarking failed!")

            print(f"\nProgress: {i}/{len(MODELS)} models completed")

    finally:
        # Always restore the original script
        restore_extraction_script()

    print("\n" + "="*80)
    print("BENCHMARKING COMPLETE!")
    print("="*80)
    print("\nGenerated benchmark files:")
    for timestamp, _ in MODELS:
        output_file = f"{timestamp}_nearest_benchmark.txt"
        if os.path.exists(output_file):
            print(f"  ✓ {output_file}")
        else:
            print(f"  ✗ {output_file} (not found)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
