"""
Merge public Roboflow dataset with your custom annotated data
For use BEFORE uploading to your main project
"""

import os
import shutil
from pathlib import Path
import yaml

def merge_yolo_datasets(custom_dir, public_dir, output_dir):
    """
    Merge two YOLO format datasets
    
    Args:
        custom_dir: Path to your custom dataset (e.g., "./custom_data")
        public_dir: Path to downloaded public dataset (e.g., "./corrosion-dataset-1")
        output_dir: Where to save merged dataset (e.g., "./merged_dataset")
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create standard YOLO structure
    for split in ['train', 'valid', 'test']:
        os.makedirs(f"{output_dir}/{split}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/{split}/labels", exist_ok=True)
    
    # Copy custom data
    print("📋 Copying custom data...")
    copy_dataset_split(custom_dir, output_dir, prefix="custom")
    
    # Copy public data
    print("📋 Copying public data...")
    copy_dataset_split(public_dir, output_dir, prefix="public")
    
    # Update data.yaml
    create_merged_yaml(output_dir, custom_dir, public_dir)
    
    print("✅ Datasets merged successfully!")
    print(f"   Output: {output_dir}")
    

def copy_dataset_split(source_dir, dest_dir, prefix=""):
    """Copy images and labels from source to destination with unique naming"""
    
    for split in ['train', 'valid', 'test']:
        src_img_dir = Path(source_dir) / split / 'images'
        src_lbl_dir = Path(source_dir) / split / 'labels'
        
        if not src_img_dir.exists():
            print(f"   ⚠️ Skipping {split} (not found in {source_dir})")
            continue
        
        dest_img_dir = Path(dest_dir) / split / 'images'
        dest_lbl_dir = Path(dest_dir) / split / 'labels'
        
        # Copy images
        for img_file in src_img_dir.glob('*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                new_name = f"{prefix}_{img_file.name}" if prefix else img_file.name
                shutil.copy(img_file, dest_img_dir / new_name)
        
        # Copy labels
        if src_lbl_dir.exists():
            for lbl_file in src_lbl_dir.glob('*.txt'):
                new_name = f"{prefix}_{lbl_file.name}" if prefix else lbl_file.name
                shutil.copy(lbl_file, dest_lbl_dir / new_name)
        
        img_count = len(list(dest_img_dir.glob('*')))
        print(f"   ✓ {split}: {img_count} images")


def create_merged_yaml(output_dir, custom_dir, public_dir):
    """Create data.yaml for merged dataset"""
    
    yaml_content = {
        'path': output_dir,
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 1,  # Number of classes (1 for corrosion detection)
        'names': ['corrosion']
    }
    
    yaml_path = Path(output_dir) / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"   ✓ Created: {yaml_path}")


# Example usage
if __name__ == "__main__":
    merge_yolo_datasets(
        custom_dir="./my_custom_dataset",  # Your annotated data
        public_dir="./corrosion-dataset-1",  # Downloaded public dataset
        output_dir="./merged_corrosion_dataset"
    )
    
    # Now upload the merged dataset to Roboflow!
    print("\n📤 Next step: Upload merged_corrosion_dataset to Roboflow")