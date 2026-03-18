"""Data loading and preprocessing classes"""
import os
import glob
import pickle
from typing import List, Tuple
from dataclasses import dataclass

import cv2
import numpy as np

from config import ExperimentConfig


class DataLoader:
    """Handles loading and preprocessing of image data"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
    def load_images_and_labels(self) -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
        """
        Load images from directory structure: data_dir/<class>/*.jpg
        
        Returns:
            images: List of BGR images (resized)
            labels: Array of integer labels
            class_names: List of class names
        """
        images = []
        labels = []
        
        class_names = sorted(os.listdir(self.config.data_dir))
        class_to_idx = {c: i for i, c in enumerate(class_names)}
        
        print(f"Loading images from: {self.config.data_dir}")
        print(f"Found classes: {class_names}")
        
        for class_name in class_names:
            class_dir = os.path.join(self.config.data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            image_files = glob.glob(os.path.join(class_dir, "*"))
            
            for img_path in image_files:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not read {img_path}")
                    continue
                
                # Resize image to limit computation
                img_resized = self._resize_image(img)
                images.append(img_resized)
                labels.append(class_to_idx[class_name])
        
        print(f"Loaded {len(images)} images from {len(class_names)} classes")
        return images, np.array(labels), class_names
    
    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        """Resize image maintaining aspect ratio"""
        h, w = img.shape[:2]
        max_dim = max(h, w)
        
        if max_dim > self.config.image_max_size:
            scale = self.config.image_max_size / max_dim
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return img


@dataclass
class PreprocessedData:
    """Container for preprocessed data that can be reused across experiments"""
    images: List[np.ndarray]
    labels: np.ndarray
    class_names: List[str]
    image_descriptors: List[np.ndarray]
    all_descriptors: np.ndarray
    
    def save(self, filepath: str):
        """Save preprocessed data to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Preprocessed data saved to: {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'PreprocessedData':
        """Load preprocessed data from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"Preprocessed data loaded from: {filepath}")
        return data

