"""Feature extraction classes"""
from typing import List, Tuple

import cv2
import numpy as np

from config import ExperimentConfig


class FeatureExtractor:
    """Extracts SIFT features from images"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.sift = cv2.SIFT_create(
            nfeatures=config.sift_n_features,
            contrastThreshold=config.sift_contrast_threshold
        )
    
    def extract_descriptors(self, images: List[np.ndarray]) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Extract SIFT descriptors from all images
        
        Returns:
            image_descriptors: List of descriptor arrays per image (N_i x 128)
            all_descriptors_stacked: All descriptors concatenated (M x 128)
                where N_i = number of keypoints in image i
                      M = total keypoints across all images
        """
        image_descriptors = []
        all_descriptors = []
        
        print("Extracting SIFT descriptors...")
        for i, img in enumerate(images):
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(images)} images")
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.sift.detectAndCompute(gray, None)
            
            # Handle images with no keypoints
            if descriptors is None:
                descriptors = np.zeros((0, 128), dtype=np.float32)
            
            image_descriptors.append(descriptors)
            
            if descriptors.shape[0] > 0:
                all_descriptors.append(descriptors)
        
        if len(all_descriptors) == 0:
            raise RuntimeError("No descriptors found in any image")
        
        all_descriptors_stacked = np.vstack(all_descriptors).astype(np.float32)
        print(f"Total descriptors extracted: {all_descriptors_stacked.shape[0]}")
        print(f"Descriptors per image: min={min(len(d) for d in image_descriptors)}, "
              f"max={max(len(d) for d in image_descriptors)}, "
              f"mean={np.mean([len(d) for d in image_descriptors]):.1f}")
        
        return image_descriptors, all_descriptors_stacked

