"""Vocabulary building classes"""
from typing import List

import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans

from config import ExperimentConfig


class VocabularyBuilder:
    """Builds visual vocabulary using K-Means clustering"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.kmeans = None
        self.vocabulary = None
    
    def build_vocabulary(self, descriptors: np.ndarray) -> np.ndarray:
        """
        Build visual vocabulary by clustering descriptors
        
        Args:
            descriptors: All SIFT descriptors (M x 128)
        
        Returns:
            vocabulary: Cluster centers (vocab_size x 128)
        """
        print(f"Building vocabulary with {self.config.vocab_size} visual words...")
        
        if self.config.use_minibatch_kmeans:
            self.kmeans = MiniBatchKMeans(
                n_clusters=self.config.vocab_size,
                random_state=self.config.random_state,
                batch_size=self.config.kmeans_batch_size,
                verbose=0
            )
        else:
            self.kmeans = KMeans(
                n_clusters=self.config.vocab_size,
                random_state=self.config.random_state,
                verbose=0
            )
        
        self.kmeans.fit(descriptors)
        self.vocabulary = self.kmeans.cluster_centers_
        
        print(f"Vocabulary built. Shape: {self.vocabulary.shape}")
        return self.vocabulary
    
    def compute_histograms(self, image_descriptors: List[np.ndarray]) -> np.ndarray:
        """
        Compute histogram of visual words for each image
        
        Args:
            image_descriptors: List of descriptor arrays per image
        
        Returns:
            histograms: Array of shape (n_images, vocab_size)
        """
        n_images = len(image_descriptors)
        histograms = np.zeros((n_images, self.config.vocab_size), dtype=np.float32)
        
        print(f"Computing BoVW histograms for {n_images} images...")
        for i, descriptors in enumerate(image_descriptors):
            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{n_images} histograms")
            
            if descriptors.shape[0] == 0:
                # No features detected in this image
                continue
            
            visual_words = self.kmeans.predict(descriptors)
            
            hist, _ = np.histogram(
                visual_words,
                bins=np.arange(self.config.vocab_size + 1)
            )
            
            # Normalize histogram (L2 normalization)
            if self.config.normalize_histograms and hist.sum() > 0:
                hist = hist.astype(np.float32)
                hist = hist / np.linalg.norm(hist)
            
            histograms[i] = hist
        
        print(f"Histograms computed. Shape: {histograms.shape}")
        return histograms

