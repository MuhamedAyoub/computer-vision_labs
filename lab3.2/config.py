"""Configuration classes for BoVW experiments"""
from typing import Dict
from dataclasses import dataclass, asdict


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    # Data parameters
    data_dir: str = "flowers"
    image_max_size: int = 400
    test_size: float = 0.2
    random_state: int = 42
    
    # Feature extraction parameters
    sift_n_features: int = 0  # 0 = unlimited
    sift_contrast_threshold: float = 0.04
    
    # Vocabulary parameters
    vocab_size: int = 150
    use_minibatch_kmeans: bool = True
    kmeans_batch_size: int = 1000
    
    # Classifier parameters
    classifier_type: str = 'knn'  # 'knn' or 'svm'
    
    # KNN-specific parameters
    knn_neighbors: int = 5
    knn_metric: str = 'euclidean'
    knn_weights: str = 'uniform'  # 'uniform' or 'distance'
    
    # SVM-specific parameters
    svm_kernel: str = 'rbf'  # 'linear', 'poly', 'rbf', 'sigmoid'
    svm_C: float = 1.0
    svm_gamma: str = 'scale'  # 'scale', 'auto', or float value
    svm_degree: int = 3  # for polynomial kernel
    
    # Feature normalization
    normalize_histograms: bool = True
    use_feature_scaling: bool = False  # StandardScaler before classification
    
    # Experiment metadata
    experiment_name: str = "baseline"
    experiment_description: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)

