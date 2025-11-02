"""Classification classes"""
from typing import Dict, List, Any

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score
)

from config import ExperimentConfig


class Classifier:
    """Handles classification using KNN or SVM"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model = None
        self.scaler = None
        self._create_model()
    
    def _create_model(self):
        """Create the appropriate classifier based on config"""
        if self.config.classifier_type.lower() == 'knn':
            self.model = KNeighborsClassifier(
                n_neighbors=self.config.knn_neighbors,
                metric=self.config.knn_metric,
                weights=self.config.knn_weights
            )
            print(f"Created KNN classifier (k={self.config.knn_neighbors}, "
                  f"metric={self.config.knn_metric}, weights={self.config.knn_weights})")
        
        elif self.config.classifier_type.lower() == 'svm':
            self.model = SVC(
                kernel=self.config.svm_kernel,
                C=self.config.svm_C,
                gamma=self.config.svm_gamma,
                degree=self.config.svm_degree,
                random_state=self.config.random_state
            )
            print(f"Created SVM classifier (kernel={self.config.svm_kernel}, "
                  f"C={self.config.svm_C}, gamma={self.config.svm_gamma})")
        
        else:
            raise ValueError(f"Unknown classifier type: {self.config.classifier_type}. "
                           "Use 'knn' or 'svm'")
        
        # Create scaler if needed
        if self.config.use_feature_scaling:
            self.scaler = StandardScaler()
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train classifier with optional feature scaling"""
        print(f"Training {self.config.classifier_type.upper()} classifier...")
        print(f"  Training samples: {X_train.shape[0]}")
        print(f"  Feature dimension: {X_train.shape[1]}")
        
        # Apply feature scaling if enabled
        if self.scaler is not None:
            print("  Applying StandardScaler to features...")
            X_train = self.scaler.fit_transform(X_train)
        
        self.model.fit(X_train, y_train)
        print("Training complete!")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with optional scaling"""
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, 
                 class_names: List[str]) -> Dict[str, Any]:
        """
        Evaluate model and return metrics
        
        Returns:
            metrics: Dictionary containing various performance metrics
        """
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(
                y_test, y_pred, 
                target_names=class_names,
                output_dict=True,
                zero_division=0
            )
        }
        
        return metrics

