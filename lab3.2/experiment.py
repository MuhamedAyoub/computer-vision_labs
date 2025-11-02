"""Experiment execution classes"""
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import asdict
from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import ExperimentConfig
from data import DataLoader, PreprocessedData
from features import FeatureExtractor
from vocabulary import VocabularyBuilder
from classifier import Classifier


class ExperimentRunner:
    """Runs complete experiment pipeline"""
    
    def __init__(self, config: ExperimentConfig, preprocessed_data: Optional[PreprocessedData] = None):
        self.config = config
        self.preprocessed_data = preprocessed_data
        self.results = {}
        
    def run(self, force_preprocess: bool = False) -> Dict[str, Any]:
        """
        Execute complete experiment pipeline
        
        Args:
            force_preprocess: If True, reload and reprocess images even if preprocessed_data exists
        """
        start_time = datetime.now()
        print(f"\n{'='*60}")
        print(f"Running Experiment: {self.config.experiment_name}")
        print(f"{'='*60}\n")
        
        # 1-2. Load data and extract features (only if not already done)
        if self.preprocessed_data is None or force_preprocess:
            print(">>> STEP 1: Loading images and extracting features")
            data_loader = DataLoader(self.config)
            images, labels, class_names = data_loader.load_images_and_labels()
            
            feature_extractor = FeatureExtractor(self.config)
            image_descriptors, all_descriptors = feature_extractor.extract_descriptors(images)
            
            self.preprocessed_data = PreprocessedData(
                images=images,
                labels=labels,
                class_names=class_names,
                image_descriptors=image_descriptors,
                all_descriptors=all_descriptors
            )
        else:
            print(">>> Using preprocessed data (skipping image loading and feature extraction)")
            images = self.preprocessed_data.images
            labels = self.preprocessed_data.labels
            class_names = self.preprocessed_data.class_names
            image_descriptors = self.preprocessed_data.image_descriptors
            all_descriptors = self.preprocessed_data.all_descriptors
        
        # 3. Build vocabulary
        print(f"\n>>> STEP 2: Building vocabulary (vocab_size={self.config.vocab_size})")
        vocab_builder = VocabularyBuilder(self.config)
        vocabulary = vocab_builder.build_vocabulary(all_descriptors)
        
        # 4. Compute histograms
        print(f"\n>>> STEP 3: Computing BoVW histograms")
        histograms = vocab_builder.compute_histograms(image_descriptors)
        
        # CRITICAL: Verify shapes match
        print(f"\n>>> Verifying data consistency:")
        print(f"  Images: {len(images)}")
        print(f"  Labels: {labels.shape[0]}")
        print(f"  Histograms: {histograms.shape[0]}")
        
        if histograms.shape[0] != labels.shape[0]:
            raise ValueError(f"Shape mismatch! Histograms: {histograms.shape[0]}, Labels: {labels.shape[0]}")
        
        # 5. Train/test split
        print(f"\n>>> STEP 4: Splitting data")
        X_train, X_test, y_train, y_test = train_test_split(
            histograms, labels,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=labels
        )
        
        print(f"  Train set: {X_train.shape[0]} samples")
        print(f"  Test set: {X_test.shape[0]} samples")
        
        # 6. Train classifier
        print(f"\n>>> STEP 5: Training classifier")
        classifier = Classifier(self.config)
        classifier.train(X_train, y_train)
        
        # 7. Evaluate
        print(f"\n>>> STEP 6: Evaluating model")
        metrics = classifier.evaluate(X_test, y_test, class_names)
        
        # Store results
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        self.results = {
            'config': self.config.to_dict(),
            'metrics': metrics,
            'class_names': class_names,
            'timestamp': start_time.isoformat(),
            'duration_seconds': duration,
            'data_stats': {
                'n_images': len(images),
                'n_classes': len(class_names),
                'n_train': X_train.shape[0],
                'n_test': X_test.shape[0],
                'n_descriptors': all_descriptors.shape[0]
            }
        }
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _print_summary(self):
        """Print experiment summary"""
        print(f"\n{'='*60}")
        print("EXPERIMENT RESULTS")
        print(f"{'='*60}")
        print(f"Experiment: {self.config.experiment_name}")
        print(f"Classifier: {self.config.classifier_type.upper()}")
        print(f"Duration: {self.results['duration_seconds']:.2f} seconds")
        print(f"\nMetrics:")
        print(f"  Accuracy:  {self.results['metrics']['accuracy']:.4f}")
        print(f"  Precision: {self.results['metrics']['precision']:.4f}")
        print(f"  Recall:    {self.results['metrics']['recall']:.4f}")
        print(f"  F1-Score:  {self.results['metrics']['f1_score']:.4f}")
        print(f"{'='*60}\n")


class HyperparameterTuner:
    """Performs hyperparameter tuning using grid search"""
    
    def __init__(self, base_config: ExperimentConfig):
        self.base_config = base_config
        self.results = []
        self.preprocessed_data = None
    
    def tune(self, param_grid: Dict[str, List]) -> pd.DataFrame:
        """
        Run grid search over parameter combinations
        
        Args:
            param_grid: Dictionary mapping parameter names to lists of values
                Example: {
                    'vocab_size': [50, 100, 150],
                    'classifier_type': ['knn', 'svm'],
                    'knn_neighbors': [3, 5, 7],
                }
        
        Returns:
            results_df: DataFrame with all experiment results
        """
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        print(f"{'='*60}")
        print(f"STARTING GRID SEARCH")
        print(f"{'='*60}")
        print(f"Total combinations: {len(combinations)}")
        print(f"Parameters: {param_names}\n")
        
        # CRITICAL: Extract features ONLY ONCE before all experiments
        print(f"{'='*60}")
        print("PREPROCESSING DATA (ONE TIME ONLY)")
        print(f"{'='*60}\n")
        
        data_loader = DataLoader(self.base_config)
        images, labels, class_names = data_loader.load_images_and_labels()
        
        feature_extractor = FeatureExtractor(self.base_config)
        image_descriptors, all_descriptors = feature_extractor.extract_descriptors(images)
        
        self.preprocessed_data = PreprocessedData(
            images=images,
            labels=labels,
            class_names=class_names,
            image_descriptors=image_descriptors,
            all_descriptors=all_descriptors
        )
        
        print(f"\n{'='*60}")
        print("RUNNING EXPERIMENTS")
        print(f"{'='*60}\n")
        
        for i, param_combo in enumerate(combinations, 1):
            # Create config for this combination
            config = ExperimentConfig(**asdict(self.base_config))
            
            # Update parameters
            for param_name, param_value in zip(param_names, param_combo):
                setattr(config, param_name, param_value)
            
            # Set experiment name
            config.experiment_name = f"exp_{i:03d}_" + "_".join(
                f"{name}={value}" for name, value in zip(param_names, param_combo)
            )
            
            print(f"\n[{i}/{len(combinations)}] {config.experiment_name}")
            
            # Run experiment with preprocessed data
            runner = ExperimentRunner(config, preprocessed_data=self.preprocessed_data)
            results = runner.run()
            self.results.append(results)
        
        # Convert to DataFrame
        results_df = self._results_to_dataframe()
        
        print(f"\n{'='*60}")
        print("GRID SEARCH COMPLETE")
        print(f"{'='*60}\n")
        
        return results_df
    
    def _results_to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame"""
        records = []
        
        for result in self.results:
            record = {
                'experiment_name': result['config']['experiment_name'],
                'timestamp': result['timestamp'],
                'duration_seconds': result['duration_seconds'],
                
                # Configuration
                'classifier_type': result['config']['classifier_type'],
                'vocab_size': result['config']['vocab_size'],
                'image_max_size': result['config']['image_max_size'],
                'normalize_histograms': result['config']['normalize_histograms'],
                'use_feature_scaling': result['config']['use_feature_scaling'],
                'use_minibatch_kmeans': result['config']['use_minibatch_kmeans'],
                
                # KNN parameters
                'knn_neighbors': result['config']['knn_neighbors'],
                'knn_metric': result['config']['knn_metric'],
                'knn_weights': result['config']['knn_weights'],
                
                # SVM parameters
                'svm_kernel': result['config']['svm_kernel'],
                'svm_C': result['config']['svm_C'],
                'svm_gamma': result['config']['svm_gamma'],
                
                # Metrics
                'accuracy': result['metrics']['accuracy'],
                'precision': result['metrics']['precision'],
                'recall': result['metrics']['recall'],
                'f1_score': result['metrics']['f1_score'],
                
                # Data stats
                'n_images': result['data_stats']['n_images'],
                'n_descriptors': result['data_stats']['n_descriptors'],
            }
            records.append(record)
        
        return pd.DataFrame(records)

