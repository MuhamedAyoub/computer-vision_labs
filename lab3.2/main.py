"""
Bag of Visual Words (BoVW) - Image Classification Experiment Framework
Main entry point for running experiments and hyperparameter tuning.
"""

from config import ExperimentConfig
from experiment import HyperparameterTuner
from visualization import ResultsVisualizer
from manager import ExperimentManager


if __name__ == "__main__":
    # Define base configuration
    base_config = ExperimentConfig(
        data_dir="UCMerced_LandUse",
        experiment_description="Comprehensive grid search: KNN vs SVM"
    )
    
    param_grid = {
        'vocab_size': [50, 100, 150],
        'classifier_type': ['knn', 'svm'],
        'knn_neighbors': [3, 5, 7],  
        'svm_kernel': ['linear', 'rbf'],  
        'svm_C': [0.1, 1.0, 10.0],  
        'use_feature_scaling': [True, False]
    }
    
    # Run grid search
    tuner = HyperparameterTuner(base_config)
    results_df = tuner.tune(param_grid)
    
    # Display results
    print("\n" + "="*60)
    print("GRID SEARCH RESULTS - TOP 10 CONFIGURATIONS")
    print("="*60)
    top_results = results_df.sort_values('accuracy', ascending=False).head(10)
    print(top_results[['experiment_name', 'classifier_type', 'vocab_size', 
                       'accuracy', 'f1_score', 'duration_seconds']])
    
    # Compare KNN vs SVM
    print("\n" + "="*60)
    print("KNN vs SVM COMPARISON")
    print("="*60)
    comparison = results_df.groupby('classifier_type').agg({
        'accuracy': ['mean', 'std', 'max'],
        'f1_score': ['mean', 'std', 'max'],
        'duration_seconds': ['mean', 'std']
    }).round(4)
    print(comparison)
    
    visualizer = ResultsVisualizer()
    visualizer.plot_metrics_comparison(results_df.head(20))
    visualizer.plot_parameter_impact(results_df, 'vocab_size')
    visualizer.plot_classifier_comparison(results_df)
    visualizer.plot_training_time_comparison(results_df)
    
    best_knn = results_df[results_df['classifier_type'] == 'knn'].sort_values('accuracy', ascending=False).iloc[0]
    best_svm = results_df[results_df['classifier_type'] == 'svm'].sort_values('accuracy', ascending=False).iloc[0]
    
    print("\n" + "="*60)
    print("BEST KNN CONFIGURATION")
    print("="*60)
    print(f"Accuracy: {best_knn['accuracy']:.4f}")
    print(f"Vocab Size: {best_knn['vocab_size']}")
    print(f"K Neighbors: {best_knn['knn_neighbors']}")
    print(f"Metric: {best_knn['knn_metric']}")
    
    print("\n" + "="*60)
    print("BEST SVM CONFIGURATION")
    print("="*60)
    print(f"Accuracy: {best_svm['accuracy']:.4f}")
    print(f"Vocab Size: {best_svm['vocab_size']}")
    print(f"Kernel: {best_svm['svm_kernel']}")
    print(f"C: {best_svm['svm_C']}")
    
    # Save results
    manager = ExperimentManager()
    manager.save_results_dataframe(results_df)
    manager.save_preprocessed_data(tuner.preprocessed_data)
    
    # Get best overall configuration
    best_idx = results_df['accuracy'].idxmax()
    best_config = results_df.loc[best_idx]
    print("\n" + "="*60)
    print("BEST OVERALL CONFIGURATION")
    print("="*60)
    print(best_config)
