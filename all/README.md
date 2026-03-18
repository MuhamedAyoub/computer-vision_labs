# Bag of Visual Words (BoVW) - Image Classification Framework

A modular Python framework for image classification using Bag of Visual Words with SIFT features.



## Module Overview

### `config.py`
Contains `ExperimentConfig` dataclass that defines all experiment parameters including:
- Data loading parameters
- Feature extraction settings (SIFT)
- Vocabulary building parameters
- Classifier configuration (KNN/SVM)
- Normalization options

### `data.py`
- **DataLoader**: Loads images from directory structure and resizes them
- **PreprocessedData**: Container for preprocessed data that can be cached and reused

### `features.py`
- **FeatureExtractor**: Extracts SIFT descriptors from images using OpenCV

### `vocabulary.py`
- **VocabularyBuilder**: Builds visual vocabulary using K-Means clustering and computes BoVW histograms

### `classifier.py`
- **Classifier**: Trains and evaluates KNN or SVM classifiers with optional feature scaling

### `experiment.py`
- **ExperimentRunner**: Executes the complete experiment pipeline
- **HyperparameterTuner**: Performs grid search over parameter combinations

### `visualization.py`
- **ResultsVisualizer**: Provides various plotting utilities for:
  - Confusion matrices
  - Metrics comparison
  - Parameter impact analysis
  - Classifier comparison
  - Training time comparison

### `manager.py`
- **ExperimentManager**: Handles saving and loading of:
  - Experiment results (JSON)
  - Results DataFrames (CSV)
  - Preprocessed data (pickle)

## Usage

### Running a Grid Search Experiment

```python
from config import ExperimentConfig
from experiment import HyperparameterTuner
from visualization import ResultsVisualizer
from manager import ExperimentManager

# Define base configuration
base_config = ExperimentConfig(
    data_dir="UCMerced_LandUse",
    experiment_description="Comprehensive grid search: KNN vs SVM"
)

# Define parameter grid
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

# Visualize results
visualizer = ResultsVisualizer()
visualizer.plot_classifier_comparison(results_df)

# Save results
manager = ExperimentManager()
manager.save_results_dataframe(results_df)
```

### Running a Single Experiment

```python
from config import ExperimentConfig
from experiment import ExperimentRunner

config = ExperimentConfig(
    data_dir="flowers",
    vocab_size=150,
    classifier_type='knn',
    knn_neighbors=5
)

runner = ExperimentRunner(config)
results = runner.run()
```

## Dependencies

- numpy
- opencv-python (cv2)
- scikit-learn
- pandas
- matplotlib
- seaborn

## Data Directory Structure

Expected structure:
```
data_dir/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   └── ...
└── ...
```

