"""Experiment management and persistence classes"""
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from data import PreprocessedData


class ExperimentManager:
    """Manages saving and loading experiment results"""
    
    def __init__(self, output_dir: str = "experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def save_results(self, results: Dict, filename: str = None):
        """Save experiment results to JSON"""
        if filename is None:
            filename = f"{results['config']['experiment_name']}.json"
        
        filepath = self.output_dir / filename
        
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = self._make_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"Results saved to: {filepath}")
    
    def save_results_dataframe(self, df: pd.DataFrame, filename: str = "all_results.csv"):
        """Save results DataFrame to CSV"""
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        print(f"Results DataFrame saved to: {filepath}")
    
    def save_preprocessed_data(self, data: PreprocessedData, filename: str = "preprocessed_data.pkl"):
        """Save preprocessed data for reuse"""
        filepath = self.output_dir / filename
        data.save(str(filepath))
    
    def load_preprocessed_data(self, filename: str = "preprocessed_data.pkl") -> PreprocessedData:
        """Load preprocessed data"""
        filepath = self.output_dir / filename
        return PreprocessedData.load(str(filepath))
    
    def load_results(self, filename: str) -> Dict:
        """Load experiment results from JSON"""
        filepath = self.output_dir / filename
        
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        return results
    
    def _make_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj

