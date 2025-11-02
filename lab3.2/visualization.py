"""Visualization classes for experiment results"""
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


class ResultsVisualizer:
    """Visualizes experiment results"""
    
    @staticmethod
    def plot_confusion_matrix(metrics: Dict, class_names: List[str], 
                             experiment_name: str = "", figsize=(10, 8)):
        """Plot confusion matrix heatmap"""
        cm = metrics['confusion_matrix']
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        
        plt.title(f'Confusion Matrix\n{experiment_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_metrics_comparison(results_df: pd.DataFrame, figsize=(12, 6)):
        """Plot comparison of metrics across experiments"""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            results_df.plot(x='experiment_name', y=metric, 
                          kind='bar', ax=ax, legend=False)
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_xlabel('')
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_parameter_impact(results_df: pd.DataFrame, param_name: str, 
                             metric: str = 'accuracy', figsize=(10, 6)):
        """Plot impact of a specific parameter on performance"""
        plt.figure(figsize=figsize)
        
        grouped = results_df.groupby(param_name)[metric].agg(['mean', 'std'])
        
        plt.errorbar(grouped.index, grouped['mean'], 
                    yerr=grouped['std'], marker='o', capsize=5, markersize=8, linewidth=2)
        
        plt.title(f'Impact of {param_name} on {metric}')
        plt.xlabel(param_name)
        plt.ylabel(metric.replace('_', ' ').title())
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_best_vs_worst(results_df: pd.DataFrame, metric: str = 'accuracy',
                          n_experiments: int = 5, figsize=(12, 6)):
        """Compare best and worst performing experiments"""
        sorted_df = results_df.sort_values(metric, ascending=False)
        
        best_df = sorted_df.head(n_experiments)
        worst_df = sorted_df.tail(n_experiments)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Best experiments
        best_df.plot(x='experiment_name', y=metric, kind='barh', 
                    ax=ax1, legend=False, color='green')
        ax1.set_title(f'Top {n_experiments} Experiments')
        ax1.set_xlabel(metric.title())
        
        # Worst experiments
        worst_df.plot(x='experiment_name', y=metric, kind='barh', 
                     ax=ax2, legend=False, color='red')
        ax2.set_title(f'Bottom {n_experiments} Experiments')
        ax2.set_xlabel(metric.title())
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_classifier_comparison(results_df: pd.DataFrame, figsize=(14, 5)):
        """Compare KNN vs SVM performance across different metrics"""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Group by classifier type
        classifier_stats = results_df.groupby('classifier_type')[metrics].agg(['mean', 'std'])
        
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Get mean and std for each classifier
            means = classifier_stats[metric]['mean']
            stds = classifier_stats[metric]['std']
            
            x = np.arange(len(means))
            ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7,
                  color=['#3498db', '#e74c3c'])
            ax.set_xticks(x)
            ax.set_xticklabels(means.index, rotation=0)
            ax.set_ylabel('Score')
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('KNN vs SVM - Performance Comparison', y=1.02, fontsize=14, fontweight='bold')
        plt.show()
    
    @staticmethod
    def plot_svm_kernel_comparison(results_df: pd.DataFrame, figsize=(10, 6)):
        """Compare different SVM kernels"""
        svm_results = results_df[results_df['classifier_type'] == 'svm']
        
        if len(svm_results) == 0:
            print("No SVM results found in DataFrame")
            return
        
        kernel_stats = svm_results.groupby('svm_kernel').agg({
            'accuracy': ['mean', 'std', 'count'],
            'f1_score': ['mean', 'std']
        })
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Accuracy comparison
        kernels = kernel_stats.index
        acc_means = kernel_stats['accuracy']['mean']
        acc_stds = kernel_stats['accuracy']['std']
        
        ax1.bar(kernels, acc_means, yerr=acc_stds, capsize=5, alpha=0.7)
        ax1.set_ylabel('Accuracy')
        ax1.set_title('SVM Kernel Comparison - Accuracy')
        ax1.grid(axis='y', alpha=0.3)
        
        # F1-Score comparison
        f1_means = kernel_stats['f1_score']['mean']
        f1_stds = kernel_stats['f1_score']['std']
        
        ax2.bar(kernels, f1_means, yerr=f1_stds, capsize=5, alpha=0.7, color='orange')
        ax2.set_ylabel('F1-Score')
        ax2.set_title('SVM Kernel Comparison - F1-Score')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_training_time_comparison(results_df: pd.DataFrame, figsize=(10, 6)):
        """Compare training time across different configurations"""
        plt.figure(figsize=figsize)
        
        # Group by classifier type
        time_stats = results_df.groupby('classifier_type')['duration_seconds'].agg(['mean', 'std'])
        
        x = np.arange(len(time_stats))
        plt.bar(x, time_stats['mean'], yerr=time_stats['std'], 
               capsize=5, alpha=0.7, color=['#3498db', '#e74c3c'])
        plt.xticks(x, time_stats.index)
        plt.ylabel('Duration (seconds)')
        plt.title('Training Time Comparison')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

