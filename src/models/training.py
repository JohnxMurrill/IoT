import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles model training, hyperparameter tuning, and evaluation"""
    
    def __init__(self, models=None, output_dir=None):
        """
        Initialize the model trainer
        
        Args:
            models (dict): Dictionary of model name to model instance
            output_dir (Path): Directory to save models and results
        """
        self.models = models or {}
        self.output_dir = Path(output_dir) if output_dir else Path("models")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        
    def add_model(self, name, model):
        """Add a model to the trainer"""
        self.models[name] = model
        
    def train_all(self, X_train, y_train):
        """Train all registered models"""
        for name, model in self.models.items():
            logger.info(f"Training model: {name}")
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Save trained model
            model_path = self.output_dir / f"{name}.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Model saved to {model_path}")
            
            self.results[name] = {'training_time': training_time}
    
    def tune_hyperparameters(self, model_name, param_grid, X_train, y_train, cv=5, scoring='accuracy'):
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Args:
            model_name (str): Name of the model to tune
            param_grid (dict): Hyperparameter grid to search
            X_train: Training features
            y_train: Training labels
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric
            
        Returns:
            Best estimator found
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
            
        logger.info(f"Tuning hyperparameters for {model_name}")
        model = self.models[model_name]
        
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")
        
        # Update the model with the best estimator
        self.models[model_name] = grid_search.best_estimator_
        
        # Save tuning results
        self.results[model_name]['tuning'] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        return grid_search.best_estimator_
    
    def evaluate_all(self, X_test, y_test):
        """Evaluate all models on test data"""
        evaluation_results = {}
        
        for name, model in self.models.items():
            logger.info(f"Evaluating model: {name}")
            
            start_time = time.time()
            
            if hasattr(model, 'evaluate'):
                eval_result = model.evaluate(X_test, y_test)
            else:
                # Fall back to manual evaluation
                y_pred = model.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred)
                conf_matrix = confusion_matrix(y_test, y_pred)
                
                eval_result = {
                    'accuracy': accuracy,
                    'report': report,
                    'confusion_matrix': conf_matrix
                }
                
                logger.info(f"Model {name} Accuracy: {accuracy:.4f}")
            
            inference_time = time.time() - start_time
            eval_result['inference_time'] = inference_time
            
            # Save evaluation results
            evaluation_results[name] = eval_result
            
            # Update the overall results dictionary
            if name in self.results:
                self.results[name]['evaluation'] = eval_result
            else:
                self.results[name] = {'evaluation': eval_result}
                
            # Plot confusion matrix
            self._plot_confusion_matrix(
                eval_result['confusion_matrix'], 
                title=f"Confusion Matrix - {name}",
                save_path=self.output_dir / f"{name}_confusion_matrix.png"
            )
        
        return evaluation_results
    
    def _plot_confusion_matrix(self, cm, title=None, save_path=None, class_names=None):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        
        plt.title(title or 'Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if class_names:
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks + 0.5, class_names, rotation=45)
            plt.yticks(tick_marks + 0.5, class_names, rotation=0)
        
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def summarize_results(self):
        """Generate a summary of results for all models"""
        if not self.results:
            logger.warning("No results available. Train and evaluate models first.")
            return None
            
        summary = {}
        
        for name, result in self.results.items():
            model_summary = {}
            
            # Extract training time if available
            if 'training_time' in result:
                model_summary['training_time'] = result['training_time']
                
            # Extract evaluation metrics if available
            if 'evaluation' in result:
                eval_result = result['evaluation']
                model_summary['accuracy'] = eval_result.get('accuracy', None)
                model_summary['inference_time'] = eval_result.get('inference_time', None)
                
            summary[name] = model_summary
            
        # Convert to DataFrame for easier comparison
        summary_df = pd.DataFrame(summary).T
        summary_df = summary_df.sort_values('accuracy', ascending=False)
        
        # Save summary to CSV
        summary_path = self.output_dir / "model_comparison.csv"
        summary_df.to_csv(summary_path)
        logger.info(f"Model comparison saved to {summary_path}")
        
        return summary_df