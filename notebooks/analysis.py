import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
import joblib

# Add project root to Python path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

def load_models(models_dir):
    """Load all saved models from directory"""
    models_dir = Path(models_dir)
    models = {}
    
    for model_path in models_dir.glob("*.joblib"):
        model_name = model_path.stem
        models[model_name] = joblib.load(model_path)
        
    return models

def plot_roc_curves(models, X_test, y_test, output_dir=None):
    """Plot ROC curves for all models (multi-class)"""
    # Get unique classes
    classes = np.unique(y_test)
    n_classes = len(classes)
    
    # Binarize labels for multi-class ROC
    y_bin = label_binarize(y_test, classes=classes)
    
    plt.figure(figsize=(12, 10))
    
    for model_name, model in models.items():
        # Check if model supports predict_proba
        if hasattr(model, 'predict_proba'):
            # Get probability scores
            try:
                y_score = model.predict_proba(X_test)
                
                # Calculate ROC curve and ROC area for each class
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                
                # Calculate micro-average ROC curve and ROC area
                fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_score.ravel())
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                
                # Plot ROC curve for micro-average
                plt.plot(
                    fpr["micro"], 
                    tpr["micro"],
                    label=f'{model_name} (AUC = {roc_auc["micro"]:.2f})',
                    linewidth=2
                )
            except (AttributeError, ValueError) as e:
                print(f"Could not calculate ROC for {model_name}: {e}")
        else:
            print(f"Model {model_name} does not support predict_proba")
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curves (Micro-Average)')
    plt.legend(loc="lower right")
    
    if output_dir:
        plt.savefig(Path(output_dir) / "roc_curves.png")
    else:
        plt.show()
    
    plt.close()

def plot_feature_importance(model, feature_names, top_n=20, output_dir=None):
    """Plot feature importance for models that support it"""
    if hasattr(model, 'feature_importance'):
        importances = model.feature_importance()
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).mean(axis=0)
    else:
        print("Model does not provide feature importance")
        return
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Take top N features
    if top_n and len(importance_df) > top_n:
        importance_df = importance_df.head(top_n)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance')
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(Path(output_dir) / "feature_importance.png")
    else:
        plt.show()
    
    plt.close()
    
    return importance_df

def plot_tsne_visualization(X, y, output_dir=None):
    """Create t-SNE visualization of data clusters"""
    # Reduce dimensionality with t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    # Create DataFrame for plotting
    tsne_df = pd.DataFrame({
        'tsne_1': X_tsne[:, 0],
        'tsne_2': X_tsne[:, 1],
        'label': y
    })
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x='tsne_1', y='tsne_2', 
        hue='label', 
        palette='viridis',
        data=tsne_df,
        alpha=0.7
    )
    plt.title('t-SNE Visualization of Data')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if output_dir:
        plt.savefig(Path(output_dir) / "tsne_visualization.png")
    else:
        plt.show()
    
    plt.close()

def analyze_misclassifications(model, X_test, y_test, feature_names):
    """Analyze patterns in misclassified instances"""
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Create DataFrame with features and predictions
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df['actual'] = y_test
    test_df['predicted'] = y_pred
    test_df['misclassified'] = (y_test != y_pred)
    
    # Summary of misclassifications by class
    misclass_summary = test_df[test_df['misclassified']].groupby(['actual', 'predicted']).size().reset_index()
    misclass_summary.columns = ['Actual', 'Predicted', 'Count']
    misclass_summary = misclass_summary.sort_values('Count', ascending=False)
    
    # Calculate mean feature values for misclassified vs correctly classified
    misclass_means = test_df[test_df['misclassified']].mean()
    correct_means = test_df[~test_df['misclassified']].mean()
    
    # Compare differences
    comparison = pd.DataFrame({
        'Misclassified_Mean': misclass_means,
        'Correct_Mean': correct_means,
        'Difference': misclass_means - correct_means
    })
    
    return {
        'misclass_summary': misclass_summary,
        'feature_comparison': comparison.sort_values('Difference', key=abs, ascending=False)
    }

def main():
    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('classification_analysis')
    
    # Directories
    models_dir = Path("models")
    output_dir = Path("analysis_results")
    output_dir.mkdir(exist_ok=True, parents=True)