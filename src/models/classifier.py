import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

logger = logging.getLogger(__name__)

class BaseClassifier:
    """Base classifier interface for all models"""
    
    def __init__(self, **kwargs):
        self.model = None
        self.model_params = kwargs
        
    def fit(self, X, y):
        """Train the model on input data"""
        logger.info(f"Training {self.__class__.__name__}")
        self.model.fit(X, y)
        return self
        
    def predict(self, X):
        """Predict classes for input data"""
        return self.model.predict(X)
    
    def evaluate(self, X, y_true):
        """Evaluate model performance"""
        y_pred = self.predict(X)
        
        # Generate evaluation metrics
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        logger.info(f"Model Accuracy: {accuracy:.4f}")
        logger.info(f"Classification Report:\n{report}")
        
        return {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': conf_matrix
        }

class RandomForestModel(BaseClassifier):
    """Random Forest classifier implementation"""
    
    def __init__(self, n_estimators=100, max_depth=None, **kwargs):
        super().__init__(**kwargs)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            **kwargs
        )
        
    def feature_importance(self):
        """Return feature importance if available"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None

class SVMModel(BaseClassifier):
    """Support Vector Machine classifier implementation"""
    
    def __init__(self, kernel='rbf', C=1.0, **kwargs):
        super().__init__(**kwargs)
        self.model = SVC(
            kernel=kernel,
            C=C,
            **kwargs
        )

class GaussianNBModel(BaseClassifier):
    """Gaussian Naive Bayes classifier implementation"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = GaussianNB(**kwargs)

class KNNModel(BaseClassifier):
    """K-Nearest Neighbors classifier implementation"""
    
    def __init__(self, n_neighbors=5, **kwargs):
        super().__init__(**kwargs)
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            **kwargs
        )

class LogisticRegressionModel(BaseClassifier):
    """Logistic Regression classifier implementation"""
    
    def __init__(self, max_iter=1000, **kwargs):
        super().__init__(**kwargs)
        self.model = LogisticRegression(
            max_iter=max_iter,
            **kwargs
        )

class GMMClassifier(BaseEstimator, ClassifierMixin):
    """Gaussian Mixture Model for classification"""
    
    def __init__(self, n_components=5, **kwargs):
        self.n_components = n_components
        self.kwargs = kwargs
        self.models = {}
        self.classes_ = None
        
    def fit(self, X, y):
        """Train a GMM for each class"""
        logger.info("Training GMM Classifier")
        self.classes_ = np.unique(y)
        
        # Train a separate GMM for each class
        for class_label in self.classes_:
            # Filter data for this class
            X_class = X[y == class_label]
            
            if len(X_class) > 0:
                # Create and train GMM for this class
                gmm = GaussianMixture(
                    n_components=min(self.n_components, len(X_class)),
                    **self.kwargs
                )
                gmm.fit(X_class)
                self.models[class_label] = gmm
            else:
                logger.warning(f"No samples for class {class_label}, skipping GMM training")
                
        return self
        
    def predict_proba(self, X):
        """Calculate probability for each class based on GMM scores"""
        # Initialize probabilities array
        proba = np.zeros((X.shape[0], len(self.classes_)))
        
        # For each class, calculate log probability density
        for i, class_label in enumerate(self.classes_):
            if class_label in self.models:
                # Calculate score for each sample
                score = self.models[class_label].score_samples(X)
                proba[:, i] = score
            else:
                proba[:, i] = -np.inf
                
        # Normalize to get probabilities (softmax)
        proba = np.exp(proba - np.max(proba, axis=1, keepdims=True))
        proba = proba / np.sum(proba, axis=1, keepdims=True)
        
        return proba
        
    def predict(self, X):
        """Predict class labels for samples in X"""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
        
    def evaluate(self, X, y_true):
        """Evaluate model performance"""
        y_pred = self.predict(X)
        
        # Generate evaluation metrics
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        logger.info(f"GMM Model Accuracy: {accuracy:.4f}")
        logger.info(f"Classification Report:\n{report}")
        
        return {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': conf_matrix
        }