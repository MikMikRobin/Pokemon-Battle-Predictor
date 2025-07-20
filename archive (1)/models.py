"""
Model module for Pokemon Battle Predictor.
Contains model classes for different battle formats.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.metrics import classification_report, brier_score_loss, log_loss
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from typing import Dict, List, Tuple, Union, Any
import config
from feature_engineering import FeatureEngineer

class PokemonBattleModel:
    """Base class for Pokemon battle prediction models."""
    
    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize the model.
        
        Args:
            model_type: Type of model to use ("random_forest" or "logistic_regression")
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.explainer = None
        
        if model_type == "random_forest":
            self.model = RandomForestClassifier(**config.RANDOM_FOREST_PARAMS)
        elif model_type == "logistic_regression":
            self.model = LogisticRegression(**config.LOGISTIC_REGRESSION_PARAMS)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str] = None):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            feature_names: Names of features (for interpretability)
        """
        self.feature_names = feature_names
        self.model.fit(X_train, y_train)
        
        # Create explainer for model interpretability
        if self.model_type == "random_forest":
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.explainer = shap.LinearExplainer(self.model, X_train)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make binary predictions.
        
        Args:
            X: Features
            
        Returns:
            Binary predictions (0 or 1)
        """
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of team A winning.
        
        Args:
            X: Features
            
        Returns:
            Probability predictions
        """
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'brier_score': brier_score_loss(y_test, y_proba),
            'log_loss': log_loss(y_test, y_proba)
        }
        
        print(f"Model Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = config.CV_FOLDS) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Labels
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary of cross-validation metrics
        """
        cv_accuracy = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        cv_roc_auc = cross_val_score(self.model, X, y, cv=cv, scoring='roc_auc')
        
        cv_metrics = {
            'cv_accuracy_mean': cv_accuracy.mean(),
            'cv_accuracy_std': cv_accuracy.std(),
            'cv_roc_auc_mean': cv_roc_auc.mean(),
            'cv_roc_auc_std': cv_roc_auc.std()
        }
        
        print(f"Cross-Validation Results ({cv} folds):")
        print(f"  Accuracy: {cv_metrics['cv_accuracy_mean']:.4f} ± {cv_metrics['cv_accuracy_std']:.4f}")
        print(f"  ROC AUC: {cv_metrics['cv_roc_auc_mean']:.4f} ± {cv_metrics['cv_roc_auc_std']:.4f}")
        
        return cv_metrics
    
    def tune_hyperparameters(self, X: np.ndarray, y: np.ndarray, param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Tune hyperparameters using grid search.
        
        Args:
            X: Features
            y: Labels
            param_grid: Grid of hyperparameters to search
            
        Returns:
            Best hyperparameters
        """
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=config.CV_FOLDS,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        print(f"Best Hyperparameters: {grid_search.best_params_}")
        print(f"Best Score: {grid_search.best_score_:.4f}")
        
        # Update model with best hyperparameters
        self.model = grid_search.best_estimator_
        
        return grid_search.best_params_
    
    def calibrate_probabilities(self, X_train: np.ndarray, y_train: np.ndarray, method: str = 'sigmoid'):
        """
        Calibrate probability predictions.
        
        Args:
            X_train: Training features
            y_train: Training labels
            method: Calibration method ('sigmoid' or 'isotonic')
        """
        calibrated_model = CalibratedClassifierCV(
            base_estimator=self.model,
            method=method,
            cv=5
        )
        
        calibrated_model.fit(X_train, y_train)
        self.model = calibrated_model
        
        return self
    
    def plot_feature_importance(self, top_n: int = 20, save_path: str = None):
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to show
            save_path: Path to save the plot
        """
        if self.model_type == "random_forest":
            importances = self.model.feature_importances_
            if self.feature_names is None:
                feature_names = [f"Feature {i}" for i in range(len(importances))]
            else:
                feature_names = self.feature_names
            
            # Create DataFrame for plotting
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
            
            # Plot
            plt.figure(figsize=(10, 8))
            sns.barplot(x='Importance', y='Feature', data=importance_df)
            plt.title(f'Top {top_n} Feature Importance')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
            
            plt.show()
        
        elif self.model_type == "logistic_regression":
            coef = self.model.coef_[0]
            if self.feature_names is None:
                feature_names = [f"Feature {i}" for i in range(len(coef))]
            else:
                feature_names = self.feature_names
            
            # Create DataFrame for plotting
            coef_df = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coef
            })
            
            # Sort by absolute coefficient value
            coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
            coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False).head(top_n)
            
            # Plot
            plt.figure(figsize=(10, 8))
            sns.barplot(x='Coefficient', y='Feature', data=coef_df)
            plt.title(f'Top {top_n} Feature Coefficients')
            plt.axvline(x=0, color='k', linestyle='--')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
            
            plt.show()
    
    def plot_shap_summary(self, X_test: np.ndarray, save_path: str = None):
        """
        Plot SHAP summary.
        
        Args:
            X_test: Test features
            save_path: Path to save the plot
        """
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_test)
        
        # For random forest, shap_values is a list with one element per class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use values for class 1 (win)
        
        # Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, 
            X_test, 
            feature_names=self.feature_names if self.feature_names else None,
            show=False
        )
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    def explain_prediction(self, X: np.ndarray) -> Dict[str, float]:
        """
        Explain a single prediction.
        
        Args:
            X: Features for a single instance
            
        Returns:
            Dictionary mapping features to their contributions
        """
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # For random forest, shap_values is a list with one element per class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use values for class 1 (win)
        
        # Create explanation dictionary
        if self.feature_names is None:
            feature_names = [f"Feature {i}" for i in range(X.shape[1])]
        else:
            feature_names = self.feature_names
        
        explanation = {
            feature: float(shap_value)
            for feature, shap_value in zip(feature_names, shap_values[0])
        }
        
        return explanation
    
    def save_model(self, path: str):
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model
        """
        joblib.dump(self, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: str) -> 'PokemonBattleModel':
        """
        Load a model from a file.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded model
        """
        model = joblib.load(path)
        print(f"Model loaded from {path}")
        return model


class VGCBattleModel(PokemonBattleModel):
    """Model for predicting VGC (2v2) battles."""
    
    def __init__(self, model_type: str = "random_forest"):
        """Initialize the VGC battle model."""
        super().__init__(model_type)
        self.feature_engineer = None
    
    def set_feature_engineer(self, feature_engineer: FeatureEngineer):
        """Set the feature engineer."""
        self.feature_engineer = feature_engineer
        return self
    
    def predict_battle(self, team1_ids: List[int], team2_ids: List[int]) -> Dict[str, Any]:
        """
        Predict the outcome of a VGC battle.
        
        Args:
            team1_ids: IDs of Pokemon in the first team (4 Pokemon)
            team2_ids: IDs of Pokemon in the second team (4 Pokemon)
            
        Returns:
            Dictionary with prediction results
        """
        if self.feature_engineer is None:
            raise ValueError("Feature engineer not set. Call set_feature_engineer() first.")
        
        # Create features
        features = self.feature_engineer.create_vgc_features(team1_ids, team2_ids)
        features = features.reshape(1, -1)  # Reshape for single prediction
        
        # Make prediction
        win_probability = self.predict_proba(features)[0]
        prediction = 1 if win_probability > 0.5 else 0
        
        # Get explanation
        explanation = self.explain_prediction(features)
        
        # Sort features by importance
        top_features = sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        
        result = {
            'team1_win_probability': float(win_probability),
            'team2_win_probability': float(1 - win_probability),
            'predicted_winner': 'Team 1' if prediction == 1 else 'Team 2',
            'confidence': float(max(win_probability, 1 - win_probability)),
            'top_features': dict(top_features)
        }
        
        return result


class StandardBattleModel(PokemonBattleModel):
    """Model for predicting Standard (6v6) battles."""
    
    def __init__(self, model_type: str = "random_forest"):
        """Initialize the Standard battle model."""
        super().__init__(model_type)
        self.feature_engineer = None
    
    def set_feature_engineer(self, feature_engineer: FeatureEngineer):
        """Set the feature engineer."""
        self.feature_engineer = feature_engineer
        return self
    
    def predict_battle(self, team1_ids: List[int], team2_ids: List[int]) -> Dict[str, Any]:
        """
        Predict the outcome of a Standard battle.
        
        Args:
            team1_ids: IDs of Pokemon in the first team (6 Pokemon)
            team2_ids: IDs of Pokemon in the second team (6 Pokemon)
            
        Returns:
            Dictionary with prediction results
        """
        if self.feature_engineer is None:
            raise ValueError("Feature engineer not set. Call set_feature_engineer() first.")
        
        # Create features
        features = self.feature_engineer.create_standard_features(team1_ids, team2_ids)
        features = features.reshape(1, -1)  # Reshape for single prediction
        
        # Make prediction
        win_probability = self.predict_proba(features)[0]
        prediction = 1 if win_probability > 0.5 else 0
        
        # Get explanation
        explanation = self.explain_prediction(features)
        
        # Sort features by importance
        top_features = sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        
        result = {
            'team1_win_probability': float(win_probability),
            'team2_win_probability': float(1 - win_probability),
            'predicted_winner': 'Team 1' if prediction == 1 else 'Team 2',
            'confidence': float(max(win_probability, 1 - win_probability)),
            'top_features': dict(top_features)
        }
        
        return result 