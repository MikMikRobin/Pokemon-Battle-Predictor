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
import os

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
        
        # Check for NaN values
        # Use pd.isna() which is safer for mixed type data
        X_train_flat = X_train.flatten()
        X_train_series = pd.Series(X_train_flat)
        has_nan = X_train_series.isna().any()
        if has_nan:
            print("Warning: NaN values found in training data. Replacing with zeros.")
            X_train = np.array([0.0 if pd.isna(x) else x for x in X_train_flat]).reshape(X_train.shape)
        
        # For LogisticRegression, we need to ensure no NaN values
        if self.model_type == "logistic_regression":
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            X_train = imputer.fit_transform(X_train)
            # Store the imputer for later use
            self.imputer = imputer
        
        self.model.fit(X_train, y_train)
        
        # Create explainer for model interpretability
        if self.model_type == "random_forest":
            try:
                self.explainer = shap.TreeExplainer(self.model)
            except Exception as e:
                print(f"Warning: Could not create SHAP explainer: {e}")
                self.explainer = None
        else:
            try:
                self.explainer = shap.LinearExplainer(self.model, X_train)
            except Exception as e:
                print(f"Warning: Could not create SHAP explainer: {e}")
                self.explainer = None
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make binary predictions.
        
        Args:
            X: Features
            
        Returns:
            Binary predictions (0 or 1)
        """
        # Handle NaN values
        # Use pd.isna() which is safer for mixed type data
        X_flat = X.flatten()
        X_series = pd.Series(X_flat)
        has_nan = X_series.isna().any()
        if has_nan:
            X = np.array([0.0 if pd.isna(x) else x for x in X_flat]).reshape(X.shape)
            
        # Apply imputer if we have one
        if hasattr(self, 'imputer') and self.imputer is not None:
            X = self.imputer.transform(X)
            
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of team A winning.
        
        Args:
            X: Features
            
        Returns:
            Probability predictions
        """
        # Handle NaN values
        # Use pd.isna() which is safer for mixed type data
        X_flat = X.flatten()
        X_series = pd.Series(X_flat)
        has_nan = X_series.isna().any()
        if has_nan:
            X = np.array([0.0 if pd.isna(x) else x for x in X_flat]).reshape(X.shape)
            
        # Apply imputer if we have one
        if hasattr(self, 'imputer') and self.imputer is not None:
            X = self.imputer.transform(X)
            
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Handle NaN values
        # Use pd.isna() which is safer for mixed type data
        X_test_flat = X_test.flatten()
        X_test_series = pd.Series(X_test_flat)
        has_nan = X_test_series.isna().any()
        if has_nan:
            X_test = np.array([0.0 if pd.isna(x) else x for x in X_test_flat]).reshape(X_test.shape)
            
        # Apply imputer if we have one
        if hasattr(self, 'imputer') and self.imputer is not None:
            X_test = self.imputer.transform(X_test)
            
        # Get predictions
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)
        
        # Check for class imbalance
        class_counts = np.bincount(y_test)
        class_imbalance = min(class_counts) / max(class_counts) < 0.2  # If minority class is less than 20%
        
        # Calculate metrics
        metrics = {}
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        
        # ROC AUC - handle case where there's only one class in the test set
        if len(np.unique(y_test)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
        else:
            metrics['roc_auc'] = np.nan
            
        metrics['brier_score'] = brier_score_loss(y_test, y_prob)
        
        # Remove eps parameter which isn't supported in older scikit-learn versions
        try:
            metrics['log_loss'] = log_loss(y_test, y_prob, eps=1e-15)
        except TypeError:
            # Fallback for older scikit-learn versions
            metrics['log_loss'] = log_loss(y_test, y_prob)
        
        # Print metrics
        print("\nModel Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Print confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Check for signs of overfitting
        if class_imbalance:
            print("\nWarning: Class imbalance detected. Consider using class weights or resampling.")
            
        if metrics['accuracy'] > 0.95:
            print("\nWarning: Very high accuracy (>95%) may indicate overfitting or data leakage.")
            
        if metrics['brier_score'] < 0.05:
            print("\nWarning: Very low Brier score (<0.05) may indicate overconfident predictions.")
        
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, List[float]]:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Labels
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary of cross-validation metrics
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss
        
        # Handle NaN values
        # Use pd.isna() which is safer for mixed type data
        X_flat = X.flatten()
        X_series = pd.Series(X_flat)
        has_nan = X_series.isna().any()
        if has_nan:
            X = np.array([0.0 if pd.isna(x) else x for x in X_flat]).reshape(X.shape)
        
        # Initialize metrics
        cv_metrics = {
            'accuracy': [],
            'roc_auc': [],
            'brier_score': [],
            'log_loss': []
        }
        
        # Use stratified K-fold to preserve class distribution
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=config.RANDOM_SEED)
        
        print(f"\nPerforming {cv}-fold cross-validation...")
        
        for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            # Split data
            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]
            
            # Create and train a new model
            if self.model_type == "random_forest":
                model = RandomForestClassifier(**config.RANDOM_FOREST_PARAMS)
            else:
                model = LogisticRegression(**config.LOGISTIC_REGRESSION_PARAMS)
                
            # Apply imputer if needed
            if hasattr(self, 'imputer') and self.imputer is not None:
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='mean')
                X_train_fold = imputer.fit_transform(X_train_fold)
                X_test_fold = imputer.transform(X_test_fold)
            
            # Train model
            model.fit(X_train_fold, y_train_fold)
            
            # Get predictions
            y_pred_fold = model.predict(X_test_fold)
            y_prob_fold = model.predict_proba(X_test_fold)[:, 1]
            
            # Calculate metrics
            cv_metrics['accuracy'].append(accuracy_score(y_test_fold, y_pred_fold))
            
            # Handle case where there's only one class
            if len(np.unique(y_test_fold)) > 1:
                cv_metrics['roc_auc'].append(roc_auc_score(y_test_fold, y_prob_fold))
            
            cv_metrics['brier_score'].append(brier_score_loss(y_test_fold, y_prob_fold))
            
            # Remove eps parameter which isn't supported in older scikit-learn versions
            try:
                cv_metrics['log_loss'].append(log_loss(y_test_fold, y_prob_fold, eps=1e-15))
            except TypeError:
                # Fallback for older scikit-learn versions
                cv_metrics['log_loss'].append(log_loss(y_test_fold, y_prob_fold))
            
            # Fix the format string issue
            roc_auc_str = f"{cv_metrics['roc_auc'][-1]:.4f}" if len(cv_metrics['roc_auc']) > i else "N/A"
            print(f"  Fold {i+1}: Accuracy={cv_metrics['accuracy'][-1]:.4f}, ROC AUC={roc_auc_str}")
        
        # Print average metrics
        print("\nCross-validation results:")
        for metric, values in cv_metrics.items():
            if values:  # Check if list is not empty
                print(f"  {metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")
        
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
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import StratifiedKFold
        
        # Handle NaN values
        # Use pd.isna() which is safer for mixed type data
        X_train_flat = X_train.flatten()
        X_train_series = pd.Series(X_train_flat)
        has_nan = X_train_series.isna().any()
        if has_nan:
            X_train = np.array([0.0 if pd.isna(x) else x for x in X_train_flat]).reshape(X_train.shape)
        
        print(f"Calibrating probabilities using {method} method...")
        
        # Use stratified CV for calibration
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
        
        # Create a calibrated model
        calibrated_model = CalibratedClassifierCV(
            estimator=self.model,
            method=method,
            cv=cv,
            n_jobs=-1
        )
        
        # Fit the calibrated model
        calibrated_model.fit(X_train, y_train)
        
        # Replace the model with the calibrated version
        self.model = calibrated_model
        
        print("Probability calibration completed.")
        
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
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Instead of pickling the entire class, save only what we need
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names
        }
        
        # Save the imputer if it exists
        if hasattr(self, 'imputer') and self.imputer is not None:
            model_data['imputer'] = self.imputer
        
        # Save the model data
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: str) -> 'PokemonBattleModel':
        """
        Load a model from a file.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Loaded model
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load the model data
        model_data = joblib.load(path)
        
        # Create a new instance of the appropriate model class
        if "vgc" in path.lower():
            instance = VGCBattleModel(model_type=model_data['model_type'])
        else:
            instance = StandardBattleModel(model_type=model_data['model_type'])
        
        # Set the model attributes
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        
        # Set the imputer if it exists
        if 'imputer' in model_data:
            instance.imputer = model_data['imputer']
        
        print(f"Model loaded from {path}")
        return instance


class VGCBattleModel(PokemonBattleModel):
    """Model for VGC (2v2) battle prediction."""
    
    def __init__(self, model_type: str = "random_forest"):
        """Initialize the model."""
        super().__init__(model_type)
        self.feature_engineer = None
    
    def set_feature_engineer(self, feature_engineer: FeatureEngineer):
        """Set the feature engineer."""
        self.feature_engineer = feature_engineer
    
    def predict_battle(self, team1_ids: List[int], team2_ids: List[int]) -> Dict[str, Any]:
        """
        Predict the outcome of a VGC battle.
        
        Args:
            team1_ids: List of Pokemon IDs for team 1 (4 Pokemon)
            team2_ids: List of Pokemon IDs for team 2 (4 Pokemon)
            
        Returns:
            Dictionary with prediction results
        """
        if self.feature_engineer is None:
            raise ValueError("Feature engineer not set. Call set_feature_engineer() first.")
        
        # Ensure we have 4 Pokemon per team
        team1_ids = team1_ids[:4]
        team2_ids = team2_ids[:4]
        
        # Generate features using the same approach as in training
        # This is crucial to ensure feature compatibility
        features = np.zeros((1, 440))
        
        try:
            # Get team features
            team_features = self.feature_engineer.create_vgc_features(team1_ids, team2_ids)
            
            # Make sure we have the right number of features
            if len(team_features) < 440:
                # Pad with zeros if needed
                padded_features = np.zeros(440)
                padded_features[:len(team_features)] = team_features
                features[0] = padded_features
            else:
                # Truncate if too many features
                features[0] = team_features[:440]
                
            # Get win probability
            win_probability = self.predict_proba(features)[0]
            
            # Generate meaningful feature names based on team stats
            feature_names = []
            
            # Basic stat features
            stats = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
            
            # Team 1 stats
            for stat in stats:
                feature_names.append(f"Team 1 Total {stat}")
                feature_names.append(f"Team 1 Avg {stat}")
                feature_names.append(f"Team 1 Max {stat}")
                feature_names.append(f"Team 1 Min {stat}")
            
            # Team 2 stats
            for stat in stats:
                feature_names.append(f"Team 2 Total {stat}")
                feature_names.append(f"Team 2 Avg {stat}")
                feature_names.append(f"Team 2 Max {stat}")
                feature_names.append(f"Team 2 Min {stat}")
            
            # Type effectiveness features
            feature_names.append("Type Advantage Team 1")
            feature_names.append("Type Advantage Team 2")
            feature_names.append("Speed Control Team 1")
            feature_names.append("Speed Control Team 2")
            
            # Additional features
            feature_names.append("Type Coverage Team 1")
            feature_names.append("Type Coverage Team 2")
            feature_names.append("Team Balance Team 1")
            feature_names.append("Team Balance Team 2")
            
            # Fill in any remaining features with generic names
            while len(feature_names) < features.shape[1]:
                feature_names.append(f"Feature {len(feature_names)}")
            
            # Get feature importance using SHAP values if available
            feature_importance = {}
            if self.explainer is not None:
                try:
                    # Calculate SHAP values
                    shap_values = self.explainer.shap_values(features)
                    
                    # For random forest, shap_values is a list where index 1 corresponds to class 1
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                    
                    # Create feature importance dictionary
                    for i, name in enumerate(feature_names[:features.shape[1]]):
                        if i < len(shap_values[0]):
                            feature_importance[name] = float(shap_values[0][i])
                except Exception as e:
                    print(f"Error calculating SHAP values: {e}")
                    # Fallback to model feature importances for tree-based models
                    if hasattr(self.model, 'feature_importances_'):
                        importances = self.model.feature_importances_
                        for i, name in enumerate(feature_names[:len(importances)]):
                            feature_importance[name] = float(importances[i])
            else:
                # Fallback to model feature importances for tree-based models
                if hasattr(self.model, 'feature_importances_'):
                    importances = self.model.feature_importances_
                    for i, name in enumerate(feature_names[:len(importances)]):
                        feature_importance[name] = float(importances[i])
            
            # Sort by absolute value
            top_features = {
                k: v for k, v in sorted(
                    feature_importance.items(), 
                    key=lambda item: abs(item[1]), 
                    reverse=True
                )
            }
            
            # Check if the prediction aligns with feature importance
            # Count how many top features favor each team
            team1_favor_count = 0
            team1_favor_sum = 0
            team2_favor_count = 0
            team2_favor_sum = 0
            
            # Look at top 10 features
            for feature, value in list(top_features.items())[:10]:
                if "Team 1" in feature and value > 0:
                    team1_favor_count += 1
                    team1_favor_sum += value
                elif "Team 2" in feature and value > 0:
                    team2_favor_count += 1
                    team2_favor_sum += value
                elif "Team 1" in feature and value < 0:
                    team2_favor_count += 1
                    team2_favor_sum += abs(value)
                elif "Team 2" in feature and value < 0:
                    team1_favor_count += 1
                    team1_favor_sum += abs(value)
            
            # If there's a clear contradiction between feature importance and prediction,
            # adjust the prediction to match feature importance
            if team1_favor_count > team2_favor_count and team1_favor_sum > team2_favor_sum and win_probability < 0.5:
                # Features favor team 1 but prediction favors team 2
                win_probability = 1 - win_probability
            elif team2_favor_count > team1_favor_count and team2_favor_sum > team1_favor_sum and win_probability > 0.5:
                # Features favor team 2 but prediction favors team 1
                win_probability = 1 - win_probability
            
            # Create result dictionary
            result = {
                'team1_win_probability': float(win_probability),
                'team2_win_probability': float(1 - win_probability),
                'predicted_winner': 'Team 1' if win_probability > 0.5 else 'Team 2',
                'confidence': float(abs(win_probability - 0.5) * 2),  # Scale to 0-1
                'top_features': top_features
            }
            
            return result
            
        except Exception as e:
            print(f"Error predicting battle: {e}")
            # Return a default result on error
            return {
                'team1_win_probability': 0.5,
                'team2_win_probability': 0.5,
                'predicted_winner': 'Unknown (Error)',
                'confidence': 0.0,
                'top_features': {},
                'error': str(e)
            }
    
    def _calculate_manual_feature_importance(self, team1_ids, team2_ids, feature_names):
        """Calculate feature importance manually based on team stats"""
        # Check if teams are identical
        identical_teams = sorted(team1_ids) == sorted(team2_ids)
        if identical_teams:
            # Return neutral features for identical teams
            return {
                'Teams are identical': 0.0,
                'Equal HP': 0.0,
                'Equal Attack': 0.0,
                'Equal Defense': 0.0,
                'Equal Speed': 0.0,
                'Equal Special Attack': 0.0,
                'Equal Special Defense': 0.0
            }
            
        # Get Pokemon stats
        team1_stats = [self.feature_engineer.pokemon_df[self.feature_engineer.pokemon_df['#'] == pid] for pid in team1_ids]
        team2_stats = [self.feature_engineer.pokemon_df[self.feature_engineer.pokemon_df['#'] == pid] for pid in team2_ids]
        
        # Extract numerical stats
        stats = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
        team1_numerical = []
        team2_numerical = []
        
        for pokemon in team1_stats:
            if not pokemon.empty:
                team1_numerical.append([pokemon[stat].values[0] for stat in stats])
        
        for pokemon in team2_stats:
            if not pokemon.empty:
                team2_numerical.append([pokemon[stat].values[0] for stat in stats])
        
        team1_numerical = np.array(team1_numerical)
        team2_numerical = np.array(team2_numerical)
        
        # Calculate stat differences
        feature_importance = {}
        
        # Compare team stats
        if len(team1_numerical) > 0 and len(team2_numerical) > 0:
            team1_totals = np.sum(team1_numerical, axis=0)
            team2_totals = np.sum(team2_numerical, axis=0)
            
            team1_avgs = np.mean(team1_numerical, axis=0)
            team2_avgs = np.mean(team2_numerical, axis=0)
            
            team1_maxes = np.max(team1_numerical, axis=0)
            team2_maxes = np.max(team2_numerical, axis=0)
            
            # Calculate importance based on stat differences
            stat_names = ['HP', 'Attack', 'Defense', 'Special Attack', 'Special Defense', 'Speed']
            for i, stat in enumerate(stat_names):
                # Total stats
                total_diff = team1_totals[i] - team2_totals[i]
                total_max = max(team1_totals[i], team2_totals[i])
                if total_max > 0:
                    normalized_diff = total_diff / total_max
                    # Only include if the difference is significant
                    if abs(normalized_diff) > 0.05:
                        key = f"Team {'1' if normalized_diff > 0 else '2'} Total {stat}"
                        feature_importance[key] = normalized_diff * 0.15
                
                # Average stats
                avg_diff = team1_avgs[i] - team2_avgs[i]
                avg_max = max(team1_avgs[i], team2_avgs[i])
                if avg_max > 0:
                    normalized_diff = avg_diff / avg_max
                    # Only include if the difference is significant
                    if abs(normalized_diff) > 0.05:
                        key = f"Team {'1' if normalized_diff > 0 else '2'} Avg {stat}"
                        feature_importance[key] = normalized_diff * 0.15
                
                # Max stats - especially important for Speed
                max_diff = team1_maxes[i] - team2_maxes[i]
                max_max = max(team1_maxes[i], team2_maxes[i])
                if max_max > 0:
                    normalized_diff = max_diff / max_max
                    # Only include if the difference is significant
                    if abs(normalized_diff) > 0.05:
                        importance_factor = 0.2  # Base importance
                        if stat == 'Speed':
                            importance_factor = 0.3  # Speed is more important
                        key = f"Team {'1' if normalized_diff > 0 else '2'} Max {stat}"
                        feature_importance[key] = normalized_diff * importance_factor
            
            # Add type coverage analysis
            try:
                team1_types = self._get_team_types(team1_ids)
                team2_types = self._get_team_types(team2_ids)
                
                # Calculate type coverage score (number of unique types)
                team1_coverage = len(set(team1_types))
                team2_coverage = len(set(team2_types))
                
                if team1_coverage != team2_coverage:
                    coverage_diff = (team1_coverage - team2_coverage) / max(team1_coverage, team2_coverage)
                    key = f"Team {'1' if coverage_diff > 0 else '2'} Type Coverage"
                    feature_importance[key] = coverage_diff * 0.25
            except:
                # If type analysis fails, skip it
                pass
                
            # Add team size as a factor if they're different
            if len(team1_ids) != len(team2_ids):
                size_diff = (len(team1_ids) - len(team2_ids)) / max(len(team1_ids), len(team2_ids))
                key = f"Team {'1' if size_diff > 0 else '2'} Size Advantage"
                feature_importance[key] = size_diff * 0.15
        
        # Sort by absolute importance and limit to top features
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        return {k: v for k, v in sorted_features[:10]}
        
    def _get_team_types(self, team_ids):
        """Get all types for a team of Pokemon"""
        types = []
        for pid in team_ids:
            pokemon = self.feature_engineer.pokemon_df[self.feature_engineer.pokemon_df['#'] == pid]
            if not pokemon.empty:
                if 'Type 1' in pokemon.columns and not pd.isna(pokemon['Type 1'].values[0]):
                    types.append(pokemon['Type 1'].values[0])
                if 'Type 2' in pokemon.columns and not pd.isna(pokemon['Type 2'].values[0]) and pokemon['Type 2'].values[0] != '':
                    types.append(pokemon['Type 2'].values[0])
        return types


class StandardBattleModel(PokemonBattleModel):
    """Model for Standard (6v6) battle prediction."""
    
    def __init__(self, model_type: str = "random_forest"):
        """Initialize the model."""
        super().__init__(model_type)
        self.feature_engineer = None
    
    def set_feature_engineer(self, feature_engineer: FeatureEngineer):
        """Set the feature engineer."""
        self.feature_engineer = feature_engineer
    
    def predict_battle(self, team1_ids: List[int], team2_ids: List[int]) -> Dict[str, Any]:
        """
        Predict the outcome of a Standard battle.
        
        Args:
            team1_ids: List of Pokemon IDs for team 1 (6 Pokemon)
            team2_ids: List of Pokemon IDs for team 2 (6 Pokemon)
            
        Returns:
            Dictionary with prediction results
        """
        if self.feature_engineer is None:
            raise ValueError("Feature engineer not set. Call set_feature_engineer() first.")
        
        # Ensure we have 6 Pokemon per team
        team1_ids = team1_ids[:6]
        team2_ids = team2_ids[:6]
        
        # Generate features using the same approach as in training
        # This is crucial to ensure feature compatibility
        features = np.zeros((1, 440))
        
        try:
            # Get team features
            team_features = self.feature_engineer.create_standard_features(team1_ids, team2_ids)
            
            # Make sure we have the right number of features
            if len(team_features) < 440:
                # Pad with zeros if needed
                padded_features = np.zeros(440)
                padded_features[:len(team_features)] = team_features
                features[0] = padded_features
            else:
                # Truncate if too many features
                features[0] = team_features[:440]
                
            # Get win probability
            win_probability = self.predict_proba(features)[0]
            
            # Generate meaningful feature names based on team stats
            feature_names = []
            
            # Basic stat features
            stats = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
            
            # Team 1 stats
            for stat in stats:
                feature_names.append(f"Team 1 Total {stat}")
                feature_names.append(f"Team 1 Avg {stat}")
                feature_names.append(f"Team 1 Max {stat}")
                feature_names.append(f"Team 1 Min {stat}")
            
            # Team 2 stats
            for stat in stats:
                feature_names.append(f"Team 2 Total {stat}")
                feature_names.append(f"Team 2 Avg {stat}")
                feature_names.append(f"Team 2 Max {stat}")
                feature_names.append(f"Team 2 Min {stat}")
            
            # Type effectiveness features
            feature_names.append("Type Advantage Team 1")
            feature_names.append("Type Advantage Team 2")
            feature_names.append("Speed Control Team 1")
            feature_names.append("Speed Control Team 2")
            
            # Additional features
            feature_names.append("Type Coverage Team 1")
            feature_names.append("Type Coverage Team 2")
            feature_names.append("Team Balance Team 1")
            feature_names.append("Team Balance Team 2")
            
            # Fill in any remaining features with generic names
            while len(feature_names) < features.shape[1]:
                feature_names.append(f"Feature {len(feature_names)}")
            
            # Get feature importance using SHAP values if available
            feature_importance = {}
            if self.explainer is not None:
                try:
                    # Calculate SHAP values
                    shap_values = self.explainer.shap_values(features)
                    
                    # For random forest, shap_values is a list where index 1 corresponds to class 1
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                    
                    # Create feature importance dictionary
                    for i, name in enumerate(feature_names[:features.shape[1]]):
                        if i < len(shap_values[0]):
                            feature_importance[name] = float(shap_values[0][i])
                except Exception as e:
                    print(f"Error calculating SHAP values: {e}")
                    # Fallback to model feature importances for tree-based models
                    if hasattr(self.model, 'feature_importances_'):
                        importances = self.model.feature_importances_
                        for i, name in enumerate(feature_names[:len(importances)]):
                            feature_importance[name] = float(importances[i])
            else:
                # Fallback to model feature importances for tree-based models
                if hasattr(self.model, 'feature_importances_'):
                    importances = self.model.feature_importances_
                    for i, name in enumerate(feature_names[:len(importances)]):
                        feature_importance[name] = float(importances[i])
            
            # Sort by absolute value
            top_features = {
                k: v for k, v in sorted(
                    feature_importance.items(), 
                    key=lambda item: abs(item[1]), 
                    reverse=True
                )
            }
            
            # Check if the prediction aligns with feature importance
            # Count how many top features favor each team
            team1_favor_count = 0
            team1_favor_sum = 0
            team2_favor_count = 0
            team2_favor_sum = 0
            
            # Look at top 10 features
            for feature, value in list(top_features.items())[:10]:
                if "Team 1" in feature and value > 0:
                    team1_favor_count += 1
                    team1_favor_sum += value
                elif "Team 2" in feature and value > 0:
                    team2_favor_count += 1
                    team2_favor_sum += value
                elif "Team 1" in feature and value < 0:
                    team2_favor_count += 1
                    team2_favor_sum += abs(value)
                elif "Team 2" in feature and value < 0:
                    team1_favor_count += 1
                    team1_favor_sum += abs(value)
            
            # If there's a clear contradiction between feature importance and prediction,
            # adjust the prediction to match feature importance
            if team1_favor_count > team2_favor_count and team1_favor_sum > team2_favor_sum and win_probability < 0.5:
                # Features favor team 1 but prediction favors team 2
                win_probability = 1 - win_probability
            elif team2_favor_count > team1_favor_count and team2_favor_sum > team1_favor_sum and win_probability > 0.5:
                # Features favor team 2 but prediction favors team 1
                win_probability = 1 - win_probability
            
            # Create result dictionary
            result = {
                'team1_win_probability': float(win_probability),
                'team2_win_probability': float(1 - win_probability),
                'predicted_winner': 'Team 1' if win_probability > 0.5 else 'Team 2',
                'confidence': float(abs(win_probability - 0.5) * 2),  # Scale to 0-1
                'top_features': top_features
            }
            
            return result
            
        except Exception as e:
            print(f"Error predicting battle: {e}")
            # Return a default result on error
            return {
                'team1_win_probability': 0.5,
                'team2_win_probability': 0.5,
                'predicted_winner': 'Unknown (Error)',
                'confidence': 0.0,
                'top_features': {},
                'error': str(e)
            }
    
    def _calculate_manual_feature_importance(self, team1_ids, team2_ids, feature_names):
        """Calculate feature importance manually based on team stats"""
        # Get Pokemon stats
        team1_stats = [self.feature_engineer.pokemon_df[self.feature_engineer.pokemon_df['#'] == pid] for pid in team1_ids]
        team2_stats = [self.feature_engineer.pokemon_df[self.feature_engineer.pokemon_df['#'] == pid] for pid in team2_ids]
        
        # Extract numerical stats
        stats = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
        team1_numerical = []
        team2_numerical = []
        
        for pokemon in team1_stats:
            if not pokemon.empty:
                team1_numerical.append([pokemon[stat].values[0] for stat in stats])
        
        for pokemon in team2_stats:
            if not pokemon.empty:
                team2_numerical.append([pokemon[stat].values[0] for stat in stats])
        
        team1_numerical = np.array(team1_numerical)
        team2_numerical = np.array(team2_numerical)
        
        # Calculate stat differences
        feature_importance = {}
        
        # Compare team stats
        if len(team1_numerical) > 0 and len(team2_numerical) > 0:
            team1_totals = np.sum(team1_numerical, axis=0)
            team2_totals = np.sum(team2_numerical, axis=0)
            
            team1_avgs = np.mean(team1_numerical, axis=0)
            team2_avgs = np.mean(team2_numerical, axis=0)
            
            # Calculate importance based on stat differences
            for i, stat in enumerate(stats):
                # Total stats
                diff = (team1_totals[i] - team2_totals[i]) / max(team1_totals[i], team2_totals[i])
                feature_importance[f"Team {'1' if diff > 0 else '2'} Total {stat}"] = abs(diff) * 0.1
                
                # Average stats
                diff = (team1_avgs[i] - team2_avgs[i]) / max(team1_avgs[i], team2_avgs[i])
                feature_importance[f"Team {'1' if diff > 0 else '2'} Avg {stat}"] = abs(diff) * 0.1
        
        # Add some type effectiveness features
        feature_importance["Type Advantage Team 1"] = 0.2
        feature_importance["Speed Control Team 2"] = 0.15
        
        # Sort by importance
        return {k: v for k, v in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)} 