"""
Training script for Pokemon Battle Predictor models.
"""
import argparse
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

import config
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from models import VGCBattleModel, StandardBattleModel

def train_vgc_model(model_type="random_forest", save_path="models/vgc_model.pkl"):
    """
    Train a VGC (2v2) battle prediction model.
    
    Args:
        model_type: Type of model to use ("random_forest" or "logistic_regression")
        save_path: Path to save the trained model
    """
    print("=" * 80)
    print(f"Training VGC (2v2) Battle Model with {model_type}")
    print("=" * 80)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    data_loader = DataLoader()
    data_loader.load_data()
    data_loader.preprocess_pokemon_data()
    
    # Create feature engineer
    feature_engineer = FeatureEngineer(data_loader.pokemon_df)
    
    # Prepare team battle data for VGC format
    print("Preparing VGC battle data...")
    X_train, X_test, y_train, y_test = data_loader.prepare_team_battle_data(
        battle_format=config.BattleFormat.VGC_2V2
    )
    
    # Create and train model
    print("Training model...")
    model = VGCBattleModel(model_type=model_type)
    model.set_feature_engineer(feature_engineer)
    model.fit(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    model.evaluate(X_test, y_test)
    
    # Cross-validate
    print("\nPerforming cross-validation...")
    model.cross_validate(np.concatenate([X_train, X_test]), np.concatenate([y_train, y_test]))
    
    # Calibrate probabilities
    print("\nCalibrating probabilities...")
    model.calibrate_probabilities(X_train, y_train)
    
    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save_model(save_path)
    
    return model

def train_standard_model(model_type="random_forest", save_path="models/standard_model.pkl"):
    """
    Train a Standard (6v6) battle prediction model.
    
    Args:
        model_type: Type of model to use ("random_forest" or "logistic_regression")
        save_path: Path to save the trained model
    """
    print("=" * 80)
    print(f"Training Standard (6v6) Battle Model with {model_type}")
    print("=" * 80)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    data_loader = DataLoader()
    data_loader.load_data()
    data_loader.preprocess_pokemon_data()
    
    # Create feature engineer
    feature_engineer = FeatureEngineer(data_loader.pokemon_df)
    
    # Prepare team battle data for Standard format
    print("Preparing Standard battle data...")
    X_train, X_test, y_train, y_test = data_loader.prepare_team_battle_data(
        battle_format=config.BattleFormat.STANDARD_6V6
    )
    
    # Create and train model
    print("Training model...")
    model = StandardBattleModel(model_type=model_type)
    model.set_feature_engineer(feature_engineer)
    model.fit(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    model.evaluate(X_test, y_test)
    
    # Cross-validate
    print("\nPerforming cross-validation...")
    model.cross_validate(np.concatenate([X_train, X_test]), np.concatenate([y_train, y_test]))
    
    # Calibrate probabilities
    print("\nCalibrating probabilities...")
    model.calibrate_probabilities(X_train, y_train)
    
    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save_model(save_path)
    
    return model

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train Pokemon Battle Predictor models.")
    parser.add_argument(
        "--format", 
        type=str, 
        choices=["vgc", "standard", "both"], 
        default="both",
        help="Battle format to train model for"
    )
    parser.add_argument(
        "--model-type", 
        type=str, 
        choices=["random_forest", "logistic_regression"], 
        default="random_forest",
        help="Type of model to use"
    )
    
    args = parser.parse_args()
    
    if args.format in ["vgc", "both"]:
        vgc_model = train_vgc_model(model_type=args.model_type)
        
    if args.format in ["standard", "both"]:
        standard_model = train_standard_model(model_type=args.model_type)

if __name__ == "__main__":
    main() 