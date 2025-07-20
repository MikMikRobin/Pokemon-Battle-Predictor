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
    
    # Debug: Print data types
    print("\nDebugging data types:")
    print(f"Pokemon data types: {data_loader.pokemon_df.dtypes}")
    print(f"Processed Pokemon data types: {data_loader.processed_pokemon_df.dtypes}")
    
    # Create feature engineer
    feature_engineer = FeatureEngineer(data_loader.pokemon_df)
    
    # Prepare team battle data for VGC format
    print("\nPreparing VGC battle data...")
    try:
        X_train, X_test, y_train, y_test = data_loader.prepare_team_battle_data(
            battle_format=config.BattleFormat.VGC_2V2
        )
        print(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
        print(f"X_test shape: {X_test.shape}, dtype: {X_test.dtype}")
    except Exception as e:
        print(f"Error preparing battle data: {e}")
        import traceback
        traceback.print_exc()
        return None
    
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
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    try:
        if args.format in ["vgc", "both"]:
            print("\nTraining VGC model...")
            vgc_model = train_vgc_model(model_type=args.model_type)
            
        if args.format in ["standard", "both"]:
            print("\nTraining Standard model...")
            standard_model = train_standard_model(model_type=args.model_type)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nTry running with --debug flag for more information.")
        if args.debug:
            # Print more detailed information
            import sys
            print("\n\nDetailed error information:")
            print(f"Python version: {sys.version}")
            print(f"NumPy version: {np.__version__}")
            print(f"Pandas version: {pd.__version__}")
            
            # Try to print more information about the data
            try:
                data_loader = DataLoader()
                data_loader.load_data()
                data_loader.preprocess_pokemon_data()
                data_loader.debug_data()
            except Exception as debug_e:
                print(f"Error during debug data collection: {debug_e}")

if __name__ == "__main__":
    main() 