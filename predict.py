"""
Prediction script for Pokemon Battle Predictor models.
"""
import argparse
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt

import config
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from models import VGCBattleModel, StandardBattleModel, PokemonBattleModel
from pokemon_utils import display_battle

def load_pokemon_data():
    """Load and preprocess Pokemon data."""
    data_loader = DataLoader()
    data_loader.load_data()
    data_loader.preprocess_pokemon_data()
    
    # Create feature engineer
    feature_engineer = FeatureEngineer(data_loader.pokemon_df)
    
    return data_loader, feature_engineer

def predict_vgc_battle(team1_ids, team2_ids, model_path="models/vgc_model.pkl"):
    """
    Predict the outcome of a VGC (2v2) battle.
    
    Args:
        team1_ids: List of Pokemon IDs for team 1 (4 Pokemon)
        team2_ids: List of Pokemon IDs for team 2 (4 Pokemon)
        model_path: Path to the trained model
        
    Returns:
        Dictionary with prediction results
    """
    # Load data and feature engineer
    data_loader, feature_engineer = load_pokemon_data()
    
    # Load model
    model = PokemonBattleModel.load_model(model_path)
    
    # Set feature engineer
    if hasattr(model, 'set_feature_engineer'):
        model.set_feature_engineer(feature_engineer)
    
    # Get team Pokemon names
    team1_names = [data_loader.pokemon_df[data_loader.pokemon_df['#'] == pid]['Name'].values[0] for pid in team1_ids]
    team2_names = [data_loader.pokemon_df[data_loader.pokemon_df['#'] == pid]['Name'].values[0] for pid in team2_ids]
    
    # Make prediction
    result = model.predict_battle(team1_ids, team2_ids)
    
    # Add team information
    result['team1'] = [{'id': int(pid), 'name': name} for pid, name in zip(team1_ids, team1_names)]
    result['team2'] = [{'id': int(pid), 'name': name} for pid, name in zip(team2_ids, team2_names)]
    
    return result

def predict_standard_battle(team1_ids, team2_ids, model_path="models/standard_model.pkl"):
    """
    Predict the outcome of a Standard (6v6) battle.
    
    Args:
        team1_ids: List of Pokemon IDs for team 1 (6 Pokemon)
        team2_ids: List of Pokemon IDs for team 2 (6 Pokemon)
        model_path: Path to the trained model
        
    Returns:
        Dictionary with prediction results
    """
    # Load data and feature engineer
    data_loader, feature_engineer = load_pokemon_data()
    
    # Load model
    model = PokemonBattleModel.load_model(model_path)
    
    # Set feature engineer
    if hasattr(model, 'set_feature_engineer'):
        model.set_feature_engineer(feature_engineer)
    
    # Get team Pokemon names
    team1_names = [data_loader.pokemon_df[data_loader.pokemon_df['#'] == pid]['Name'].values[0] for pid in team1_ids]
    team2_names = [data_loader.pokemon_df[data_loader.pokemon_df['#'] == pid]['Name'].values[0] for pid in team2_ids]
    
    # Make prediction
    result = model.predict_battle(team1_ids, team2_ids)
    
    # Add team information
    result['team1'] = [{'id': int(pid), 'name': name} for pid, name in zip(team1_ids, team1_names)]
    result['team2'] = [{'id': int(pid), 'name': name} for pid, name in zip(team2_ids, team2_names)]
    
    return result

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Make Pokemon battle predictions.")
    parser.add_argument(
        "--format", 
        type=str, 
        choices=["vgc", "standard"], 
        required=True,
        help="Battle format to predict"
    )
    parser.add_argument(
        "--team1", 
        type=int, 
        nargs="+", 
        required=True,
        help="Pokemon IDs for team 1"
    )
    parser.add_argument(
        "--team2", 
        type=int, 
        nargs="+", 
        required=True,
        help="Pokemon IDs for team 2"
    )
    parser.add_argument(
        "--model-path", 
        type=str,
        help="Path to the trained model"
    )
    parser.add_argument(
        "--output", 
        type=str,
        help="Path to save prediction results as JSON"
    )
    parser.add_argument(
        "--show-sprites", 
        action="store_true",
        help="Display Pokemon sprites in the battle prediction"
    )
    parser.add_argument(
        "--save-image", 
        type=str,
        help="Path to save the battle visualization image"
    )
    
    args = parser.parse_args()
    
    # Set default model path if not provided
    if args.model_path is None:
        if args.format == "vgc":
            args.model_path = "models/vgc_model.pkl"
        else:
            args.model_path = "models/standard_model.pkl"
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        print("Please train the model first using train.py")
        return
    
    # Make prediction
    if args.format == "vgc":
        # Ensure we have at least 4 Pokemon for VGC format
        if len(args.team1) < 4 or len(args.team2) < 4:
            print("Error: VGC format requires at least 4 Pokemon per team")
            return
        
        result = predict_vgc_battle(args.team1[:4], args.team2[:4], args.model_path)
    else:
        # Ensure we have at least 6 Pokemon for Standard format
        if len(args.team1) < 6 or len(args.team2) < 6:
            print("Error: Standard format requires at least 6 Pokemon per team")
            return
        
        result = predict_standard_battle(args.team1[:6], args.team2[:6], args.model_path)
    
    # Print prediction result
    print("\nPrediction Result:")
    print(f"Team 1: {', '.join([p['name'] for p in result['team1']])}")
    print(f"Team 2: {', '.join([p['name'] for p in result['team2']])}")
    print(f"Predicted Winner: {result['predicted_winner']}")
    print(f"Team 1 Win Probability: {result['team1_win_probability']:.2f}")
    print(f"Team 2 Win Probability: {result['team2_win_probability']:.2f}")
    print(f"Confidence: {result['confidence']:.2f}")
    
    print("\nTop Contributing Features:")
    for feature, value in list(result['top_features'].items())[:5]:
        print(f"  {feature}: {value:.4f}")
    
    # Save result to file if output path is provided
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nPrediction saved to {args.output}")
    
    # Display battle visualization with Pokemon sprites if requested
    if args.show_sprites or args.save_image:
        team1_ids = [p['id'] for p in result['team1']]
        team2_ids = [p['id'] for p in result['team2']]
        team1_names = [p['name'] for p in result['team1']]
        team2_names = [p['name'] for p in result['team2']]
        
        print("\nGenerating battle visualization with Pokémon sprites...")
        battle_fig = display_battle(team1_ids, team1_names, team2_ids, team2_names, result)
        
        # Save the visualization if requested
        if args.save_image:
            plt.savefig(args.save_image, dpi=300, bbox_inches='tight')
            print(f"Battle visualization saved to {args.save_image}")
        
        # Show the visualization if requested
        if args.show_sprites:
            plt.show()

if __name__ == "__main__":
    main() 