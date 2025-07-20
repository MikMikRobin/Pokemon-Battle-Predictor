"""
Test script to verify prediction fixes.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from predict import predict_vgc_battle
from data_loader import DataLoader
from pokemon_utils import display_battle

# Load Pokemon data
print("Loading Pokemon data...")
data_loader = DataLoader()
data_loader.load_data()

# Create two teams with clear differences
# Team 1: Strong legendary Pokemon
team1_names = ["Mewtwo", "Groudon", "Kyogre", "Rayquaza"]
team1_ids = []

# Team 2: Weak basic Pokemon
team2_names = ["Rattata", "Pidgey", "Caterpie", "Weedle"]
team2_ids = []

# Find Pokemon IDs by name
for name in team1_names:
    pokemon = data_loader.pokemon_df[data_loader.pokemon_df['Name'] == name]
    if not pokemon.empty:
        team1_ids.append(int(pokemon.iloc[0]['#']))
    else:
        print(f"Warning: {name} not found. Using a random legendary Pokemon instead.")
        legendary = data_loader.pokemon_df[data_loader.pokemon_df['Legendary'] == 'True']
        if not legendary.empty:
            random_legendary = legendary.sample(1)
            team1_ids.append(int(random_legendary.iloc[0]['#']))
            print(f"Using {random_legendary.iloc[0]['Name']} instead.")

for name in team2_names:
    pokemon = data_loader.pokemon_df[data_loader.pokemon_df['Name'] == name]
    if not pokemon.empty:
        team2_ids.append(int(pokemon.iloc[0]['#']))
    else:
        print(f"Warning: {name} not found. Using a random non-legendary Pokemon instead.")
        non_legendary = data_loader.pokemon_df[data_loader.pokemon_df['Legendary'] == 'False']
        if not non_legendary.empty:
            random_non_legendary = non_legendary[non_legendary['HP'] < 50].sample(1)
            team2_ids.append(int(random_non_legendary.iloc[0]['#']))
            print(f"Using {random_non_legendary.iloc[0]['Name']} instead.")

# Print team information
print("\nTeam 1 (Strong):")
for pid in team1_ids:
    pokemon = data_loader.pokemon_df[data_loader.pokemon_df['#'] == pid].iloc[0]
    print(f"  {pokemon['Name']} (#{pid}): HP={pokemon['HP']}, Attack={pokemon['Attack']}, Defense={pokemon['Defense']}, Speed={pokemon['Speed']}")

print("\nTeam 2 (Weak):")
for pid in team2_ids:
    pokemon = data_loader.pokemon_df[data_loader.pokemon_df['#'] == pid].iloc[0]
    print(f"  {pokemon['Name']} (#{pid}): HP={pokemon['HP']}, Attack={pokemon['Attack']}, Defense={pokemon['Defense']}, Speed={pokemon['Speed']}")

# Make prediction
print("\nPredicting battle outcome...")
result = predict_vgc_battle(team1_ids, team2_ids)

# Print prediction result
print("\nPrediction Result:")
print(f"Team 1: {', '.join([p['name'] for p in result['team1']])}")
print(f"Team 2: {', '.join([p['name'] for p in result['team2']])}")
print(f"Predicted Winner: {result['predicted_winner']}")
print(f"Team 1 Win Probability: {result['team1_win_probability']:.4f}")
print(f"Team 2 Win Probability: {result['team2_win_probability']:.4f}")
print(f"Confidence: {result['confidence']:.4f}")

# Print top features
print("\nTop Contributing Factors:")
for feature, value in list(result['top_features'].items())[:10]:
    print(f"  {feature}: {value:.4f}")

# Check for consistency
team1_favoring_features = 0
team2_favoring_features = 0
team1_total_importance = 0
team2_total_importance = 0

for feature, value in result['top_features'].items():
    if "Team 1" in feature and value > 0:
        team1_favoring_features += 1
        team1_total_importance += value
    elif "Team 2" in feature and value > 0:
        team2_favoring_features += 1
        team2_total_importance += value
    elif "Team 1" in feature and value < 0:
        team2_favoring_features += 1
        team2_total_importance += abs(value)
    elif "Team 2" in feature and value < 0:
        team1_favoring_features += 1
        team1_total_importance += abs(value)

print("\nConsistency Check:")
print(f"  Features favoring Team 1: {team1_favoring_features} (total importance: {team1_total_importance:.4f})")
print(f"  Features favoring Team 2: {team2_favoring_features} (total importance: {team2_total_importance:.4f})")

if (team1_favoring_features > team2_favoring_features and result['predicted_winner'] == 'Team 1') or \
   (team2_favoring_features > team1_favoring_features and result['predicted_winner'] == 'Team 2'):
    print("  ✅ Prediction is consistent with feature importance")
else:
    print("  ❌ Prediction is inconsistent with feature importance")

# Check if prediction makes sense based on team stats
team1_total_stats = 0
team2_total_stats = 0

for pid in team1_ids:
    pokemon = data_loader.pokemon_df[data_loader.pokemon_df['#'] == pid].iloc[0]
    team1_total_stats += pokemon['HP'] + pokemon['Attack'] + pokemon['Defense'] + pokemon['Sp. Atk'] + pokemon['Sp. Def'] + pokemon['Speed']

for pid in team2_ids:
    pokemon = data_loader.pokemon_df[data_loader.pokemon_df['#'] == pid].iloc[0]
    team2_total_stats += pokemon['HP'] + pokemon['Attack'] + pokemon['Defense'] + pokemon['Sp. Atk'] + pokemon['Sp. Def'] + pokemon['Speed']

print(f"\nTeam 1 Total Stats: {team1_total_stats}")
print(f"Team 2 Total Stats: {team2_total_stats}")

if (team1_total_stats > team2_total_stats and result['predicted_winner'] == 'Team 1') or \
   (team2_total_stats > team1_total_stats and result['predicted_winner'] == 'Team 2'):
    print("  ✅ Prediction makes sense based on team stats")
else:
    print("  ❌ Prediction does not make sense based on team stats")

# Display battle with Pokémon sprites
print("\nGenerating battle visualization with Pokémon sprites...")
team1_names = [p['name'] for p in result['team1']]
team2_names = [p['name'] for p in result['team2']]

# Create and display the battle visualization
battle_fig = display_battle(team1_ids, team1_names, team2_ids, team2_names, result)
plt.savefig('battle_prediction.png', dpi=300, bbox_inches='tight')
print("Battle visualization saved as 'battle_prediction.png'")

# Show the figure
plt.show() 