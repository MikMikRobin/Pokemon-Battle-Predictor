import pandas as pd
import numpy as np

# Load data
print("Loading data...")
pokemon_df = pd.read_csv('pokemon.csv')
teams_df = pd.read_csv('pokemon_id_each_team.csv')
combats_df = pd.read_csv('combats.csv')
team_combat_df = pd.read_csv('team_combat.csv')

# Analyze Pokemon IDs
print("\nPokemon dataset analysis:")
print(f"Number of Pokemon: {len(pokemon_df)}")
print(f"Min Pokemon ID: {pokemon_df['#'].min()}")
print(f"Max Pokemon ID: {pokemon_df['#'].max()}")
print(f"Pokemon ID range: {sorted(pokemon_df['#'].unique())[:5]}...{sorted(pokemon_df['#'].unique())[-5:]}")

# Analyze team Pokemon IDs
print("\nTeam Pokemon IDs analysis:")
team_ids = []
for col in teams_df.columns[1:]:  # Skip the first column which is team ID
    team_ids.extend(teams_df[col].dropna().astype(int).tolist())

team_ids = [int(x) for x in team_ids]
unique_team_ids = set(team_ids)
print(f"Number of Pokemon IDs in teams: {len(team_ids)}")
print(f"Number of unique Pokemon IDs in teams: {len(unique_team_ids)}")
print(f"Min Pokemon ID in teams: {min(team_ids)}")
print(f"Max Pokemon ID in teams: {max(team_ids)}")

# Check for missing Pokemon IDs
pokemon_ids = set(pokemon_df['#'])
missing_ids = [id for id in unique_team_ids if id not in pokemon_ids]
print(f"\nMissing Pokemon IDs in teams:")
print(f"Number of missing IDs: {len(missing_ids)}")
if missing_ids:
    print(f"First 20 missing IDs: {sorted(missing_ids)[:20]}")
    
# Check team combat data
print("\nTeam combat data analysis:")
print(f"Number of team combats: {len(team_combat_df)}")
team_ids_in_combat = set(team_combat_df['first']).union(set(team_combat_df['second']))
print(f"Number of unique team IDs in combats: {len(team_ids_in_combat)}")
team_ids_in_teams = set(teams_df['#'])
print(f"Number of team IDs in team data: {len(team_ids_in_teams)}")

# Check for missing team IDs
missing_team_ids = [id for id in team_ids_in_combat if id not in team_ids_in_teams]
print(f"Number of missing team IDs: {len(missing_team_ids)}")
if missing_team_ids:
    print(f"First 10 missing team IDs: {sorted(missing_team_ids)[:10]}")

# Check for NaN values in feature engineering
print("\nChecking for potential NaN sources in feature engineering:")
pokemon_with_empty_types = pokemon_df[pokemon_df['Type 1'].isna() | (pokemon_df['Type 1'] == '')]
print(f"Pokemon with empty Type 1: {len(pokemon_with_empty_types)}")
if len(pokemon_with_empty_types) > 0:
    print(pokemon_with_empty_types.head())

# Check if all team Pokemon IDs exist in the Pokemon dataset
valid_team_count = 0
invalid_team_count = 0
for _, row in teams_df.iterrows():
    team_id = row['#']
    pokemon_ids = [int(pid) for pid in row.values[1:] if not pd.isna(pid)]
    valid_pokemon = [pid for pid in pokemon_ids if pid in pokemon_df['#'].values]
    if len(valid_pokemon) == len(pokemon_ids):
        valid_team_count += 1
    else:
        invalid_team_count += 1

print(f"\nTeam validity:")
print(f"Valid teams (all Pokemon exist): {valid_team_count}")
print(f"Invalid teams (some Pokemon missing): {invalid_team_count}")
print(f"Percentage of valid teams: {valid_team_count / len(teams_df) * 100:.2f}%")

# Check individual combat data for potential issues
print("\nIndividual combat data analysis:")
print(f"Number of individual combats: {len(combats_df)}")
combat_pokemon_ids = set(combats_df['First_pokemon']).union(set(combats_df['Second_pokemon']))
print(f"Number of unique Pokemon IDs in combats: {len(combat_pokemon_ids)}")
print(f"Min Pokemon ID in combats: {min(combat_pokemon_ids)}")
print(f"Max Pokemon ID in combats: {max(combat_pokemon_ids)}")

# Check for Pokemon IDs in combats that don't exist in the Pokemon dataset
missing_combat_ids = [id for id in combat_pokemon_ids if id not in pokemon_ids]
print(f"Number of missing Pokemon IDs in combats: {len(missing_combat_ids)}")
if missing_combat_ids:
    print(f"First 10 missing combat Pokemon IDs: {sorted(missing_combat_ids)[:10]}")

# Check for specific examples of Pokemon IDs that cause NaN values
print("\nAnalyzing specific examples that might cause NaN values:")
for team_id in range(1, 6):  # Check first 5 teams
    team_row = teams_df[teams_df['#'] == team_id]
    if not team_row.empty:
        print(f"\nTeam {team_id} Pokemon IDs:")
        pokemon_ids = [int(pid) for pid in team_row.iloc[0, 1:] if not pd.isna(pid)]
        print(pokemon_ids)
        
        # Check if these Pokemon exist
        for pid in pokemon_ids:
            pokemon_exists = pid in pokemon_df['#'].values
            print(f"Pokemon ID {pid} exists: {pokemon_exists}")
            
            if not pokemon_exists:
                print(f"  This Pokemon ID doesn't exist in the dataset!")
            else:
                # Print the Pokemon's stats
                pokemon = pokemon_df[pokemon_df['#'] == pid].iloc[0]
                print(f"  Name: {pokemon['Name']}")
                print(f"  Types: {pokemon['Type 1']}/{pokemon['Type 2']}")
                print(f"  Stats: HP={pokemon['HP']}, Atk={pokemon['Attack']}, Def={pokemon['Defense']}, Spd={pokemon['Speed']}")

# Check for team battles with NaN values
print("\nAnalyzing team battles for potential NaN values:")
for _, row in team_combat_df.iloc[:5].iterrows():
    first_team = row['first']
    second_team = row['second']
    print(f"\nBattle: Team {first_team} vs Team {second_team}")
    
    # Check if these teams exist
    first_team_exists = first_team in teams_df['#'].values
    second_team_exists = second_team in teams_df['#'].values
    print(f"Team {first_team} exists: {first_team_exists}")
    print(f"Team {second_team} exists: {second_team_exists}")
    
    if first_team_exists and second_team_exists:
        # Get Pokemon IDs for each team
        first_team_pokemon = teams_df[teams_df['#'] == first_team].iloc[0, 1:].values
        second_team_pokemon = teams_df[teams_df['#'] == second_team].iloc[0, 1:].values
        
        # Filter out any NaN values
        first_team_pokemon = [int(pid) for pid in first_team_pokemon if not pd.isna(pid)]
        second_team_pokemon = [int(pid) for pid in second_team_pokemon if not pd.isna(pid)]
        
        print(f"Team {first_team} Pokemon: {first_team_pokemon}")
        print(f"Team {second_team} Pokemon: {second_team_pokemon}")
        
        # Check if all Pokemon exist
        first_team_valid = all(pid in pokemon_df['#'].values for pid in first_team_pokemon)
        second_team_valid = all(pid in pokemon_df['#'].values for pid in second_team_pokemon)
        print(f"All Team {first_team} Pokemon exist: {first_team_valid}")
        print(f"All Team {second_team} Pokemon exist: {second_team_valid}")
        
        if not first_team_valid:
            missing = [pid for pid in first_team_pokemon if pid not in pokemon_df['#'].values]
            print(f"  Missing Pokemon IDs in Team {first_team}: {missing}")
        
        if not second_team_valid:
            missing = [pid for pid in second_team_pokemon if pid not in pokemon_df['#'].values]
            print(f"  Missing Pokemon IDs in Team {second_team}: {missing}") 