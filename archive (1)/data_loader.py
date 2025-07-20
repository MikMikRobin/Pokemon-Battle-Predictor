"""
Data loading and preprocessing module for Pokemon Battle Predictor.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import config

class DataLoader:
    def __init__(self):
        """Initialize the data loader."""
        self.pokemon_df = None
        self.combats_df = None
        self.team_combat_df = None
        self.team_pokemon_df = None
        self.type_encoder = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load all data files."""
        self.pokemon_df = pd.read_csv(config.POKEMON_CSV)
        self.combats_df = pd.read_csv(config.COMBATS_CSV)
        self.team_combat_df = pd.read_csv(config.TEAM_COMBAT_CSV)
        self.team_pokemon_df = pd.read_csv(config.POKEMON_TEAM_CSV)
        
        # Clean up column names if needed
        self.pokemon_df.columns = self.pokemon_df.columns.str.strip()
        
        # Handle missing values
        self.pokemon_df = self.pokemon_df.fillna('')
        
        return self
    
    def preprocess_pokemon_data(self):
        """Preprocess Pokemon data."""
        # Encode Pokemon types
        self.type_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        type_features = self.type_encoder.fit_transform(
            self.pokemon_df[['Type 1', 'Type 2']].fillna('None')
        )
        
        # Create type feature dataframe
        type_columns = [f'type_{col}' for col in self.type_encoder.get_feature_names_out()]
        type_df = pd.DataFrame(type_features, columns=type_columns)
        
        # Extract numerical features
        numerical_features = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
        numerical_df = self.pokemon_df[numerical_features].copy()
        
        # Scale numerical features
        scaled_features = self.scaler.fit_transform(numerical_df)
        scaled_df = pd.DataFrame(scaled_features, columns=numerical_features)
        
        # Add legendary status as binary feature
        legendary_df = pd.DataFrame({'is_legendary': self.pokemon_df['Legendary'].map({'True': 1, 'False': 0})})
        
        # Combine all features
        self.processed_pokemon_df = pd.concat([
            self.pokemon_df[['#', 'Name']], 
            scaled_df, 
            type_df, 
            legendary_df
        ], axis=1)
        
        return self
    
    def get_pokemon_by_id(self, pokemon_id):
        """Get Pokemon data by ID."""
        return self.processed_pokemon_df[self.processed_pokemon_df['#'] == pokemon_id].iloc[0]
    
    def prepare_1v1_battle_data(self):
        """Prepare data for 1v1 battle prediction."""
        battle_data = []
        
        for _, row in self.combats_df.iterrows():
            first_pokemon = self.get_pokemon_by_id(row['First_pokemon'])
            second_pokemon = self.get_pokemon_by_id(row['Second_pokemon'])
            winner = 1 if row['Winner'] == row['First_pokemon'] else 0
            
            # Exclude name and ID from features
            first_features = first_pokemon.drop(['#', 'Name']).values
            second_features = second_pokemon.drop(['#', 'Name']).values
            
            # Create feature vector: [first_pokemon_features, second_pokemon_features]
            features = np.concatenate([first_features, second_features])
            
            battle_data.append((features, winner))
        
        # Convert to numpy arrays
        X = np.array([data[0] for data in battle_data])
        y = np.array([data[1] for data in battle_data])
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED
        )
        
        return X_train, X_test, y_train, y_test
    
    def prepare_team_battle_data(self, battle_format=config.BattleFormat.VGC_2V2):
        """
        Prepare data for team battle prediction.
        
        Args:
            battle_format: Format of the battle (VGC_2V2 or STANDARD_6V6)
        
        Returns:
            Training and testing data for team battles
        """
        team_battle_data = []
        
        for _, row in self.team_combat_df.iterrows():
            first_team_id = row['first']
            second_team_id = row['second']
            winner = 1 if row['winner'] == first_team_id else 0
            
            # Get Pokemon IDs for each team
            first_team_pokemon = self.team_pokemon_df[self.team_pokemon_df['#'] == first_team_id].iloc[0, 1:].values
            second_team_pokemon = self.team_pokemon_df[self.team_pokemon_df['#'] == second_team_id].iloc[0, 1:].values
            
            # Filter out any NaN values
            first_team_pokemon = [int(pid) for pid in first_team_pokemon if not pd.isna(pid)]
            second_team_pokemon = [int(pid) for pid in second_team_pokemon if not pd.isna(pid)]
            
            # Limit team size based on battle format
            if battle_format == config.BattleFormat.VGC_2V2:
                first_team_pokemon = first_team_pokemon[:4]  # VGC format: choose from 4
                second_team_pokemon = second_team_pokemon[:4]
            else:  # STANDARD_6V6
                first_team_pokemon = first_team_pokemon[:6]
                second_team_pokemon = second_team_pokemon[:6]
            
            # Get team features
            first_team_features = self._get_team_features(first_team_pokemon)
            second_team_features = self._get_team_features(second_team_pokemon)
            
            # Create feature vector: [first_team_features, second_team_features]
            features = np.concatenate([first_team_features, second_team_features])
            
            team_battle_data.append((features, winner))
        
        # Convert to numpy arrays
        X = np.array([data[0] for data in team_battle_data])
        y = np.array([data[1] for data in team_battle_data])
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED
        )
        
        return X_train, X_test, y_train, y_test
    
    def _get_team_features(self, pokemon_ids):
        """
        Generate aggregated features for a team of Pokemon.
        
        Args:
            pokemon_ids: List of Pokemon IDs in the team
            
        Returns:
            Aggregated team features
        """
        # Get individual Pokemon features
        pokemon_features = []
        for pid in pokemon_ids:
            pokemon = self.get_pokemon_by_id(pid)
            # Exclude name and ID
            features = pokemon.drop(['#', 'Name']).values
            pokemon_features.append(features)
        
        # Convert to numpy array
        pokemon_features = np.array(pokemon_features)
        
        # Calculate aggregated features
        team_features = []
        
        # Sum of all stats
        team_features.extend(np.sum(pokemon_features, axis=0))
        
        # Mean of all stats
        team_features.extend(np.mean(pokemon_features, axis=0))
        
        # Max of all stats
        team_features.extend(np.max(pokemon_features, axis=0))
        
        # Min of all stats
        team_features.extend(np.min(pokemon_features, axis=0))
        
        # Standard deviation of stats
        team_features.extend(np.std(pokemon_features, axis=0))
    
        return np.array(team_features) 