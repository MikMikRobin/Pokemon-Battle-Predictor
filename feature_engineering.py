"""
Feature engineering module for Pokemon Battle Predictor.
Creates advanced features for battle prediction.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import config

class FeatureEngineer:
    def __init__(self, pokemon_df):
        """
        Initialize the feature engineer.
        
        Args:
            pokemon_df: Processed Pokemon dataframe
        """
        self.pokemon_df = pokemon_df
        self.type_effectiveness = self._build_type_effectiveness_matrix()
        
    def _build_type_effectiveness_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Build a type effectiveness matrix.
        
        Returns:
            Dictionary mapping attack types to defense types with effectiveness multipliers
        """
        # Pokemon type effectiveness chart (simplified)
        # 2.0 = super effective, 0.5 = not very effective, 0.0 = no effect
        effectiveness = {
            'Normal': {'Rock': 0.5, 'Ghost': 0.0, 'Steel': 0.5},
            'Fire': {'Fire': 0.5, 'Water': 0.5, 'Grass': 2.0, 'Ice': 2.0, 'Bug': 2.0, 'Rock': 0.5, 'Dragon': 0.5, 'Steel': 2.0},
            'Water': {'Fire': 2.0, 'Water': 0.5, 'Grass': 0.5, 'Ground': 2.0, 'Rock': 2.0, 'Dragon': 0.5},
            'Electric': {'Water': 2.0, 'Electric': 0.5, 'Grass': 0.5, 'Ground': 0.0, 'Flying': 2.0, 'Dragon': 0.5},
            'Grass': {'Fire': 0.5, 'Water': 2.0, 'Grass': 0.5, 'Poison': 0.5, 'Ground': 2.0, 'Flying': 0.5, 'Bug': 0.5, 'Rock': 2.0, 'Dragon': 0.5, 'Steel': 0.5},
            'Ice': {'Fire': 0.5, 'Water': 0.5, 'Grass': 2.0, 'Ice': 0.5, 'Ground': 2.0, 'Flying': 2.0, 'Dragon': 2.0, 'Steel': 0.5},
            'Fighting': {'Normal': 2.0, 'Ice': 2.0, 'Poison': 0.5, 'Flying': 0.5, 'Psychic': 0.5, 'Bug': 0.5, 'Rock': 2.0, 'Ghost': 0.0, 'Dark': 2.0, 'Steel': 2.0, 'Fairy': 0.5},
            'Poison': {'Grass': 2.0, 'Poison': 0.5, 'Ground': 0.5, 'Rock': 0.5, 'Ghost': 0.5, 'Steel': 0.0, 'Fairy': 2.0},
            'Ground': {'Fire': 2.0, 'Electric': 2.0, 'Grass': 0.5, 'Poison': 2.0, 'Flying': 0.0, 'Bug': 0.5, 'Rock': 2.0, 'Steel': 2.0},
            'Flying': {'Electric': 0.5, 'Grass': 2.0, 'Fighting': 2.0, 'Bug': 2.0, 'Rock': 0.5, 'Steel': 0.5},
            'Psychic': {'Fighting': 2.0, 'Poison': 2.0, 'Psychic': 0.5, 'Dark': 0.0, 'Steel': 0.5},
            'Bug': {'Fire': 0.5, 'Grass': 2.0, 'Fighting': 0.5, 'Poison': 0.5, 'Flying': 0.5, 'Psychic': 2.0, 'Ghost': 0.5, 'Dark': 2.0, 'Steel': 0.5, 'Fairy': 0.5},
            'Rock': {'Fire': 2.0, 'Ice': 2.0, 'Fighting': 0.5, 'Ground': 0.5, 'Flying': 2.0, 'Bug': 2.0, 'Steel': 0.5},
            'Ghost': {'Normal': 0.0, 'Psychic': 2.0, 'Ghost': 2.0, 'Dark': 0.5},
            'Dragon': {'Dragon': 2.0, 'Steel': 0.5, 'Fairy': 0.0},
            'Dark': {'Fighting': 0.5, 'Psychic': 2.0, 'Ghost': 2.0, 'Dark': 0.5, 'Fairy': 0.5},
            'Steel': {'Fire': 0.5, 'Water': 0.5, 'Electric': 0.5, 'Ice': 2.0, 'Rock': 2.0, 'Steel': 0.5, 'Fairy': 2.0},
            'Fairy': {'Fire': 0.5, 'Fighting': 2.0, 'Poison': 0.5, 'Dragon': 2.0, 'Dark': 2.0, 'Steel': 0.5}
        }
        
        return effectiveness
    
    def calculate_type_effectiveness(self, attacker_type: str, defender_types: List[str]) -> float:
        """
        Calculate type effectiveness multiplier.
        
        Args:
            attacker_type: Type of the attacking move
            defender_types: Types of the defending Pokemon
            
        Returns:
            Effectiveness multiplier
        """
        # Handle empty or NaN attacker type
        if attacker_type is None or pd.isna(attacker_type) or attacker_type == '' or attacker_type == 'None':
            return 1.0
            
        multiplier = 1.0
        for defender_type in defender_types:
            # Handle empty or NaN defender type
            if defender_type is None or pd.isna(defender_type) or defender_type == '' or defender_type == 'None':
                continue
                
            if attacker_type in self.type_effectiveness and defender_type in self.type_effectiveness[attacker_type]:
                multiplier *= self.type_effectiveness[attacker_type][defender_type]
                
        return multiplier
    
    def create_matchup_features(self, first_pokemon_id: int, second_pokemon_id: int) -> Dict[str, float]:
        """
        Create matchup features for two Pokemon.
        
        Args:
            first_pokemon_id: ID of the first Pokemon
            second_pokemon_id: ID of the second Pokemon
            
        Returns:
            Dictionary of matchup features
        """
        first_pokemon = self.pokemon_df[self.pokemon_df['#'] == first_pokemon_id].iloc[0]
        second_pokemon = self.pokemon_df[self.pokemon_df['#'] == second_pokemon_id].iloc[0]
        
        first_types = [first_pokemon['Type 1'], first_pokemon['Type 2']]
        second_types = [second_pokemon['Type 1'], second_pokemon['Type 2']]
        
        # Calculate type effectiveness in both directions
        first_to_second_effectiveness = 1.0
        for attack_type in first_types:
            if attack_type:  # Skip empty types
                first_to_second_effectiveness *= self.calculate_type_effectiveness(attack_type, second_types)
                
        second_to_first_effectiveness = 1.0
        for attack_type in second_types:
            if attack_type:  # Skip empty types
                second_to_first_effectiveness *= self.calculate_type_effectiveness(attack_type, first_types)
        
        # Calculate stat differentials
        stat_diff = {}
        for stat in ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']:
            stat_diff[f'{stat}_diff'] = first_pokemon[stat] - second_pokemon[stat]
            stat_diff[f'{stat}_ratio'] = first_pokemon[stat] / max(1, second_pokemon[stat])
        
        # Calculate speed advantage
        speed_advantage = 1 if first_pokemon['Speed'] > second_pokemon['Speed'] else -1 if first_pokemon['Speed'] < second_pokemon['Speed'] else 0
        
        # Create final feature dictionary
        features = {
            'type_effectiveness_1_to_2': first_to_second_effectiveness,
            'type_effectiveness_2_to_1': second_to_first_effectiveness,
            'type_advantage': first_to_second_effectiveness - second_to_first_effectiveness,
            'speed_advantage': speed_advantage,
            'is_legendary_1': 1 if first_pokemon['Legendary'] == 'True' else 0,
            'is_legendary_2': 1 if second_pokemon['Legendary'] == 'True' else 0,
            **stat_diff
        }
        
        return features
    
    def create_team_matchup_features(self, team1_ids: List[int], team2_ids: List[int]) -> Dict[str, float]:
        """
        Create matchup features for two teams.
        
        Args:
            team1_ids: IDs of Pokemon in the first team
            team2_ids: IDs of Pokemon in the second team
            
        Returns:
            Dictionary of team matchup features
        """
        # Get individual Pokemon stats
        team1_stats = [self.pokemon_df[self.pokemon_df['#'] == pid].iloc[0] for pid in team1_ids]
        team2_stats = [self.pokemon_df[self.pokemon_df['#'] == pid].iloc[0] for pid in team2_ids]
        
        # Calculate team-level stats
        team1_avg_stats = {stat: np.mean([p[stat] for p in team1_stats]) for stat in ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']}
        team2_avg_stats = {stat: np.mean([p[stat] for p in team2_stats]) for stat in ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']}
        
        team1_max_stats = {stat: np.max([p[stat] for p in team1_stats]) for stat in ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']}
        team2_max_stats = {stat: np.max([p[stat] for p in team2_stats]) for stat in ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']}
        
        # Calculate stat differentials
        avg_stat_diff = {f'avg_{stat}_diff': team1_avg_stats[stat] - team2_avg_stats[stat] for stat in team1_avg_stats}
        max_stat_diff = {f'max_{stat}_diff': team1_max_stats[stat] - team2_max_stats[stat] for stat in team1_max_stats}
        
        # Calculate type effectiveness matrix for all matchups
        type_effectiveness_matrix = np.zeros((len(team1_ids), len(team2_ids)))
        for i, pid1 in enumerate(team1_ids):
            for j, pid2 in enumerate(team2_ids):
                matchup = self.create_matchup_features(pid1, pid2)
                type_effectiveness_matrix[i, j] = matchup['type_effectiveness_1_to_2']
        
        # Calculate average and best type matchups
        avg_type_effectiveness = np.mean(type_effectiveness_matrix)
        max_type_effectiveness = np.max(type_effectiveness_matrix)
        
        # Count legendary Pokemon
        team1_legendary_count = sum(1 for p in team1_stats if p['Legendary'] == 'True')
        team2_legendary_count = sum(1 for p in team2_stats if p['Legendary'] == 'True')
        
        # Calculate speed control (how many Pokemon are faster than opponent's average)
        team1_speed_control = sum(1 for p in team1_stats if p['Speed'] > team2_avg_stats['Speed'])
        team2_speed_control = sum(1 for p in team2_stats if p['Speed'] > team1_avg_stats['Speed'])
        
        # Create final feature dictionary
        features = {
            'avg_type_effectiveness': avg_type_effectiveness,
            'max_type_effectiveness': max_type_effectiveness,
            'legendary_diff': team1_legendary_count - team2_legendary_count,
            'speed_control_diff': team1_speed_control - team2_speed_control,
            **avg_stat_diff,
            **max_stat_diff
        }
        
        return features
    
    def create_vgc_features(self, team1_ids: List[int], team2_ids: List[int]) -> np.ndarray:
        """
        Create features specifically for VGC format (2v2 battles).
        
        Args:
            team1_ids: IDs of Pokemon in the first team (4 Pokemon)
            team2_ids: IDs of Pokemon in the second team (4 Pokemon)
            
        Returns:
            Feature vector for VGC battle
        """
        # In VGC, players bring 4 Pokemon and select 2 for battle
        # We'll create features for all possible 2v2 combinations
        
        # Ensure we have 4 Pokemon per team
        team1_ids = team1_ids[:4]
        team2_ids = team2_ids[:4]
        
        # Basic team stats
        team_features = self.create_team_matchup_features(team1_ids, team2_ids)
        
        # Calculate features for best possible 2v2 matchups
        best_matchup_score = 0
        best_pair_features = {}
        
        # For each possible pair from team 1
        for i in range(len(team1_ids)):
            for j in range(i+1, len(team1_ids)):
                pair1 = [team1_ids[i], team1_ids[j]]
                
                # For each possible pair from team 2
                for k in range(len(team2_ids)):
                    for l in range(k+1, len(team2_ids)):
                        pair2 = [team2_ids[k], team2_ids[l]]
                        
                        # Calculate matchup score
                        matchup = self.create_team_matchup_features(pair1, pair2)
                        matchup_score = (
                            matchup['avg_type_effectiveness'] + 
                            matchup['speed_control_diff'] + 
                            matchup['avg_Attack_diff'] / 100 + 
                            matchup['avg_Defense_diff'] / 100
                        )
                        
                        # Keep track of best matchup
                        if matchup_score > best_matchup_score:
                            best_matchup_score = matchup_score
                            best_pair_features = matchup
        
        # Combine all features
        all_features = {
            **team_features,
            'best_matchup_score': best_matchup_score,
            **{f'best_{k}': v for k, v in best_pair_features.items()}
        }
        
        # Convert to numpy array
        return np.array(list(all_features.values()))
    
    def create_standard_features(self, team1_ids: List[int], team2_ids: List[int]) -> np.ndarray:
        """
        Create features specifically for Standard format (6v6 battles).
        
        Args:
            team1_ids: IDs of Pokemon in the first team (6 Pokemon)
            team2_ids: IDs of Pokemon in the second team (6 Pokemon)
            
        Returns:
            Feature vector for Standard battle
        """
        # Ensure we have 6 Pokemon per team
        team1_ids = team1_ids[:6]
        team2_ids = team2_ids[:6]
        
        # Basic team stats
        team_features = self.create_team_matchup_features(team1_ids, team2_ids)
        
        # Calculate type coverage (how many types each team can hit super effectively)
        team1_coverage = self._calculate_type_coverage(team1_ids)
        team2_coverage = self._calculate_type_coverage(team2_ids)
        
        # Calculate team balance metrics
        team1_balance = self._calculate_team_balance(team1_ids)
        team2_balance = self._calculate_team_balance(team2_ids)
        
        # Combine all features
        all_features = {
            **team_features,
            'type_coverage_diff': team1_coverage - team2_coverage,
            'team_balance_diff': team1_balance - team2_balance
        }
        
        # Convert to numpy array
        return np.array(list(all_features.values()))
    
    def _calculate_type_coverage(self, team_ids: List[int]) -> float:
        """
        Calculate how many types a team can hit super effectively.
        
        Args:
            team_ids: IDs of Pokemon in the team
            
        Returns:
            Type coverage score
        """
        all_types = ['Normal', 'Fire', 'Water', 'Electric', 'Grass', 'Ice', 
                     'Fighting', 'Poison', 'Ground', 'Flying', 'Psychic', 'Bug', 
                     'Rock', 'Ghost', 'Dragon', 'Dark', 'Steel', 'Fairy']
        
        coverage_count = 0
        
        for target_type in all_types:
            can_hit_super_effective = False
            
            for pid in team_ids:
                pokemon = self.pokemon_df[self.pokemon_df['#'] == pid].iloc[0]
                attack_types = [pokemon['Type 1'], pokemon['Type 2']]
                
                for attack_type in attack_types:
                    if attack_type and attack_type in self.type_effectiveness:
                        if target_type in self.type_effectiveness[attack_type] and self.type_effectiveness[attack_type][target_type] > 1.0:
                            can_hit_super_effective = True
                            break
                
                if can_hit_super_effective:
                    break
            
            if can_hit_super_effective:
                coverage_count += 1
        
        return coverage_count / len(all_types)
    
    def _calculate_team_balance(self, team_ids: List[int]) -> float:
        """
        Calculate team balance based on stat distribution.
        
        Args:
            team_ids: IDs of Pokemon in the team
            
        Returns:
            Team balance score (higher is more balanced)
        """
        stats = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
        team_stats = []
        
        for pid in team_ids:
            pokemon = self.pokemon_df[self.pokemon_df['#'] == pid].iloc[0]
            team_stats.append([pokemon[stat] for stat in stats])
        
        team_stats = np.array(team_stats)
        
        # Calculate coefficient of variation for each stat across the team
        stat_means = np.mean(team_stats, axis=0)
        stat_stds = np.std(team_stats, axis=0)
        stat_cvs = stat_stds / stat_means
        
        # Lower CV means more balanced team (less variation)
        # Convert to a score where higher is better
        balance_score = 1 / (np.mean(stat_cvs) + 0.1)
        
        return balance_score 