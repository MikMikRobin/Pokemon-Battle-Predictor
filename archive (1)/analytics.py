"""
Analytics module for Pokemon Battle Predictor.
Provides data visualization and insights.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import pickle
import json
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

import config
from data_loader import DataLoader
from feature_engineering import FeatureEngineer

class BattleAnalytics:
    """Class for analyzing Pokemon battle data and generating insights."""
    
    def __init__(self, data_path="analytics_data.pkl"):
        """
        Initialize the analytics module.
        
        Args:
            data_path: Path to save/load analytics data
        """
        self.data_path = data_path
        self.data_loader = None
        self.feature_engineer = None
        self.pokemon_df = None
        self.combats_df = None
        self.team_combat_df = None
        self.team_pokemon_df = None
        
        # Analytics results
        self.pokemon_winrates = None
        self.stat_correlations = None
        self.type_effectiveness = None
        self.team_winrates_vgc = None
        self.team_winrates_standard = None
        self.stat_importance = None
        
    def load_data(self):
        """Load and preprocess data."""
        print("Loading and preprocessing data...")
        self.data_loader = DataLoader()
        self.data_loader.load_data()
        self.data_loader.preprocess_pokemon_data()
        
        # Store references to dataframes
        self.pokemon_df = self.data_loader.pokemon_df
        self.combats_df = self.data_loader.combats_df
        self.team_combat_df = self.data_loader.team_combat_df
        self.team_pokemon_df = self.data_loader.team_pokemon_df
        
        # Create feature engineer
        self.feature_engineer = FeatureEngineer(self.pokemon_df)
        
        return self
    
    def run_analytics(self, force_recompute=False):
        """
        Run all analytics.
        
        Args:
            force_recompute: If True, recompute all analytics even if saved data exists
        """
        if os.path.exists(self.data_path) and not force_recompute:
            print(f"Loading analytics from {self.data_path}...")
            self.load_analytics()
        else:
            print("Running analytics...")
            self.compute_pokemon_winrates()
            self.compute_stat_correlations()
            self.compute_type_effectiveness()
            self.compute_team_winrates()
            self.compute_stat_importance()
            self.save_analytics()
        
        return self
    
    def compute_pokemon_winrates(self):
        """Compute win rates for individual Pokemon."""
        print("Computing Pokemon win rates...")
        
        # Count wins and total battles for each Pokemon
        pokemon_battles = defaultdict(int)
        pokemon_wins = defaultdict(int)
        
        for _, row in tqdm(self.combats_df.iterrows(), total=len(self.combats_df)):
            first_pokemon = row['First_pokemon']
            second_pokemon = row['Second_pokemon']
            winner = row['Winner']
            
            pokemon_battles[first_pokemon] += 1
            pokemon_battles[second_pokemon] += 1
            
            if winner == first_pokemon:
                pokemon_wins[first_pokemon] += 1
            else:
                pokemon_wins[second_pokemon] += 1
        
        # Calculate win rates
        winrates = {}
        for pokemon_id in pokemon_battles:
            battles = pokemon_battles[pokemon_id]
            wins = pokemon_wins[pokemon_id]
            winrates[pokemon_id] = {
                'pokemon_id': pokemon_id,
                'battles': battles,
                'wins': wins,
                'winrate': wins / battles if battles > 0 else 0
            }
        
        # Convert to DataFrame
        self.pokemon_winrates = pd.DataFrame.from_dict(winrates, orient='index')
        
        # Add Pokemon names
        pokemon_names = {row['#']: row['Name'] for _, row in self.pokemon_df.iterrows()}
        self.pokemon_winrates['name'] = self.pokemon_winrates['pokemon_id'].map(pokemon_names)
        
        # Sort by win rate
        self.pokemon_winrates = self.pokemon_winrates.sort_values('winrate', ascending=False)
    
    def compute_stat_correlations(self):
        """Compute correlations between Pokemon stats and win rates."""
        print("Computing stat correlations with win rates...")
        
        if self.pokemon_winrates is None:
            self.compute_pokemon_winrates()
        
        # Merge Pokemon stats with win rates
        stats_df = self.pokemon_df[['#', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']]
        merged_df = pd.merge(stats_df, self.pokemon_winrates, left_on='#', right_on='pokemon_id')
        
        # Compute correlations
        self.stat_correlations = merged_df[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'winrate']].corr()['winrate'].drop('winrate')
    
    def compute_type_effectiveness(self):
        """Compute win rates by type matchups."""
        print("Computing type effectiveness...")
        
        # Extract types for each Pokemon
        pokemon_types = {
            row['#']: (row['Type 1'], row['Type 2']) 
            for _, row in self.pokemon_df.iterrows()
        }
        
        # Count wins by type matchup
        type_matchups = defaultdict(lambda: {'wins': 0, 'battles': 0})
        
        for _, row in tqdm(self.combats_df.iterrows(), total=len(self.combats_df)):
            first_pokemon = row['First_pokemon']
            second_pokemon = row['Second_pokemon']
            winner = row['Winner']
            
            # Get types
            first_types = pokemon_types.get(first_pokemon, ('Unknown', ''))
            second_types = pokemon_types.get(second_pokemon, ('Unknown', ''))
            
            # Record type matchup result
            if winner == first_pokemon:
                for attack_type in first_types:
                    if attack_type:  # Skip empty types
                        for defense_type in second_types:
                            if defense_type:  # Skip empty types
                                matchup = (attack_type, defense_type)
                                type_matchups[matchup]['wins'] += 1
                                type_matchups[matchup]['battles'] += 1
            else:
                for attack_type in second_types:
                    if attack_type:  # Skip empty types
                        for defense_type in first_types:
                            if defense_type:  # Skip empty types
                                matchup = (attack_type, defense_type)
                                type_matchups[matchup]['wins'] += 1
                                type_matchups[matchup]['battles'] += 1
            
            # Always record the battle count for both directions
            for attack_type in first_types:
                if attack_type:
                    for defense_type in second_types:
                        if defense_type:
                            matchup = (attack_type, defense_type)
                            if matchup not in type_matchups:
                                type_matchups[matchup] = {'wins': 0, 'battles': 1}
                            else:
                                type_matchups[matchup]['battles'] += 1
        
        # Calculate win rates
        self.type_effectiveness = {
            matchup: {
                'attack_type': matchup[0],
                'defense_type': matchup[1],
                'wins': data['wins'],
                'battles': data['battles'],
                'winrate': data['wins'] / data['battles'] if data['battles'] > 0 else 0
            }
            for matchup, data in type_matchups.items()
        }
    
    def compute_team_winrates(self):
        """Compute win rates for teams in both formats."""
        print("Computing team win rates...")
        
        # VGC format (2v2)
        self._compute_format_team_winrates(config.BattleFormat.VGC_2V2)
        
        # Standard format (6v6)
        self._compute_format_team_winrates(config.BattleFormat.STANDARD_6V6)
    
    def _compute_format_team_winrates(self, battle_format):
        """
        Compute win rates for teams in a specific format.
        
        Args:
            battle_format: Format of the battle (VGC_2V2 or STANDARD_6V6)
        """
        # Count wins and total battles for each team
        team_battles = defaultdict(int)
        team_wins = defaultdict(int)
        
        for _, row in tqdm(self.team_combat_df.iterrows(), total=len(self.team_combat_df)):
            first_team_id = row['first']
            second_team_id = row['second']
            winner = row['winner']
            
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
            
            # Create team keys (sorted tuple of Pokemon IDs)
            first_team_key = tuple(sorted(first_team_pokemon))
            second_team_key = tuple(sorted(second_team_pokemon))
            
            # Record battle
            team_battles[first_team_key] += 1
            team_battles[second_team_key] += 1
            
            # Record win
            if winner == first_team_id:
                team_wins[first_team_key] += 1
            else:
                team_wins[second_team_key] += 1
        
        # Calculate win rates
        team_winrates = {}
        for team_key in team_battles:
            battles = team_battles[team_key]
            wins = team_wins.get(team_key, 0)
            
            # Get Pokemon names
            pokemon_names = []
            for pid in team_key:
                name = self.pokemon_df[self.pokemon_df['#'] == pid]['Name'].values
                pokemon_names.append(name[0] if len(name) > 0 else f"Unknown ({pid})")
            
            team_winrates[team_key] = {
                'pokemon_ids': team_key,
                'pokemon_names': pokemon_names,
                'battles': battles,
                'wins': wins,
                'winrate': wins / battles if battles > 0 else 0
            }
        
        # Store results
        if battle_format == config.BattleFormat.VGC_2V2:
            self.team_winrates_vgc = team_winrates
        else:
            self.team_winrates_standard = team_winrates
    
    def compute_stat_importance(self):
        """Compute the importance of different stats for winning battles."""
        print("Computing stat importance...")
        
        # Prepare data for analysis
        battle_data = []
        
        for _, row in tqdm(self.combats_df.iterrows(), total=len(self.combats_df)):
            first_pokemon = self.pokemon_df[self.pokemon_df['#'] == row['First_pokemon']].iloc[0]
            second_pokemon = self.pokemon_df[self.pokemon_df['#'] == row['Second_pokemon']].iloc[0]
            
            # Calculate stat differences
            stat_diffs = {}
            for stat in ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']:
                stat_diffs[f'{stat}_diff'] = first_pokemon[stat] - second_pokemon[stat]
            
            # Record outcome
            outcome = 1 if row['Winner'] == row['First_pokemon'] else 0
            
            battle_data.append({
                'outcome': outcome,
                **stat_diffs
            })
        
        # Convert to DataFrame
        battle_df = pd.DataFrame(battle_data)
        
        # Compute correlation with outcome
        self.stat_importance = battle_df.corr()['outcome'].drop('outcome').sort_values(ascending=False)
    
    def save_analytics(self):
        """Save analytics results to file."""
        print(f"Saving analytics to {self.data_path}...")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        
        # Save data
        with open(self.data_path, 'wb') as f:
            pickle.dump({
                'pokemon_winrates': self.pokemon_winrates,
                'stat_correlations': self.stat_correlations,
                'type_effectiveness': self.type_effectiveness,
                'team_winrates_vgc': self.team_winrates_vgc,
                'team_winrates_standard': self.team_winrates_standard,
                'stat_importance': self.stat_importance
            }, f)
    
    def load_analytics(self):
        """Load analytics results from file."""
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
            
            self.pokemon_winrates = data['pokemon_winrates']
            self.stat_correlations = data['stat_correlations']
            self.type_effectiveness = data['type_effectiveness']
            self.team_winrates_vgc = data['team_winrates_vgc']
            self.team_winrates_standard = data['team_winrates_standard']
            self.stat_importance = data['stat_importance']
    
    def plot_top_pokemon_winrates(self, top_n=10, save_path=None):
        """
        Plot top Pokemon by win rate.
        
        Args:
            top_n: Number of top Pokemon to show
            save_path: Path to save the plot
        """
        if self.pokemon_winrates is None:
            print("Pokemon win rates not computed. Run compute_pokemon_winrates() first.")
            return
        
        # Filter to Pokemon with at least 10 battles
        min_battles = 10
        filtered_df = self.pokemon_winrates[self.pokemon_winrates['battles'] >= min_battles]
        
        # Get top N Pokemon
        top_pokemon = filtered_df.head(top_n)
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x='winrate', y='name', data=top_pokemon)
        plt.title(f'Top {top_n} Pokemon by Win Rate (min {min_battles} battles)')
        plt.xlabel('Win Rate')
        plt.ylabel('Pokemon')
        plt.xlim(0, 1)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    def plot_stat_correlations(self, save_path=None):
        """
        Plot correlations between stats and win rates.
        
        Args:
            save_path: Path to save the plot
        """
        if self.stat_correlations is None:
            print("Stat correlations not computed. Run compute_stat_correlations() first.")
            return
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x=self.stat_correlations.index, y=self.stat_correlations.values)
        plt.title('Correlation Between Pokemon Stats and Win Rate')
        plt.xlabel('Stat')
        plt.ylabel('Correlation with Win Rate')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    def plot_type_effectiveness_heatmap(self, save_path=None):
        """
        Plot type effectiveness heatmap.
        
        Args:
            save_path: Path to save the plot
        """
        if self.type_effectiveness is None:
            print("Type effectiveness not computed. Run compute_type_effectiveness() first.")
            return
        
        # Convert to DataFrame
        rows = []
        for matchup, data in self.type_effectiveness.items():
            if data['battles'] >= 5:  # Filter to matchups with at least 5 battles
                rows.append({
                    'attack_type': data['attack_type'],
                    'defense_type': data['defense_type'],
                    'winrate': data['winrate']
                })
        
        df = pd.DataFrame(rows)
        
        # Create pivot table
        pivot_df = df.pivot(index='attack_type', columns='defense_type', values='winrate')
        
        # Plot
        plt.figure(figsize=(14, 12))
        sns.heatmap(pivot_df, annot=True, cmap='RdYlGn', center=0.5, vmin=0, vmax=1)
        plt.title('Type Effectiveness (Win Rate)')
        plt.xlabel('Defending Type')
        plt.ylabel('Attacking Type')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    def plot_top_teams(self, battle_format=config.BattleFormat.VGC_2V2, top_n=5, save_path=None):
        """
        Plot top teams by win rate.
        
        Args:
            battle_format: Format of the battle (VGC_2V2 or STANDARD_6V6)
            top_n: Number of top teams to show
            save_path: Path to save the plot
        """
        # Get team win rates for the specified format
        if battle_format == config.BattleFormat.VGC_2V2:
            team_winrates = self.team_winrates_vgc
            format_name = "VGC (2v2)"
        else:
            team_winrates = self.team_winrates_standard
            format_name = "Standard (6v6)"
        
        if team_winrates is None:
            print(f"Team win rates for {format_name} not computed. Run compute_team_winrates() first.")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'team': ', '.join(data['pokemon_names']),
                'battles': data['battles'],
                'wins': data['wins'],
                'winrate': data['winrate']
            }
            for data in team_winrates.values()
        ])
        
        # Filter to teams with at least 5 battles
        min_battles = 5
        filtered_df = df[df['battles'] >= min_battles]
        
        # Get top N teams
        top_teams = filtered_df.sort_values('winrate', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(14, 8))
        sns.barplot(x='winrate', y='team', data=top_teams)
        plt.title(f'Top {top_n} Teams by Win Rate in {format_name} Format (min {min_battles} battles)')
        plt.xlabel('Win Rate')
        plt.ylabel('Team')
        plt.xlim(0, 1)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    def plot_stat_importance(self, save_path=None):
        """
        Plot importance of stats for winning battles.
        
        Args:
            save_path: Path to save the plot
        """
        if self.stat_importance is None:
            print("Stat importance not computed. Run compute_stat_importance() first.")
            return
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x=self.stat_importance.index, y=self.stat_importance.values)
        plt.title('Importance of Stat Differences for Winning Battles')
        plt.xlabel('Stat Difference (First Pokemon - Second Pokemon)')
        plt.ylabel('Correlation with First Pokemon Winning')
        plt.xticks(rotation=45)
        plt.axhline(y=0, color='k', linestyle='--')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    def generate_insights_report(self, output_path="insights_report.json"):
        """
        Generate a comprehensive insights report.
        
        Args:
            output_path: Path to save the report
        """
        print("Generating insights report...")
        
        # Ensure all analytics have been computed
        if any(x is None for x in [
            self.pokemon_winrates, self.stat_correlations, self.type_effectiveness,
            self.team_winrates_vgc, self.team_winrates_standard, self.stat_importance
        ]):
            print("Some analytics are missing. Run run_analytics() first.")
            return
        
        # Top Pokemon by win rate
        min_battles = 10
        filtered_winrates = self.pokemon_winrates[self.pokemon_winrates['battles'] >= min_battles]
        top_pokemon = filtered_winrates.head(10)[['name', 'winrate', 'battles', 'wins']].to_dict('records')
        
        # Most important stats
        important_stats = {
            stat: float(corr) for stat, corr in self.stat_correlations.items()
        }
        
        # Best type matchups
        type_matchups = []
        for matchup, data in self.type_effectiveness.items():
            if data['battles'] >= 5:  # Filter to matchups with at least 5 battles
                type_matchups.append({
                    'attack_type': data['attack_type'],
                    'defense_type': data['defense_type'],
                    'winrate': float(data['winrate']),
                    'battles': int(data['battles'])
                })
        
        best_type_matchups = sorted(type_matchups, key=lambda x: x['winrate'], reverse=True)[:10]
        
        # Best teams
        best_vgc_teams = []
        for data in self.team_winrates_vgc.values():
            if data['battles'] >= 5:  # Filter to teams with at least 5 battles
                best_vgc_teams.append({
                    'pokemon': data['pokemon_names'],
                    'winrate': float(data['winrate']),
                    'battles': int(data['battles']),
                    'wins': int(data['wins'])
                })
        
        best_vgc_teams = sorted(best_vgc_teams, key=lambda x: x['winrate'], reverse=True)[:5]
        
        best_standard_teams = []
        for data in self.team_winrates_standard.values():
            if data['battles'] >= 5:  # Filter to teams with at least 5 battles
                best_standard_teams.append({
                    'pokemon': data['pokemon_names'],
                    'winrate': float(data['winrate']),
                    'battles': int(data['battles']),
                    'wins': int(data['wins'])
                })
        
        best_standard_teams = sorted(best_standard_teams, key=lambda x: x['winrate'], reverse=True)[:5]
        
        # Stat importance
        stat_importance = {
            stat: float(corr) for stat, corr in self.stat_importance.items()
        }
        
        # Create report
        report = {
            'top_pokemon': top_pokemon,
            'important_stats': important_stats,
            'best_type_matchups': best_type_matchups,
            'best_vgc_teams': best_vgc_teams,
            'best_standard_teams': best_standard_teams,
            'stat_importance': stat_importance
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Insights report saved to {output_path}")
        
        return report
    
    def generate_all_plots(self, output_dir="plots"):
        """
        Generate all plots and save them to the specified directory.
        
        Args:
            output_dir: Directory to save plots
        """
        print("Generating all plots...")
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate plots
        self.plot_top_pokemon_winrates(save_path=f"{output_dir}/top_pokemon_winrates.png")
        self.plot_stat_correlations(save_path=f"{output_dir}/stat_correlations.png")
        self.plot_type_effectiveness_heatmap(save_path=f"{output_dir}/type_effectiveness.png")
        self.plot_top_teams(battle_format=config.BattleFormat.VGC_2V2, save_path=f"{output_dir}/top_vgc_teams.png")
        self.plot_top_teams(battle_format=config.BattleFormat.STANDARD_6V6, save_path=f"{output_dir}/top_standard_teams.png")
        self.plot_stat_importance(save_path=f"{output_dir}/stat_importance.png")
        
        print(f"All plots saved to {output_dir}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Pokemon battle analytics.")
    parser.add_argument(
        "--force-recompute", 
        action="store_true",
        help="Force recomputation of analytics even if saved data exists"
    )
    parser.add_argument(
        "--output-dir", 
        type=str,
        default="analytics_output",
        help="Directory to save output files"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize analytics
    analytics = BattleAnalytics(data_path=f"{args.output_dir}/analytics_data.pkl")
    
    # Load data
    analytics.load_data()
    
    # Run analytics
    analytics.run_analytics(force_recompute=args.force_recompute)
    
    # Generate insights report
    analytics.generate_insights_report(output_path=f"{args.output_dir}/insights_report.json")
    
    # Generate plots
    analytics.generate_all_plots(output_dir=f"{args.output_dir}/plots")
    
    print("Analytics completed successfully!")

if __name__ == "__main__":
    main() 