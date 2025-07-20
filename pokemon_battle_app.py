"""
Pokemon Battle Predictor Application

This application provides three modes of operation:
1. Analytics Mode: Initializes data and creates visualizations about Pokemon stats
2. Model Training Mode: Reloads data from analytics mode and trains prediction models
3. Battle Simulation Mode: Allows users to input 2v2 or 6v6 Pokemon teams and simulates battles

"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import json
from PIL import Image
import io
import base64
import time
import pickle
from sklearn.model_selection import train_test_split

# Import project modules
from analytics import BattleAnalytics
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from models import VGCBattleModel, StandardBattleModel, PokemonBattleModel
import config
from pokemon_utils import get_pokemon_sprite_url, download_pokemon_sprite

# Set page configuration
st.set_page_config(
    page_title="Pokemon Battle Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths for saved data
ANALYTICS_DATA_PATH = "data/analytics_data.pkl"
VGC_MODEL_PATH = "models/vgc_model.pkl"
STANDARD_MODEL_PATH = "models/standard_model.pkl"
PLOTS_DIR = "plots"

# Create directories if they don't exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Custom styling
def apply_custom_styling():
    """Apply custom CSS styling to the app"""
    st.markdown("""
        <style>
        .main {
            background-color: #f5f5f5;
        }
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .stButton>button {
            background-color: #3498db;
            color: white;
            border-radius: 5px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #2980b9;
        }
        .pokemon-card {
            background-color: white;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        .metric-card {
            background-color: #3498db;
            color: white;
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .sidebar .sidebar-content {
            background-color: #2c3e50;
        }
        </style>
    """, unsafe_allow_html=True)

# Helper functions
def load_pokemon_data():
    """Load and preprocess Pokemon data"""
    data_loader = DataLoader()
    data_loader.load_data()
    data_loader.preprocess_pokemon_data()
    
    # Create feature engineer
    feature_engineer = FeatureEngineer(data_loader.pokemon_df)
    
    return data_loader, feature_engineer

@st.cache_data
def get_pokemon_list(_data_loader):
    """Get list of Pokemon with ID and Name"""
    pokemon_list = _data_loader.pokemon_df[['#', 'Name']].copy()
    return pokemon_list.sort_values('Name')

def get_pokemon_details(pokemon_id, data_loader):
    """Get detailed information about a Pokemon"""
    pokemon = data_loader.pokemon_df[data_loader.pokemon_df['#'] == pokemon_id].iloc[0]
    return pokemon

def display_pokemon_card(pokemon, col):
    """Display a Pokemon card with its details"""
    with col:
        st.markdown(f"<div class='pokemon-card'>", unsafe_allow_html=True)
        st.subheader(f"{pokemon['Name']}")
        
        # Display Pokemon sprite
        pokemon_id = int(pokemon['#'])
        sprite_displayed = False
        
        # Try to load from local file first
        sprite_path = download_pokemon_sprite(pokemon_id)
        if sprite_path and os.path.exists(sprite_path):
            try:
                image = Image.open(sprite_path)
                st.image(image, width=100)
                sprite_displayed = True
            except Exception as e:
                pass
        
        # If local file failed, try direct URL
        if not sprite_displayed:
            try:
                sprite_url = get_pokemon_sprite_url(pokemon_id)
                st.image(sprite_url, width=100)
                sprite_displayed = True
            except Exception as e:
                pass
        
        # If all sprite methods failed, show a placeholder
        if not sprite_displayed:
            # Create a colored badge based on Pokemon type
            type_color = {
                "Normal": "#A8A77A", "Fire": "#EE8130", "Water": "#6390F0",
                "Electric": "#F7D02C", "Grass": "#7AC74C", "Ice": "#96D9D6",
                "Fighting": "#C22E28", "Poison": "#A33EA1", "Ground": "#E2BF65",
                "Flying": "#A98FF3", "Psychic": "#F95587", "Bug": "#A6B91A",
                "Rock": "#B6A136", "Ghost": "#735797", "Dragon": "#6F35FC",
                "Dark": "#705746", "Steel": "#B7B7CE", "Fairy": "#D685AD"
            }
            
            color = type_color.get(pokemon["Type 1"], "#CCCCCC")
            
            st.markdown(f"""
                <div style="width:100px;height:100px;background:{color};border-radius:50%;
                display:flex;align-items:center;justify-content:center;margin:0 auto;color:white;
                font-weight:bold;text-align:center;">
                {pokemon['Name']}
                </div>
            """, unsafe_allow_html=True)
        
        # Display types
        type_html = f"<p><strong>Type:</strong> {pokemon['Type 1']}"
        if pokemon['Type 2']:
            type_html += f" / {pokemon['Type 2']}"
        type_html += "</p>"
        st.markdown(type_html, unsafe_allow_html=True)
        
        # Display stats
        st.markdown("<p><strong>Stats:</strong></p>", unsafe_allow_html=True)
        stats = {
            'HP': pokemon['HP'],
            'Attack': pokemon['Attack'],
            'Defense': pokemon['Defense'],
            'Sp. Atk': pokemon['Sp. Atk'],
            'Sp. Def': pokemon['Sp. Def'],
            'Speed': pokemon['Speed']
        }
        
        # Create a horizontal bar chart for stats
        fig, ax = plt.subplots(figsize=(5, 3))
        bars = ax.barh(list(stats.keys()), list(stats.values()), color='#3498db')
        ax.set_xlim(0, 255)  # Max stat value
        ax.set_title("Base Stats")
        
        # Add values on bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 5, bar.get_y() + bar.get_height()/2, f"{width:.0f}", 
                    ha='left', va='center', fontsize=8)
        
        st.pyplot(fig)
        
        # Display legendary status
        if pokemon['Legendary'] == 'True':
            st.markdown("<p><strong>Legendary Pokemon</strong></p>", unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

# Mode 1: Analytics Mode
def analytics_mode():
    """Analytics mode to visualize Pokemon stats and battle data"""
    st.header("🔍 Pokemon Battle Analytics")
    
    # Check if analytics data exists
    if os.path.exists(ANALYTICS_DATA_PATH):
        st.info("Analytics data already exists. You can regenerate it or view existing visualizations.")
        
        col1, col2 = st.columns(2)
        with col1:
            regenerate = st.button("Regenerate Analytics Data", key="regenerate")
        with col2:
            view_existing = st.button("View Existing Visualizations", key="view_existing")
            
        if regenerate:
            run_analytics(force_recompute=True)
        elif view_existing:
            display_analytics_visualizations()
    else:
        st.warning("Analytics data not found. Please generate it first.")
        if st.button("Generate Analytics Data"):
            run_analytics(force_recompute=True)

def run_analytics(force_recompute=False):
    """Run analytics and generate visualizations"""
    st.info("Running analytics... This may take a few minutes.")
    progress_bar = st.progress(0)
    
    # Initialize analytics
    analytics = BattleAnalytics(data_path=ANALYTICS_DATA_PATH)
    
    # Load data
    analytics.load_data()
    progress_bar.progress(20)
    
    # Run analytics
    analytics.run_analytics(force_recompute=force_recompute)
    progress_bar.progress(60)
    
    # Generate plots
    os.makedirs(PLOTS_DIR, exist_ok=True)
    analytics.generate_all_plots(output_dir=PLOTS_DIR)
    progress_bar.progress(90)
    
    # Generate insights report
    analytics.generate_insights_report(output_path=os.path.join("data", "insights_report.json"))
    progress_bar.progress(100)
    
    st.success("Analytics completed successfully!")
    
    # Display visualizations
    display_analytics_visualizations()

def display_analytics_visualizations():
    """Display analytics visualizations"""
    st.subheader("Pokemon Battle Analytics Results")
    
    # Load insights report if it exists
    insights_path = os.path.join("data", "insights_report.json")
    if os.path.exists(insights_path):
        with open(insights_path, 'r') as f:
            insights = json.load(f)
        
        # Display top Pokemon by win rate
        st.subheader("Top 10 Pokemon by Win Rate")
        if os.path.exists(os.path.join(PLOTS_DIR, "top_pokemon_winrates.png")):
            st.image(os.path.join(PLOTS_DIR, "top_pokemon_winrates.png"))
            
            # Display top Pokemon as a table
            if 'top_pokemon' in insights and insights['top_pokemon']:
                top_pokemon_df = pd.DataFrame(insights['top_pokemon'])
                top_pokemon_df['winrate'] = top_pokemon_df['winrate'].apply(lambda x: f"{x:.2%}")
                st.dataframe(top_pokemon_df, use_container_width=True)
        
        # Display stat correlations
        st.subheader("Stat Correlations with Win Rate")
        if os.path.exists(os.path.join(PLOTS_DIR, "stat_correlations.png")):
            st.image(os.path.join(PLOTS_DIR, "stat_correlations.png"))
            
            # Add explanation
            st.markdown("### Key Insights")
            st.markdown(f"- {insights.get('stat_correlation_insights', 'No insights available')}")
        
        # Display type effectiveness
        st.subheader("Type Effectiveness Heatmap")
        if os.path.exists(os.path.join(PLOTS_DIR, "type_effectiveness.png")):
            st.image(os.path.join(PLOTS_DIR, "type_effectiveness.png"))
            
            # Add explanation
            st.markdown("### Type Effectiveness Insights")
            st.markdown(f"- {insights.get('type_effectiveness_insights', 'No insights available')}")
        
        # Display top teams
        st.subheader("Top Performing Teams")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top VGC (2v2) Teams")
            if os.path.exists(os.path.join(PLOTS_DIR, "top_vgc_teams.png")):
                st.image(os.path.join(PLOTS_DIR, "top_vgc_teams.png"))
        
        with col2:
            st.subheader("Top Standard (6v6) Teams")
            if os.path.exists(os.path.join(PLOTS_DIR, "top_standard_teams.png")):
                st.image(os.path.join(PLOTS_DIR, "top_standard_teams.png"))
        
        # Add team composition insights
        st.markdown("### Team Composition Insights")
        st.markdown(f"- {insights.get('team_composition_insights', 'No insights available')}")
        
        # Display stat importance
        st.subheader("Stat Importance in Battle Outcomes")
        if os.path.exists(os.path.join(PLOTS_DIR, "stat_importance.png")):
            st.image(os.path.join(PLOTS_DIR, "stat_importance.png"))
            
            # Add explanation
            st.markdown("### Stat Importance Insights")
            st.markdown(f"- {insights.get('stat_importance_insights', 'No insights available')}")
    else:
        st.error("Insights report not found. Please regenerate analytics data.")

# Mode 2: Model Training Mode
def model_training_mode():
    """Model training mode to train battle prediction models"""
    st.header("🧠 Pokemon Battle Model Training")
    
    # Check if analytics data exists
    if not os.path.exists(ANALYTICS_DATA_PATH):
        st.error("Analytics data not found. Please run Analytics Mode first.")
        return
    
    # Debug option
    if st.button("Debug Data"):
        st.info("Running data diagnostics...")
        data_loader, _ = load_pokemon_data()
        debug_info = data_loader.debug_data()
        
        # Display debug info in a more readable format
        st.subheader("Data Diagnostics Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Pokemon Count", debug_info["pokemon_count"])
            st.metric("Team Combat Count", debug_info["team_combat_count"])
            st.metric("Team Pokemon Count", debug_info["team_pokemon_count"])
        
        with col2:
            st.metric("Missing Pokemon IDs", debug_info["missing_pokemon_count"])
            st.metric("Valid Teams", debug_info["valid_teams"])
            st.metric("Valid Battles", debug_info["valid_battles"])
        
        if debug_info["valid_battles"] == 0:
            st.error("No valid battles found! This explains why the model training is failing.")
            st.info("Possible solutions:\n"
                   "1. Check that pokemon.csv contains all the Pokemon IDs referenced in pokemon_id_each_team.csv\n"
                   "2. Check that team IDs in team_combat.csv match those in pokemon_id_each_team.csv\n"
                   "3. Try using synthetic data for demonstration purposes")
    
    # Model options
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Select Model Type",
            ["random_forest", "logistic_regression"],
            index=0
        )
    
    with col2:
        battle_format = st.multiselect(
            "Select Battle Format",
            ["VGC (2v2)", "Standard (6v6)"],
            default=["VGC (2v2)", "Standard (6v6)"]
        )
    
    # Use synthetic data option
    use_synthetic = st.checkbox("Use synthetic data for demonstration", value=False)
    
    # Training button
    if st.button("Train Models"):
        train_models(model_type, battle_format, use_synthetic)

def train_models(model_type, battle_formats, use_synthetic=False):
    """Train battle prediction models"""
    st.info("Training models... This may take a few minutes.")
    progress_bar = st.progress(0)
    
    # Load data
    data_loader, feature_engineer = load_pokemon_data()
    progress_bar.progress(20)
    
    # Train VGC model if selected
    if "VGC (2v2)" in battle_formats:
        st.text("Training VGC (2v2) model...")
        
        if use_synthetic:
            # Create synthetic data for demonstration with more realistic patterns
            st.warning("Using synthetic data for VGC model training")
            num_features = 440  # Expected number of features
            num_samples = 500   # Increased number of samples for better training
            
            # Create base features with some correlation patterns
            X_base = np.random.rand(num_samples, 6)  # Base features representing Pokemon stats
            
            # Create derived features with correlations to simulate real Pokemon data patterns
            X = np.zeros((num_samples, num_features))
            
            # First 6 features are base stats
            X[:, :6] = X_base
            
            # Create team stats (sum, avg, max, min) with correlations to base stats
            for i in range(6):
                # Team 1 derived stats (sum, avg, max, min)
                X[:, 6+i*4] = X_base[:, i] * 2 + np.random.normal(0, 0.1, num_samples)  # Sum
                X[:, 7+i*4] = X_base[:, i] * 1 + np.random.normal(0, 0.1, num_samples)  # Avg
                X[:, 8+i*4] = X_base[:, i] * 1.5 + np.random.normal(0, 0.1, num_samples)  # Max
                X[:, 9+i*4] = X_base[:, i] * 0.5 + np.random.normal(0, 0.1, num_samples)  # Min
                
                # Team 2 derived stats with negative correlation to team 1
                X[:, 30+i*4] = (1 - X_base[:, i]) * 2 + np.random.normal(0, 0.1, num_samples)
                X[:, 31+i*4] = (1 - X_base[:, i]) * 1 + np.random.normal(0, 0.1, num_samples)
                X[:, 32+i*4] = (1 - X_base[:, i]) * 1.5 + np.random.normal(0, 0.1, num_samples)
                X[:, 33+i*4] = (1 - X_base[:, i]) * 0.5 + np.random.normal(0, 0.1, num_samples)
            
            # Fill remaining features with random noise
            X[:, 60:] = np.random.rand(num_samples, num_features-60) * 0.5
            
            # Generate labels with correlation to the features
            # Team with higher Speed and Attack tends to win
            team1_strength = X[:, 8] * 0.5 + X[:, 12] * 0.3  # Max Speed and Attack for team 1
            team2_strength = X[:, 32] * 0.5 + X[:, 36] * 0.3  # Max Speed and Attack for team 2
            
            # Add some randomness to make it realistic
            team1_strength += np.random.normal(0, 0.2, num_samples)
            team2_strength += np.random.normal(0, 0.2, num_samples)
            
            # Team 1 wins if their strength is higher
            y = (team1_strength > team2_strength).astype(int)
            
            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        else:
            try:
                # Prepare team battle data for VGC format
                X_train, X_test, y_train, y_test = data_loader.prepare_team_battle_data(
                    battle_format=config.BattleFormat.VGC_2V2
                )
            except Exception as e:
                st.error(f"Error preparing VGC battle data: {e}")
                st.warning("Falling back to synthetic data")
                # Create synthetic data as fallback (using the improved method)
                num_features = 440
                num_samples = 500
                X_base = np.random.rand(num_samples, 6)
                X = np.zeros((num_samples, num_features))
                X[:, :6] = X_base
                
                for i in range(6):
                    X[:, 6+i*4] = X_base[:, i] * 2 + np.random.normal(0, 0.1, num_samples)
                    X[:, 7+i*4] = X_base[:, i] * 1 + np.random.normal(0, 0.1, num_samples)
                    X[:, 8+i*4] = X_base[:, i] * 1.5 + np.random.normal(0, 0.1, num_samples)
                    X[:, 9+i*4] = X_base[:, i] * 0.5 + np.random.normal(0, 0.1, num_samples)
                    
                    X[:, 30+i*4] = (1 - X_base[:, i]) * 2 + np.random.normal(0, 0.1, num_samples)
                    X[:, 31+i*4] = (1 - X_base[:, i]) * 1 + np.random.normal(0, 0.1, num_samples)
                    X[:, 32+i*4] = (1 - X_base[:, i]) * 1.5 + np.random.normal(0, 0.1, num_samples)
                    X[:, 33+i*4] = (1 - X_base[:, i]) * 0.5 + np.random.normal(0, 0.1, num_samples)
                
                X[:, 60:] = np.random.rand(num_samples, num_features-60) * 0.5
                
                team1_strength = X[:, 8] * 0.5 + X[:, 12] * 0.3
                team2_strength = X[:, 32] * 0.5 + X[:, 36] * 0.3
                team1_strength += np.random.normal(0, 0.2, num_samples)
                team2_strength += np.random.normal(0, 0.2, num_samples)
                y = (team1_strength > team2_strength).astype(int)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and train model
        vgc_model = VGCBattleModel(model_type=model_type)
        vgc_model.set_feature_engineer(feature_engineer)
        
        # Set optimal hyperparameters based on model type
        if model_type == "random_forest":
            # Better hyperparameters for RandomForest
            vgc_model.model.n_estimators = 200
            vgc_model.model.max_depth = 20
            vgc_model.model.min_samples_split = 5
        elif model_type == "logistic_regression":
            # Better hyperparameters for LogisticRegression
            vgc_model.model.C = 0.5
            vgc_model.model.max_iter = 1000
        
        vgc_model.fit(X_train, y_train)
        
        # Evaluate model
        metrics = vgc_model.evaluate(X_test, y_test)
        
        # Calibrate probabilities
        vgc_model.calibrate_probabilities(X_train, y_train)
        
        # Save model
        os.makedirs(os.path.dirname(VGC_MODEL_PATH), exist_ok=True)
        vgc_model.save_model(VGC_MODEL_PATH)
        
        # Display metrics
        st.subheader("VGC Model Performance")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        col2.metric("ROC AUC", f"{metrics['roc_auc']:.4f}")
        col3.metric("Brier Score", f"{metrics['brier_score']:.4f}")
        col4.metric("Log Loss", f"{metrics['log_loss']:.4f}")
    
    progress_bar.progress(60)
    
    # Train Standard model if selected
    if "Standard (6v6)" in battle_formats:
        st.text("Training Standard (6v6) model...")
        
        if use_synthetic:
            # Create synthetic data for demonstration with more realistic patterns
            st.warning("Using synthetic data for Standard model training")
            num_features = 440  # Expected number of features
            num_samples = 500   # Increased number of samples for better training
            
            # Create base features with some correlation patterns
            X_base = np.random.rand(num_samples, 6)  # Base features representing Pokemon stats
            
            # Create derived features with correlations to simulate real Pokemon data patterns
            X = np.zeros((num_samples, num_features))
            
            # First 6 features are base stats
            X[:, :6] = X_base
            
            # Create team stats (sum, avg, max, min) with correlations to base stats
            for i in range(6):
                # Team 1 derived stats (sum, avg, max, min)
                X[:, 6+i*4] = X_base[:, i] * 2 + np.random.normal(0, 0.1, num_samples)  # Sum
                X[:, 7+i*4] = X_base[:, i] * 1 + np.random.normal(0, 0.1, num_samples)  # Avg
                X[:, 8+i*4] = X_base[:, i] * 1.5 + np.random.normal(0, 0.1, num_samples)  # Max
                X[:, 9+i*4] = X_base[:, i] * 0.5 + np.random.normal(0, 0.1, num_samples)  # Min
                
                # Team 2 derived stats with negative correlation to team 1
                X[:, 30+i*4] = (1 - X_base[:, i]) * 2 + np.random.normal(0, 0.1, num_samples)
                X[:, 31+i*4] = (1 - X_base[:, i]) * 1 + np.random.normal(0, 0.1, num_samples)
                X[:, 32+i*4] = (1 - X_base[:, i]) * 1.5 + np.random.normal(0, 0.1, num_samples)
                X[:, 33+i*4] = (1 - X_base[:, i]) * 0.5 + np.random.normal(0, 0.1, num_samples)
            
            # Fill remaining features with random noise
            X[:, 60:] = np.random.rand(num_samples, num_features-60) * 0.5
            
            # Generate labels with correlation to the features
            # Team with higher Speed and Attack tends to win
            team1_strength = X[:, 8] * 0.5 + X[:, 12] * 0.3  # Max Speed and Attack for team 1
            team2_strength = X[:, 32] * 0.5 + X[:, 36] * 0.3  # Max Speed and Attack for team 2
            
            # Add some randomness to make it realistic
            team1_strength += np.random.normal(0, 0.2, num_samples)
            team2_strength += np.random.normal(0, 0.2, num_samples)
            
            # Team 1 wins if their strength is higher
            y = (team1_strength > team2_strength).astype(int)
            
            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        else:
            try:
                # Prepare team battle data for Standard format
                X_train, X_test, y_train, y_test = data_loader.prepare_team_battle_data(
                    battle_format=config.BattleFormat.STANDARD_6V6
                )
            except Exception as e:
                st.error(f"Error preparing Standard battle data: {e}")
                st.warning("Falling back to synthetic data")
                # Create synthetic data as fallback (using the improved method)
                num_features = 440
                num_samples = 500
                X_base = np.random.rand(num_samples, 6)
                X = np.zeros((num_samples, num_features))
                X[:, :6] = X_base
                
                for i in range(6):
                    X[:, 6+i*4] = X_base[:, i] * 2 + np.random.normal(0, 0.1, num_samples)
                    X[:, 7+i*4] = X_base[:, i] * 1 + np.random.normal(0, 0.1, num_samples)
                    X[:, 8+i*4] = X_base[:, i] * 1.5 + np.random.normal(0, 0.1, num_samples)
                    X[:, 9+i*4] = X_base[:, i] * 0.5 + np.random.normal(0, 0.1, num_samples)
                    
                    X[:, 30+i*4] = (1 - X_base[:, i]) * 2 + np.random.normal(0, 0.1, num_samples)
                    X[:, 31+i*4] = (1 - X_base[:, i]) * 1 + np.random.normal(0, 0.1, num_samples)
                    X[:, 32+i*4] = (1 - X_base[:, i]) * 1.5 + np.random.normal(0, 0.1, num_samples)
                    X[:, 33+i*4] = (1 - X_base[:, i]) * 0.5 + np.random.normal(0, 0.1, num_samples)
                
                X[:, 60:] = np.random.rand(num_samples, num_features-60) * 0.5
                
                team1_strength = X[:, 8] * 0.5 + X[:, 12] * 0.3
                team2_strength = X[:, 32] * 0.5 + X[:, 36] * 0.3
                team1_strength += np.random.normal(0, 0.2, num_samples)
                team2_strength += np.random.normal(0, 0.2, num_samples)
                y = (team1_strength > team2_strength).astype(int)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and train model
        standard_model = StandardBattleModel(model_type=model_type)
        standard_model.set_feature_engineer(feature_engineer)
        
        # Set optimal hyperparameters based on model type
        if model_type == "random_forest":
            # Better hyperparameters for RandomForest
            standard_model.model.n_estimators = 200
            standard_model.model.max_depth = 20
            standard_model.model.min_samples_split = 5
        elif model_type == "logistic_regression":
            # Better hyperparameters for LogisticRegression
            standard_model.model.C = 0.5
            standard_model.model.max_iter = 1000
            
        standard_model.fit(X_train, y_train)
        
        # Evaluate model
        metrics = standard_model.evaluate(X_test, y_test)
        
        # Calibrate probabilities
        standard_model.calibrate_probabilities(X_train, y_train)
        
        # Save model
        os.makedirs(os.path.dirname(STANDARD_MODEL_PATH), exist_ok=True)
        standard_model.save_model(STANDARD_MODEL_PATH)
        
        # Display metrics
        st.subheader("Standard Model Performance")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        col2.metric("ROC AUC", f"{metrics['roc_auc']:.4f}")
        col3.metric("Brier Score", f"{metrics['brier_score']:.4f}")
        col4.metric("Log Loss", f"{metrics['log_loss']:.4f}")
    
    progress_bar.progress(100)
    st.success("Model training completed successfully!")

# Mode 3: Battle Simulation Mode
def battle_simulation_mode():
    """Battle simulation mode to predict outcomes of Pokemon battles"""
    st.header("⚔️ Pokemon Battle Simulator")
    
    # Check if models exist
    vgc_model_exists = os.path.exists(VGC_MODEL_PATH)
    standard_model_exists = os.path.exists(STANDARD_MODEL_PATH)
    
    if not (vgc_model_exists or standard_model_exists):
        st.error("No trained models found. Please run Model Training Mode first.")
        return
    
    # Debug mode
    debug_mode = st.sidebar.checkbox("Debug Mode", value=False, key="debug_mode_sidebar")
    if debug_mode:
        st.sidebar.info("Debug Mode enabled. Additional information will be shown.")
    
    # Battle format selection
    available_formats = []
    if vgc_model_exists:
        available_formats.append("VGC (2v2)")
    if standard_model_exists:
        available_formats.append("Standard (6v6)")
    
    battle_format = st.selectbox(
        "Select Battle Format",
        available_formats,
        index=0
    )
    
    # Load Pokemon data
    data_loader, feature_engineer = load_pokemon_data()
    pokemon_list = get_pokemon_list(data_loader)
    
    # Team selection
    st.subheader("Select Your Teams")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3 style='color: #e74c3c;'>Team 1</h3>", unsafe_allow_html=True)
        team1_random = st.checkbox("Random Team 1", key="team1_random")
        
        team1_ids = []
        if battle_format == "VGC (2v2)":
            team_size = 4
        else:  # Standard (6v6)
            team_size = 6
            
        if team1_random:
            # Generate random team
            team1_ids = random.sample(pokemon_list['#'].tolist(), team_size)
        else:
            # Manual selection
            for i in range(team_size):
                pokemon_id = st.selectbox(
                    f"Select Pokemon {i+1}",
                    options=pokemon_list['#'].tolist(),
                    format_func=lambda x: pokemon_list[pokemon_list['#'] == x]['Name'].values[0],
                    key=f"team1_pokemon_{i}"
                )
                team1_ids.append(pokemon_id)
    
    with col2:
        st.markdown("<h3 style='color: #3498db;'>Team 2</h3>", unsafe_allow_html=True)
        team2_random = st.checkbox("Random Team 2", key="team2_random")
        
        team2_ids = []
        if team2_random:
            # Generate random team
            team2_ids = random.sample(pokemon_list['#'].tolist(), team_size)
        else:
            # Manual selection
            for i in range(team_size):
                pokemon_id = st.selectbox(
                    f"Select Pokemon {i+1}",
                    options=pokemon_list['#'].tolist(),
                    format_func=lambda x: pokemon_list[pokemon_list['#'] == x]['Name'].values[0],
                    key=f"team2_pokemon_{i}"
                )
                team2_ids.append(pokemon_id)
    
    # Display selected teams
    st.subheader("Selected Teams")
    
    # Team 1
    st.markdown("<h4 style='color: #e74c3c;'>Team 1</h4>", unsafe_allow_html=True)
    team1_cols = st.columns(team_size)
    for i, pokemon_id in enumerate(team1_ids):
        pokemon = get_pokemon_details(pokemon_id, data_loader)
        display_pokemon_card(pokemon, team1_cols[i])
    
    # Team 2
    st.markdown("<h4 style='color: #3498db;'>Team 2</h4>", unsafe_allow_html=True)
    team2_cols = st.columns(team_size)
    for i, pokemon_id in enumerate(team2_ids):
        pokemon = get_pokemon_details(pokemon_id, data_loader)
        display_pokemon_card(pokemon, team2_cols[i])
    
    # Battle simulation
    if st.button("Simulate Battle"):
        simulate_battle(team1_ids, team2_ids, battle_format, data_loader, feature_engineer)

def simulate_battle(team1_ids, team2_ids, battle_format, data_loader, feature_engineer):
    """Simulate a battle between two teams"""
    st.subheader("Battle Simulation")
    
    # Check if debug mode is enabled from the sidebar
    debug_mode = st.session_state.get('debug_mode_sidebar', False)
    
    # Display debug information if enabled
    if debug_mode:
        st.subheader("Debug Information")
        
        # Show team IDs
        st.write("Team 1 IDs:", team1_ids)
        st.write("Team 2 IDs:", team2_ids)
        
        # Show Pokémon details
        st.write("Team 1 Pokémon:")
        for pid in team1_ids:
            pokemon = get_pokemon_details(pid, data_loader)
            st.write(f"ID: {pid}, Name: {pokemon['Name']}, Types: {pokemon['Type 1']}/{pokemon['Type 2']}")
            
            # Show sprite mapping
            from pokemon_utils import map_to_national_dex_id
            national_dex_id = map_to_national_dex_id(pid)
            st.write(f"Maps to National Dex ID: {national_dex_id}")
            
            # Show sprite URL
            from pokemon_utils import get_pokemon_sprite_url
            sprite_url = get_pokemon_sprite_url(pid)
            st.write(f"Sprite URL: {sprite_url}")
        
        st.write("Team 2 Pokémon:")
        for pid in team2_ids:
            pokemon = get_pokemon_details(pid, data_loader)
            st.write(f"ID: {pid}, Name: {pokemon['Name']}, Types: {pokemon['Type 1']}/{pokemon['Type 2']}")
            
            # Show sprite mapping
            from pokemon_utils import map_to_national_dex_id
            national_dex_id = map_to_national_dex_id(pid)
            st.write(f"Maps to National Dex ID: {national_dex_id}")
            
            # Show sprite URL
            from pokemon_utils import get_pokemon_sprite_url
            sprite_url = get_pokemon_sprite_url(pid)
            st.write(f"Sprite URL: {sprite_url}")
    
    # Check if teams are identical
    identical_teams = sorted(team1_ids) == sorted(team2_ids)
    
    # Display loading animation
    with st.spinner("Simulating battle..."):
        # Artificial delay for dramatic effect
        time.sleep(1)
        
        # If teams are identical, set a true 50/50 outcome
        if identical_teams:
            result = {
                'team1_win_probability': 0.5,
                'team2_win_probability': 0.5,
                'predicted_winner': 'Draw',
                'confidence': 0.0,
                'top_features': {
                    'Teams are identical': 0.0,
                    'Equal HP': 0.0,
                    'Equal Attack': 0.0,
                    'Equal Defense': 0.0,
                    'Equal Speed': 0.0
                }
            }
        else:
            # Load appropriate model
            if battle_format == "VGC (2v2)":
                model_path = VGC_MODEL_PATH
                model = PokemonBattleModel.load_model(model_path)
                model.set_feature_engineer(feature_engineer)
                result = model.predict_battle(team1_ids, team2_ids)
            else:  # Standard (6v6)
                model_path = STANDARD_MODEL_PATH
                model = PokemonBattleModel.load_model(model_path)
                model.set_feature_engineer(feature_engineer)
                result = model.predict_battle(team1_ids, team2_ids)
    
    # Display battle result with improved visuals
    st.markdown("""
    <style>
    .battle-container {
        background-color: #f8f9fa;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .winner-announcement {
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
        animation: pulse 2s infinite;
    }
    .draw-announcement {
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
        color: #8e44ad;
    }
    .team1-color { color: #e74c3c; }
    .team2-color { color: #3498db; }
    .probability-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 10px;
    }
    .probability-bar {
        height: 30px;
        border-radius: 15px;
        color: white;
        font-weight: bold;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: width 1s ease-in-out;
    }
    .confidence-meter {
        text-align: center;
        font-size: 16px;
        margin: 10px 0;
    }
    .battle-scene {
        background-color: #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin: 20px 0;
        position: relative;
        min-height: 200px;
    }
    .team-sprites {
        display: flex;
        justify-content: space-around;
        align-items: center;
    }
    .team1-sprites {
        transform: scaleX(-1); /* Flip team 1 sprites to face team 2 */
    }
    .vs-badge {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background-color: #f39c12;
        color: white;
        font-weight: bold;
        padding: 5px 15px;
        border-radius: 50%;
        z-index: 10;
        box-shadow: 0 0 10px rgba(0,0,0,0.3);
    }
    .winner-badge {
        position: absolute;
        top: -15px;
        background-color: gold;
        color: black;
        font-weight: bold;
        padding: 5px 15px;
        border-radius: 15px;
        z-index: 10;
        box-shadow: 0 0 10px rgba(0,0,0,0.3);
        animation: pulse 1.5s infinite;
    }
    .team1-winner {
        left: 25%;
    }
    .team2-winner {
        right: 25%;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="battle-container">', unsafe_allow_html=True)
    
    # Winner announcement with animation
    team1_prob = result['team1_win_probability']
    team2_prob = result['team2_win_probability']
    
    # Handle draw case (identical teams or nearly 50/50 prediction)
    is_draw = identical_teams or abs(team1_prob - 0.5) < 0.01
    
    # Create battle scene with sprites
    st.markdown('<div class="battle-scene">', unsafe_allow_html=True)
    
    # Add winner badge
    if not is_draw:
        if team1_prob > team2_prob:
            st.markdown('<div class="winner-badge team1-winner">WINNER</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="winner-badge team2-winner">WINNER</div>', unsafe_allow_html=True)
    
    # Add VS badge in the middle
    st.markdown('<div class="vs-badge">VS</div>', unsafe_allow_html=True)
    
    # Display team sprites in a battle formation
    col1, col2 = st.columns(2)
    
    # Team 1 sprites
    with col1:
        st.markdown('<div class="team-sprites team1-sprites">', unsafe_allow_html=True)
        for pokemon_id in team1_ids[:min(4, len(team1_ids))]:  # Show up to 4 sprites
            pokemon = get_pokemon_details(pokemon_id, data_loader)
            sprite_path = download_pokemon_sprite(pokemon_id)
            if sprite_path and os.path.exists(sprite_path):
                try:
                    image = Image.open(sprite_path)
                    st.image(image, width=80, caption=pokemon['Name'])
                except Exception as e:
                    st.warning(f"Error loading sprite for {pokemon['Name']}: {e}")
                    # Try direct URL as fallback
                    try:
                        sprite_url = get_pokemon_sprite_url(pokemon_id)
                        st.image(sprite_url, width=80, caption=pokemon['Name'])
                    except:
                        st.markdown(f"<div style='width:80px;height:80px;background:#ddd;text-align:center;'>{pokemon['Name']}</div>", unsafe_allow_html=True)
            else:
                # Try direct URL
                try:
                    sprite_url = get_pokemon_sprite_url(pokemon_id)
                    st.image(sprite_url, width=80, caption=pokemon['Name'])
                except:
                    st.markdown(f"<div style='width:80px;height:80px;background:#ddd;text-align:center;'>{pokemon['Name']}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Team 2 sprites
    with col2:
        st.markdown('<div class="team-sprites team2-sprites">', unsafe_allow_html=True)
        for pokemon_id in team2_ids[:min(4, len(team2_ids))]:  # Show up to 4 sprites
            pokemon = get_pokemon_details(pokemon_id, data_loader)
            sprite_path = download_pokemon_sprite(pokemon_id)
            if sprite_path and os.path.exists(sprite_path):
                try:
                    image = Image.open(sprite_path)
                    st.image(image, width=80, caption=pokemon['Name'])
                except Exception as e:
                    st.warning(f"Error loading sprite for {pokemon['Name']}: {e}")
                    # Try direct URL as fallback
                    try:
                        sprite_url = get_pokemon_sprite_url(pokemon_id)
                        st.image(sprite_url, width=80, caption=pokemon['Name'])
                    except:
                        st.markdown(f"<div style='width:80px;height:80px;background:#ddd;text-align:center;'>{pokemon['Name']}</div>", unsafe_allow_html=True)
            else:
                # Try direct URL
                try:
                    sprite_url = get_pokemon_sprite_url(pokemon_id)
                    st.image(sprite_url, width=80, caption=pokemon['Name'])
                except:
                    st.markdown(f"<div style='width:80px;height:80px;background:#ddd;text-align:center;'>{pokemon['Name']}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if is_draw:
        st.markdown('<div class="draw-announcement">Draw (50/50)</div>', unsafe_allow_html=True)
    elif team1_prob > team2_prob:
        st.markdown('<div class="winner-announcement team1-color">Team 1 Wins!</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="winner-announcement team2-color">Team 2 Wins!</div>', unsafe_allow_html=True)
    
    # Create a visual probability bar
    st.markdown('<div class="probability-container">', unsafe_allow_html=True)
    
    # Calculate bar widths based on probabilities (ensure they're valid values)
    # Use fixed values for columns to avoid errors
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="probability-bar" style="background-color: #e74c3c; width: 100%;">
            Team 1: {team1_prob:.1%}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="probability-bar" style="background-color: #3498db; width: 100%;">
            Team 2: {team2_prob:.1%}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Confidence meter
    confidence = result['confidence']
    confidence_level = "None (Draw)" if is_draw else "Very Low"
    if not is_draw:
        if confidence > 0.2: confidence_level = "Low"
        if confidence > 0.4: confidence_level = "Moderate"
        if confidence > 0.6: confidence_level = "High"
        if confidence > 0.8: confidence_level = "Very High"
    
    st.markdown(f"""
    <div class="confidence-meter">
        <strong>Prediction Confidence:</strong> {confidence:.1%} ({confidence_level})
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display top contributing features
    st.subheader("Top Contributing Factors")
    
    # Check if we have any features
    if not result['top_features'] or len(result['top_features']) == 0:
        # Generate some default feature importance based on Pokémon stats
        st.info("No feature importance data available from the model. Generating stat-based comparison.")
        
        # Get Pokémon details for both teams
        team1_pokemon = [get_pokemon_details(pid, data_loader) for pid in team1_ids]
        team2_pokemon = [get_pokemon_details(pid, data_loader) for pid in team2_ids]
        
        # Calculate average stats for each team
        team1_stats = {
            'HP': sum(p['HP'] for p in team1_pokemon) / len(team1_pokemon),
            'Attack': sum(p['Attack'] for p in team1_pokemon) / len(team1_pokemon),
            'Defense': sum(p['Defense'] for p in team1_pokemon) / len(team1_pokemon),
            'Sp. Atk': sum(p['Sp. Atk'] for p in team1_pokemon) / len(team1_pokemon),
            'Sp. Def': sum(p['Sp. Def'] for p in team1_pokemon) / len(team1_pokemon),
            'Speed': sum(p['Speed'] for p in team1_pokemon) / len(team1_pokemon)
        }
        
        team2_stats = {
            'HP': sum(p['HP'] for p in team2_pokemon) / len(team2_pokemon),
            'Attack': sum(p['Attack'] for p in team2_pokemon) / len(team2_pokemon),
            'Defense': sum(p['Defense'] for p in team2_pokemon) / len(team2_pokemon),
            'Sp. Atk': sum(p['Sp. Atk'] for p in team2_pokemon) / len(team2_pokemon),
            'Sp. Def': sum(p['Sp. Def'] for p in team2_pokemon) / len(team2_pokemon),
            'Speed': sum(p['Speed'] for p in team2_pokemon) / len(team2_pokemon)
        }
        
        # Calculate stat differences and create feature importance
        generated_features = {}
        for stat in team1_stats:
            diff = team1_stats[stat] - team2_stats[stat]
            if abs(diff) > 0:
                # Normalize the difference to a reasonable scale for visualization
                normalized_diff = diff / 100.0  # Scale factor
                if diff > 0:
                    generated_features[f"Team 1 higher {stat}"] = normalized_diff
                else:
                    generated_features[f"Team 2 higher {stat}"] = -normalized_diff
        
        # Sort by absolute value and take top 5
        sorted_features = sorted(generated_features.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        result['top_features'] = dict(sorted_features)
    
    # Get top 5 features or all if less than 5
    features = list(result['top_features'].keys())[:5]
    values = list(result['top_features'].values())[:5]
    
    # Create a dataframe for better visualization
    feature_df = pd.DataFrame({
        'Feature': features,
        'Impact': values,
        'Favors': ['Team 1' if v > 0 else 'Team 2' if v < 0 else 'Equal' for v in values],
        'AbsImpact': [abs(v) for v in values]
    })
    
    # Sort by absolute impact
    feature_df = feature_df.sort_values('AbsImpact', ascending=False).reset_index(drop=True)
    
    # Determine colors based on whether feature favors team 1 or team 2
    colors = ['#e74c3c' if row.Impact > 0 else '#3498db' if row.Impact < 0 else '#8e44ad' 
             for _, row in feature_df.iterrows()]
    
    # Create horizontal bar chart with improved styling
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')
    
    # Create bars with absolute values for proper sizing
    bars = ax.barh(feature_df['Feature'], feature_df['AbsImpact'], color=colors)
    
    # Add labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label_x_pos = width + 0.01
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f"{width:.3f}", 
                ha='left', va='center', fontweight='bold')
        
        # Add "Team 1" or "Team 2" label to indicate which team the feature favors
        team_label = feature_df.iloc[i]['Favors']
        ax.text(0.01, bar.get_y() + bar.get_height()/2, f"Favors {team_label}", 
                ha='left', va='center', color='white', fontweight='bold')
    
    ax.set_xlabel('Impact on Prediction', fontweight='bold')
    ax.set_title('Top 5 Features Influencing Battle Outcome', fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#dddddd')
    ax.spines['bottom'].set_color('#dddddd')
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Favors Team 1'),
        Patch(facecolor='#3498db', label='Favors Team 2')
    ]
    if any(v == 0 for v in values):
        legend_elements.append(Patch(facecolor='#8e44ad', label='Equal Impact'))
        
    ax.legend(handles=legend_elements, loc='lower right', frameon=False)
    
    st.pyplot(fig)
    
    # Add a table view of the feature importance for clarity
    st.markdown("### Feature Importance Details")
    
    # Format the dataframe for display
    display_df = feature_df[['Feature', 'Impact', 'Favors']].copy()
    display_df['Impact'] = display_df['Impact'].apply(lambda x: f"{x:.4f}")
    
    # Display as a styled table
    st.dataframe(display_df, use_container_width=True)
    
    # Display battle analysis
    st.subheader("Battle Analysis")
    
    # Create a more detailed and insightful explanation
    if is_draw:
        explanation = "### Battle Analysis: Draw\n\n"
        explanation += "The teams are evenly matched, resulting in a 50/50 chance for either team to win.\n\n"
        
        if identical_teams:
            explanation += "**Reason:** Both teams have identical Pokémon compositions.\n\n"
        else:
            explanation += "**Reason:** While the teams differ slightly, their overall strengths and weaknesses balance out.\n\n"
            
        explanation += "In such evenly matched battles, the outcome would likely depend on player skill, move selection, and battle strategy rather than team composition."
        
    else:
        explanation = "### Key factors in this battle prediction:\n\n"
        
        # Group features by team they favor
        team1_features = [(f, v) for f, v in list(result['top_features'].items())[:5] if v > 0]
        team2_features = [(f, v) for f, v in list(result['top_features'].items())[:5] if v < 0]
        
        if team1_features:
            explanation += "**Factors favoring Team 1:**\n"
            for feature, value in team1_features:
                explanation += f"- {feature}: +{abs(value):.3f} impact\n"
            explanation += "\n"
        
        if team2_features:
            explanation += "**Factors favoring Team 2:**\n"
            for feature, value in team2_features:
                explanation += f"- {feature}: +{abs(value):.3f} impact\n"
            explanation += "\n"
        
        # Add overall conclusion
        if team1_prob > team2_prob:
            explanation += f"**Conclusion:** Team 1 has a {team1_prob:.1%} chance of winning, primarily due to "
            explanation += f"advantages in {team1_features[0][0] if team1_features else 'overall stats'}."
        else:
            explanation += f"**Conclusion:** Team 2 has a {team2_prob:.1%} chance of winning, primarily due to "
            explanation += f"advantages in {team2_features[0][0] if team2_features else 'overall stats'}."
    
    st.markdown(explanation)
    
    # Add option to display full battle visualization using matplotlib
    if st.checkbox("Show detailed battle visualization with sprites"):
        st.subheader("Battle Visualization")
        
        # Determine winner
        winner = None
        if not is_draw:
            if team1_prob > team2_prob:
                winner = 1
            else:
                winner = 2
        
        # Create visualization using pokemon_utils
        from pokemon_utils import display_battle_result
        fig = display_battle_result(team1_ids, team2_ids, team1_prob, winner)
        
        if fig:
            st.pyplot(fig)
        else:
            st.warning("Could not create battle visualization. Some sprites may be missing.")

# Main application
def main():
    """Main application function"""
    apply_custom_styling()
    
    # App header
    st.title("🔮 Pokemon Battle Predictor")
    st.markdown("Predict the outcomes of Pokemon battles with machine learning!")
    
    # Mode selection
    st.sidebar.title("Mode Selection")
    mode = st.sidebar.radio(
        "Select Mode",
        ["Analytics Mode", "Model Training Mode", "Battle Simulation Mode"]
    )
    
    # Display selected mode
    if mode == "Analytics Mode":
        analytics_mode()
    elif mode == "Model Training Mode":
        model_training_mode()
    else:  # Battle Simulation Mode
        battle_simulation_mode()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("Pokemon Battle Predictor v1.0")

if __name__ == "__main__":
    main() 