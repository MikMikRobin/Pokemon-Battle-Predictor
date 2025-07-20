"""
Pokemon Battle Predictor Application

This application provides three modes of operation:
1. Analytics Mode: Initializes data and creates visualizations about Pokemon stats
2. Model Training Mode: Reloads data from analytics mode and trains prediction models
3. Battle Simulation Mode: Allows users to input 2v2 or 6v6 Pokemon teams and simulates battles

The application features a sleek UI built with Streamlit.
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

# Import project modules
from analytics import BattleAnalytics
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from models import VGCBattleModel, StandardBattleModel, PokemonBattleModel
import config

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
def get_pokemon_list(data_loader):
    """Get list of Pokemon with ID and Name"""
    pokemon_list = data_loader.pokemon_df[['#', 'Name']].copy()
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
        
        # Display stat correlations
        st.subheader("Stat Correlations with Win Rate")
        if os.path.exists(os.path.join(PLOTS_DIR, "stat_correlations.png")):
            st.image(os.path.join(PLOTS_DIR, "stat_correlations.png"))
            
            # Add explanation
            st.markdown(f"""
            **Key Insights:**
            - {insights.get('stat_correlation_insights', 'No insights available')}
            """)
        
        # Display type effectiveness
        st.subheader("Type Effectiveness Heatmap")
        if os.path.exists(os.path.join(PLOTS_DIR, "type_effectiveness.png")):
            st.image(os.path.join(PLOTS_DIR, "type_effectiveness.png"))
        
        # Display top teams
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top VGC (2v2) Teams")
            if os.path.exists(os.path.join(PLOTS_DIR, "top_vgc_teams.png")):
                st.image(os.path.join(PLOTS_DIR, "top_vgc_teams.png"))
        
        with col2:
            st.subheader("Top Standard (6v6) Teams")
            if os.path.exists(os.path.join(PLOTS_DIR, "top_standard_teams.png")):
                st.image(os.path.join(PLOTS_DIR, "top_standard_teams.png"))
        
        # Display stat importance
        st.subheader("Stat Importance in Battle Outcomes")
        if os.path.exists(os.path.join(PLOTS_DIR, "stat_importance.png")):
            st.image(os.path.join(PLOTS_DIR, "stat_importance.png"))
            
            # Add explanation
            st.markdown(f"""
            **Key Insights:**
            - {insights.get('stat_importance_insights', 'No insights available')}
            """)
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
    
    # Training button
    if st.button("Train Models"):
        train_models(model_type, battle_format)

def train_models(model_type, battle_formats):
    """Train battle prediction models"""
    st.info("Training models... This may take a few minutes.")
    progress_bar = st.progress(0)
    
    # Load data
    data_loader, feature_engineer = load_pokemon_data()
    progress_bar.progress(20)
    
    # Train VGC model if selected
    if "VGC (2v2)" in battle_formats:
        st.text("Training VGC (2v2) model...")
        
        # Prepare team battle data for VGC format
        X_train, X_test, y_train, y_test = data_loader.prepare_team_battle_data(
            battle_format=config.BattleFormat.VGC_2V2
        )
        
        # Create and train model
        vgc_model = VGCBattleModel(model_type=model_type)
        vgc_model.set_feature_engineer(feature_engineer)
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
        
        # Prepare team battle data for Standard format
        X_train, X_test, y_train, y_test = data_loader.prepare_team_battle_data(
            battle_format=config.BattleFormat.STANDARD_6V6
        )
        
        # Create and train model
        standard_model = StandardBattleModel(model_type=model_type)
        standard_model.set_feature_engineer(feature_engineer)
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
    
    # Display loading animation
    with st.spinner("Simulating battle..."):
        # Artificial delay for dramatic effect
        time.sleep(2)
        
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
    
    # Display battle result
    st.markdown("<div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>", unsafe_allow_html=True)
    
    # Winner announcement
    if result['team1_win_probability'] > result['team2_win_probability']:
        st.markdown("<h3 style='text-align: center; color: #e74c3c;'>Team 1 Wins!</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='text-align: center; color: #3498db;'>Team 2 Wins!</h3>", unsafe_allow_html=True)
    
    # Win probabilities
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='metric-card' style='background-color: #e74c3c;'>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='text-align: center; color: white;'>Team 1 Win Probability</h4>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align: center; color: white;'>{result['team1_win_probability']:.2%}</h2>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card' style='background-color: #3498db;'>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='text-align: center; color: white;'>Team 2 Win Probability</h4>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align: center; color: white;'>{result['team2_win_probability']:.2%}</h2>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Confidence
    st.markdown(f"<p style='text-align: center;'><strong>Prediction Confidence:</strong> {result['confidence']:.2%}</p>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display top contributing features
    st.subheader("Top Contributing Factors")
    
    # Create horizontal bar chart for feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    features = list(result['top_features'].keys())[:5]
    values = list(result['top_features'].values())[:5]
    
    # Determine colors based on whether feature favors team 1 or team 2
    colors = ['#e74c3c' if v > 0 else '#3498db' for v in values]
    
    # Create horizontal bar chart
    bars = ax.barh(features, values, color=colors)
    
    # Add labels
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 0.01 if width > 0 else width - 0.01
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f"{width:.3f}", 
                ha='left' if width > 0 else 'right', va='center')
    
    ax.set_xlabel('Impact on Prediction')
    ax.set_title('Top 5 Features Influencing Battle Outcome')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Favors Team 1'),
        Patch(facecolor='#3498db', label='Favors Team 2')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    st.pyplot(fig)
    
    # Display explanation
    st.subheader("Battle Analysis")
    
    # Generate explanation based on top features
    explanation = "Key factors in this battle prediction:\n\n"
    for feature, value in list(result['top_features'].items())[:3]:
        if value > 0:
            explanation += f"- {feature}: Favors Team 1 (+{value:.3f})\n"
        else:
            explanation += f"- {feature}: Favors Team 2 ({value:.3f})\n"
    
    st.text(explanation)

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