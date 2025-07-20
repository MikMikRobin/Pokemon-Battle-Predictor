# 🔮 Pokémon Battle Predictor

![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.8%2B-FF4B4B?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24%2B-F7931E?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-MIT-green)

A machine learning application that predicts the winning team in Pokémon battles with probability distributions, feature importance analysis, and interactive visualizations.

<div align="center">
  <img src="https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/6.png" width="100" />
  <img src="https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/9.png" width="100" />
  <img src="https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/150.png" width="100" />
  <img src="https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/445.png" width="100" />
  <img src="https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/460.png" width="100" />
</div>

## ✨ Features

- **🎮 Interactive Streamlit App** with three modes:
  - **📊 Analytics Mode**: Visualize Pokémon stats and battle data
  - **🧠 Model Training Mode**: Train custom prediction models
  - **⚔️ Battle Simulation Mode**: Predict outcomes between custom teams

- **🔍 Advanced Feature Engineering**:
  - Type effectiveness calculations
  - Team-level stat aggregations
  - Matchup-specific features
  - Speed tier analysis
  - Type coverage metrics

- **📈 Model Interpretability**:
  - Feature importance ranking
  - SHAP-based explanations
  - Individual prediction breakdowns
  - Visual battle analysis

- **🖼️ Visual Elements**:
  - Pokémon sprites from PokéAPI
  - Battle visualization with team sprites
  - Stat charts and comparisons
  - Feature importance visualizations

- **📊 Performance Metrics**:
  - Accuracy
  - ROC-AUC
  - Brier Score (probability calibration)
  - Log Loss

## 🚀 Project Overview

This project implements a machine learning-based battle prediction system for Pokémon, supporting two battle formats:
1. **VGC Format (2v2)**: Choose 2 Pokémon from a team of 4
2. **Standard Format (6v6)**: Full team battles with 6 Pokémon

The models provide:
- Win probability predictions
- Feature importance analysis
- Interpretable explanations for predictions
- Calibrated probability estimates
- Visual battle simulations with Pokémon sprites

## 📂 Project Structure

```
pokemon-battle-predictor/
├── analytics.py            # Analytics and visualization functions
├── config.py               # Configuration parameters
├── data_loader.py          # Data loading and preprocessing
├── feature_engineering.py  # Feature creation and engineering
├── models.py               # ML model implementations
├── pokemon_battle_app.py   # Streamlit application
├── pokemon_utils.py        # Pokémon sprite and visualization utilities
├── train.py                # Training script
├── predict.py              # Prediction script
├── run_train.py            # Wrapper script for training with environment variables
├── test_prediction.py      # Test script for predictions
├── data/                   # Data files
│   ├── pokemon.csv         # Pokémon stats
│   ├── combats.csv         # 1v1 battle results
│   ├── team_combat.csv     # Team battle outcomes
│   └── pokemon_id_each_team.csv  # Team composition data
├── models/                 # Saved model files
│   ├── vgc_model.pkl       # VGC battle model
│   └── standard_model.pkl  # Standard battle model
├── sprites/                # Downloaded Pokémon sprites
└── plots/                  # Generated analytics plots
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/username/pokemon-battle-predictor.git
cd pokemon-battle-predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🖥️ Usage

### 🚀 Running the Streamlit App

Launch the interactive application:
```bash
python -m streamlit run pokemon_battle_app.py
```

### 🧠 Training Models

Train both VGC and Standard battle models:
```bash
python run_train.py --format both --model-type random_forest
```

Train only VGC model with logistic regression:
```bash
python run_train.py --format vgc --model-type logistic_regression
```

### 🔮 Making Predictions

Predict VGC battle outcome:
```bash
python predict.py --format vgc --team1 6 9 3 149 --team2 131 59 94 130
```

Predict Standard battle outcome with sprite visualization:
```bash
python predict.py --format standard --team1 6 9 3 149 25 143 --team2 131 59 94 130 65 248 --show-sprites
```

Save prediction to file:
```bash
python predict.py --format vgc --team1 6 9 3 149 --team2 131 59 94 130 --output predictions/result.json
```

## 🧪 Model Details

### 🌲 Random Forest (Primary Model)
- Provides excellent probability calibration
- High interpretability with feature importance
- Good performance across different team compositions
- Handles class imbalance with balanced weights

### 📊 Logistic Regression (Alternative Model)
- Maximum interpretability
- Linear feature relationships
- Faster training and prediction
- Calibrated probabilities with CalibratedClassifierCV

## 🖼️ Sprite Visualization

The application uses the PokéAPI to display Pokémon sprites:
- Team selection with Pokémon sprites
- Battle visualization with opposing teams
- Type-colored placeholders for missing sprites
- Sprite caching for improved performance

## 📋 Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- shap
- joblib
- streamlit
- imbalanced-learn
- requests
- pillow
- tqdm

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [PokéAPI](https://pokeapi.co/) for Pokémon sprites and data
- [Kaggle Pokémon Dataset](https://www.kaggle.com/datasets/abcsds/pokemon) for the base Pokémon data
- [Streamlit](https://streamlit.io/) for the interactive web application framework 