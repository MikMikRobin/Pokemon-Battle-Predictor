# Pokemon Battle Predictor

A machine learning model to predict the winning team in Pokemon battles with probability distributions and feature importance analysis.

## Project Overview

This project implements a machine learning-based battle prediction system for Pokemon, supporting two battle formats:
1. **VGC Format (2v2)**: Choose 2 Pokemon from a team of 4
2. **Standard Format (6v6)**: Full team battles with 6 Pokemon

The models provide:
- Win probability predictions
- Feature importance analysis
- Interpretable explanations for predictions
- Calibrated probability estimates

## Dataset

The project uses the following datasets:
- `pokemon.csv`: Contains Pokemon stats (HP, Attack, Defense, etc.)
- `combats.csv`: 1v1 battle results
- `team_combat.csv`: Team battle outcomes
- `pokemon_id_each_team.csv`: Maps team IDs to Pokemon IDs

## Project Structure

```
pokemon-battle-predictor/
├── config.py               # Configuration parameters
├── data_loader.py          # Data loading and preprocessing
├── feature_engineering.py  # Feature creation and engineering
├── models.py               # ML model implementations
├── train.py                # Training script
├── predict.py              # Prediction script
├── models/                 # Saved model files
│   ├── vgc_model.pkl       # VGC battle model
│   └── standard_model.pkl  # Standard battle model
└── README.md               # Project documentation
```

## Features

- **Advanced Feature Engineering**:
  - Type effectiveness calculations
  - Team-level stat aggregations
  - Matchup-specific features
  - Speed tier analysis
  - Type coverage metrics

- **Model Interpretability**:
  - Feature importance ranking
  - SHAP-based explanations
  - Individual prediction breakdowns

- **Performance Metrics**:
  - Accuracy
  - ROC-AUC
  - Brier Score (probability calibration)
  - Log Loss

## Installation

1. Clone the repository:
```bash
git clone https://github.com/username/pokemon-battle-predictor.git
cd pokemon-battle-predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training Models

Train both VGC and Standard battle models:
```bash
python train.py --format both --model-type random_forest
```

Train only VGC model with logistic regression:
```bash
python train.py --format vgc --model-type logistic_regression
```

### Making Predictions

Predict VGC battle outcome:
```bash
python predict.py --format vgc --team1 6 9 3 149 --team2 131 59 94 130
```

Predict Standard battle outcome:
```bash
python predict.py --format standard --team1 6 9 3 149 25 143 --team2 131 59 94 130 65 248
```

Save prediction to file:
```bash
python predict.py --format vgc --team1 6 9 3 149 --team2 131 59 94 130 --output predictions/result.json
```

## Model Details

### Random Forest (Primary Model)
- Provides excellent probability calibration
- High interpretability with feature importance
- Good performance across different team compositions

### Logistic Regression (Alternative Model)
- Maximum interpretability
- Linear feature relationships
- Faster training and prediction

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- shap
- joblib

## License

This project is licensed under the MIT License - see the LICENSE file for details. 