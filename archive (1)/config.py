"""
Configuration file for Pokemon Battle Predictor.
Contains paths, model parameters, and other configurations.
"""

# Data paths
POKEMON_CSV = "pokemon.csv"
COMBATS_CSV = "combats.csv"
TEAM_COMBAT_CSV = "team_combat.csv"
POKEMON_TEAM_CSV = "pokemon_id_each_team.csv"

# Model parameters
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Feature engineering parameters
TEAM_AGGREGATION_METHODS = ["sum", "mean", "max", "min", "std"]

# Battle formats
class BattleFormat:
    VGC_2V2 = "vgc_2v2"  # 2v2 format (choose 2 from 4)
    STANDARD_6V6 = "standard_6v6"  # Full 6v6 battles

# Model hyperparameters
RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 15,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "random_state": RANDOM_SEED
}

LOGISTIC_REGRESSION_PARAMS = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "liblinear",
    "random_state": RANDOM_SEED
} 