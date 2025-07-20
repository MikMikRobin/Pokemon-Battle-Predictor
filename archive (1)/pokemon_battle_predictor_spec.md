# Pokemon Battle Predictor - ML Project Specification

## 🎯 Project Objective
Build a machine learning model to predict the winning team in Pokemon battles with probability distributions and feature importance analysis.

## 📊 Available Datasets

### 1. `combats.csv` - Individual Pokemon Battles
- **Columns**: `First_pokemon`, `Second_pokemon`, `Winner`
- **Description**: 1v1 Pokemon battle results with Pokemon IDs
- **Purpose**: Historical battle outcomes for feature engineering

### 2. `pokemon.csv` - Pokemon Statistics
- **Columns**: Pokemon ID, Name, Type 1, Type 2, HP, Attack, Defense, Sp. Atk, Sp. Def, Speed, Generation, Legendary
- **Description**: Complete Pokemon stat database
- **Purpose**: Core features for model training

### 3. `pokemon_id_each_team.csv` - Team Composition References  
- **Columns**: Team references and Pokemon IDs
- **Description**: Maps team IDs to constituent Pokemon
- **Purpose**: Links individual Pokemon to team battles

### 4. `team_combat.csv` - Team Battle Results
- **Columns**: `first`, `second`, `winner`
- **Description**: Team vs team battle outcomes
- **Purpose**: Training labels for team battle prediction

## 🎮 Battle Formats to Support
1. **2v2 battles** - Choose 2 Pokemon from team of 4 (VGC format)
2. **6v6 battles** - Full team battles (Standard format)

## 🤖 Machine Learning Requirements

### Model Type
- **Primary**: Binary Classification (Team A wins vs Team B wins)
- **Output**: Win probabilities (0-1 scale)
- **Alternative**: Multi-class if predicting victory margins

### Recommended Models (Priority Order)
1. **Random Forest** - Optimal for interpretability + probability calibration
2. **Logistic Regression** - Backup for maximum interpretability

### Key Requirements
- ✅ **Probability distributions**: Must output win probabilities, not just binary predictions
- ✅ **Feature importance**: Must identify which stats/features matter most
- ✅ **Model interpretability**: Use SHAP or similar for detailed explanations
- ✅ **Calibrated probabilities**: Reliable probability estimates for decision-making

## 🔧 Feature Engineering Strategy

### Team-Level Aggregations
```python
# Example features to create:
- team_total_attack = sum(pokemon_attacks)
- team_avg_speed = mean(pokemon_speeds)  
- team_max_hp = max(pokemon_hp)
- team_stat_variance = std(pokemon_stats)
```

### Matchup Features
- **Stat differentials**: Team A total attack vs Team B total defense
- **Type effectiveness**: Calculate type advantage matrices
- **Speed control**: Which team controls speed tiers
- **Balanced vs specialized**: Team composition analysis

### Historical Performance
- **Win rates**: Individual Pokemon performance from combats.csv
- **Meta features**: Pokemon usage rates, tier classifications

### Advanced Features
- **Type coverage**: Defensive type matchups
- **Stat distributions**: Range and variance within teams
- **Speed tiers**: Critical speed breakpoints
- **Legendary presence**: Impact of legendary Pokemon

## 📈 Model Evaluation

### Metrics to Track
- **Accuracy**: Overall prediction correctness
- **ROC-AUC**: Probability ranking quality
- **Brier Score**: Probability calibration quality  
- **Log Loss**: Penalizes confident wrong predictions
- **Precision/Recall**: For each class (Team A/B wins)

### Validation Strategy
- **Stratified K-Fold**: Maintain win/loss balance
- **Format-specific validation**: Separate 2v2 and 6v6 performance
- **Time-based splits**: If temporal data available
- **Cross-validation**: 5-fold recommended

### Probability Calibration
- **Calibration plots**: Reliability diagrams
- **Platt scaling**: Post-training probability calibration
- **Isotonic regression**: Non-parametric calibration alternative

## 🔍 Interpretability Analysis Required

### Global Feature Importance
```python
# Methods to implement:
- feature_importance_gain
- permutation_importance  
- shap_feature_importance
- partial_dependence_plots
```

### Individual Prediction Explanations
```python
# For each prediction, show:
- shap_waterfall_plot(prediction)
- feature_contributions(team_a, team_b)
- confidence_intervals(prediction)
```

### Insights to Extract
- Which stats matter most in different formats?
- How do type matchups impact win probability?
- What team compositions are most effective?
- Critical stat thresholds and breakpoints

## 📋 Implementation Checklist

### Data Preprocessing
- [ ] Load and merge all datasets correctly
- [ ] Handle missing values (especially Type 2)
- [ ] Encode categorical variables (types, generations)
- [ ] Create team-level aggregated features
- [ ] Generate matchup differential features
- [ ] Split data by battle format (2v2 vs 6v6)

### Model Development
- [ ] Implement Random Forest baseline
- [ ] Add Logistic Regression comparison
- [ ] Hyperparameter tuning (GridSearch/RandomSearch)
- [ ] Cross-validation implementation
- [ ] Probability calibration

### Interpretability
- [ ] SHAP integration for feature explanations
- [ ] Feature importance rankings
- [ ] Individual prediction breakdowns
- [ ] Partial dependence plots
- [ ] Feature interaction analysis

### Output Requirements
- [ ] Win probability for each matchup
- [ ] Confidence intervals/prediction uncertainty
- [ ] Top contributing features per prediction
- [ ] Global feature importance rankings
- [ ] Model performance metrics
- [ ] Calibration assessment

## 🎯 Expected Deliverables

1. **Trained Model**: Serialized model ready for predictions
2. **Prediction Function**: `predict_battle(team_a, team_b) -> probability, explanation`
3. **Feature Analysis**: Report on most important stats/features
4. **Model Performance**: Comprehensive evaluation metrics
5. **Interpretability Dashboard**: SHAP-based explanations and insights

## 📝 Technical Stack
- **pandas**: Data manipulation and analysis
- **scikit-learn**: ML models and evaluation
- **numpy**: Numerical computations
- **matplotlib/seaborn**: Visualization
- **shap**: Model interpretability

## 🚀 Success Criteria
- Model achieves >70% accuracy on held-out test set
- Probability predictions are well-calibrated (Brier score <0.3)
- Clear identification of top 10 most important features
- Actionable insights about battle mechanics and team composition
- Reliable predictions across both 2v2 and 6v6 formats