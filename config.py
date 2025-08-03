# Configuration file for the Movie Recommendation System

# Model Parameters
MODEL_CONFIG = {
    'n_factors': 100,          # Number of latent factors
    'n_epochs': 20,            # Number of training epochs
    'lr_all': 0.005,           # Learning rate
    'reg_all': 0.02,           # Regularization parameter
    'random_state': 42         # Random seed for reproducibility
}

# Data Configuration
DATA_CONFIG = {
    'test_size': 0.2,          # Test set size (20%)
    'min_ratings_per_movie': 10,  # Minimum ratings for movie analysis
    'rating_threshold': 3.5,   # Threshold for positive implicit feedback
    'dataset_url': 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
}

# UI Configuration
UI_CONFIG = {
    'max_recommendations': 20,  # Maximum number of recommendations
    'min_recommendations': 5,   # Minimum number of recommendations
    'default_recommendations': 10,  # Default number of recommendations
    'page_title': 'Movie Recommendation System',
    'page_icon': '🎬'
}

# Evaluation Configuration
EVAL_CONFIG = {
    'k_values': [5, 10, 20],   # K values for Precision@K
    'default_k': 10,           # Default K for Precision@K
    'rating_threshold': 3.5    # Threshold for relevant items
}

# Visualization Configuration
VIZ_CONFIG = {
    'figure_size': (10, 6),    # Default figure size
    'dpi': 100,                # Figure DPI
    'style': 'seaborn-v0_8',   # Matplotlib style
    'color_palette': 'viridis' # Color palette for plots
} 