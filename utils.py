import pandas as pd
import numpy as np
import requests
import zipfile
import os
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def download_movielens_data():
    """Download MovieLens 100K dataset if not already present."""
    from config import DATA_CONFIG
    
    data_dir = "ml-latest-small"
    
    if not os.path.exists(data_dir):
        print("Downloading MovieLens 100K dataset...")
        url = DATA_CONFIG['dataset_url']
        response = requests.get(url)
        
        with open("ml-latest-small.zip", "wb") as f:
            f.write(response.content)
        
        with zipfile.ZipFile("ml-latest-small.zip", "r") as zip_ref:
            zip_ref.extractall(".")
        
        os.remove("ml-latest-small.zip")
        print("Dataset downloaded successfully!")
    
    return data_dir

def load_data():
    """Load and preprocess MovieLens data."""
    data_dir = download_movielens_data()
    
    # Load ratings and movies
    ratings_df = pd.read_csv(f"{data_dir}/ratings.csv")
    movies_df = pd.read_csv(f"{data_dir}/movies.csv")
    
    # Merge ratings with movie information
    data = ratings_df.merge(movies_df, on='movieId')
    
    return data, ratings_df, movies_df

def convert_to_implicit_feedback(ratings_df):
    """Convert explicit ratings to implicit feedback."""
    from config import DATA_CONFIG
    
    # Create implicit feedback: rating >= threshold is positive interaction
    implicit_df = ratings_df.copy()
    threshold = DATA_CONFIG['rating_threshold']
    implicit_df['implicit_rating'] = (implicit_df['rating'] >= threshold).astype(int)
    
    # Alternative: weighted implicit feedback
    implicit_df['weighted_rating'] = np.where(
        implicit_df['rating'] >= 4, 3,  # High rating = strong positive
        np.where(implicit_df['rating'] >= 3, 2,  # Medium rating = positive
                np.where(implicit_df['rating'] >= 2, 1, 0))  # Low rating = weak positive
    )
    
    return implicit_df

def create_user_item_matrix(ratings_df, implicit=False):
    """Create user-item interaction matrix."""
    if implicit:
        # Use implicit feedback
        matrix = ratings_df.pivot_table(
            index='userId', 
            columns='movieId', 
            values='implicit_rating', 
            fill_value=0
        )
    else:
        # Use explicit ratings
        matrix = ratings_df.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating', 
            fill_value=0
        )
    
    return matrix

def calculate_sparsity(matrix):
    """Calculate sparsity of the user-item matrix."""
    total_elements = matrix.shape[0] * matrix.shape[1]
    non_zero_elements = np.count_nonzero(matrix.values)
    sparsity = 1 - (non_zero_elements / total_elements)
    return sparsity

def train_model(ratings_df, implicit=False):
    """Train SVD model using scikit-learn."""
    from config import MODEL_CONFIG, DATA_CONFIG
    
    # Create user-item matrix
    matrix = create_user_item_matrix(ratings_df, implicit=implicit)
    
    # Split data for evaluation
    # Create train/test split by randomly masking some ratings
    train_matrix = matrix.copy()
    test_ratings = []
    
    # Get non-zero ratings for testing
    non_zero_mask = matrix != 0
    non_zero_indices = np.where(non_zero_mask)
    
    # Randomly select 20% for testing
    n_test = int(0.2 * len(non_zero_indices[0]))
    test_indices = np.random.choice(
        len(non_zero_indices[0]), 
        size=n_test, 
        replace=False
    )
    
    # Store test ratings and mask them in training matrix
    for idx in test_indices:
        user_idx = non_zero_indices[0][idx]
        item_idx = non_zero_indices[1][idx]
        true_rating = matrix.iloc[user_idx, item_idx]
        
        test_ratings.append({
            'user_idx': user_idx,
            'item_idx': item_idx,
            'true_rating': true_rating
        })
        
        # Mask in training matrix
        train_matrix.iloc[user_idx, item_idx] = 0
    
    # Fill NaN values with 0 for SVD
    train_matrix = train_matrix.fillna(0)
    
    # Train SVD model
    svd = TruncatedSVD(
        n_components=MODEL_CONFIG['n_factors'],
        random_state=MODEL_CONFIG['random_state']
    )
    
    # Fit the model
    svd.fit(train_matrix)
    
    # Transform the matrix
    user_factors = svd.transform(train_matrix)
    item_factors = svd.components_.T
    
    # Calculate RMSE on test set
    predictions = []
    actuals = []
    
    for test_rating in test_ratings:
        user_idx = test_rating['user_idx']
        item_idx = test_rating['item_idx']
        true_rating = test_rating['true_rating']
        
        # Predict rating
        pred_rating = np.dot(user_factors[user_idx], item_factors[item_idx])
        
        predictions.append(pred_rating)
        actuals.append(true_rating)
    
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    
    return {
        'svd': svd,
        'user_factors': user_factors,
        'item_factors': item_factors,
        'train_matrix': train_matrix,
        'test_ratings': test_ratings,
        'rmse': rmse
    }

def get_recommendations(model_dict, user_id, movies_df, n_recommendations=10):
    """Get personalized recommendations for a user."""
    # Get user index
    user_indices = model_dict['train_matrix'].index
    if user_id not in user_indices:
        return []
    
    user_idx = user_indices.get_loc(user_id)
    
    # Get user factors
    user_factors = model_dict['user_factors'][user_idx]
    item_factors = model_dict['item_factors']
    
    # Predict ratings for all movies
    predictions = np.dot(user_factors, item_factors.T)
    
    # Get movie IDs
    movie_ids = model_dict['train_matrix'].columns
    
    # Create list of (movie_id, predicted_rating) tuples
    movie_predictions = list(zip(movie_ids, predictions))
    
    # Sort by predicted rating
    movie_predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Get top N recommendations
    top_movies = movie_predictions[:n_recommendations]
    
    # Get movie details
    recommendations = []
    for movie_id, pred_rating in top_movies:
        movie_info = movies_df[movies_df['movieId'] == movie_id]
        if not movie_info.empty:
            movie_info = movie_info.iloc[0]
            recommendations.append({
                'movieId': movie_id,
                'title': movie_info['title'],
                'genres': movie_info['genres'],
                'predicted_rating': round(pred_rating, 2)
            })
    
    return recommendations

def get_popular_items(ratings_df, movies_df, n_items=10):
    """Get most popular items based on number of ratings."""
    # Count ratings per movie
    movie_counts = ratings_df.groupby('movieId').size().reset_index(name='rating_count')
    movie_avg_ratings = ratings_df.groupby('movieId')['rating'].mean().reset_index(name='avg_rating')
    
    # Merge with movie information
    popular_movies = movie_counts.merge(movie_avg_ratings, on='movieId')
    popular_movies = popular_movies.merge(movies_df, on='movieId')
    
    # Sort by rating count and average rating
    popular_movies = popular_movies.sort_values(['rating_count', 'avg_rating'], ascending=[False, False])
    
    return popular_movies.head(n_items)

def calculate_precision_at_k(model_dict, k=10):
    """Calculate Precision@K for model evaluation."""
    from config import EVAL_CONFIG
    
    test_ratings = model_dict['test_ratings']
    user_factors = model_dict['user_factors']
    item_factors = model_dict['item_factors']
    train_matrix = model_dict['train_matrix']
    
    # Group test ratings by user
    user_predictions = {}
    for test_rating in test_ratings:
        user_idx = test_rating['user_idx']
        item_idx = test_rating['item_idx']
        true_rating = test_rating['true_rating']
        
        if user_idx not in user_predictions:
            user_predictions[user_idx] = []
        
        # Predict rating
        pred_rating = np.dot(user_factors[user_idx], item_factors[item_idx])
        user_predictions[user_idx].append((item_idx, pred_rating, true_rating))
    
    # Calculate Precision@K for each user
    precisions = []
    for user_idx, user_preds in user_predictions.items():
        # Sort by predicted rating
        user_preds.sort(key=lambda x: x[1], reverse=True)
        
        # Get top K predictions
        top_k = user_preds[:k]
        
        # Count relevant items (rating >= threshold)
        threshold = EVAL_CONFIG['rating_threshold']
        relevant_count = sum(1 for _, _, actual in top_k if actual >= threshold)
        precision = relevant_count / k if k > 0 else 0
        precisions.append(precision)
    
    return np.mean(precisions) if precisions else 0

def plot_rating_distribution(ratings_df, dark_theme=False):
    """Create histogram of ratings distribution."""
    plt.figure(figsize=(10, 6))
    
    if dark_theme:
        plt.style.use('dark_background')
        sns.set_style("darkgrid")
        colors = ['#4CAF50']
    else:
        plt.style.use('default')
        sns.set_style("whitegrid")
        colors = ['#1f77b4']
    
    sns.histplot(data=ratings_df, x='rating', bins=9, kde=True, color=colors[0])
    plt.title('Distribution of Movie Ratings', color='white' if dark_theme else 'black')
    plt.xlabel('Rating', color='white' if dark_theme else 'black')
    plt.ylabel('Count', color='white' if dark_theme else 'black')
    plt.xticks(range(1, 6))
    
    if dark_theme:
        plt.gca().tick_params(colors='white')
        plt.gca().spines['bottom'].set_color('white')
        plt.gca().spines['left'].set_color('white')
    
    return plt.gcf()

def plot_user_activity(ratings_df, dark_theme=False):
    """Create histogram of user activity (number of ratings per user)."""
    user_activity = ratings_df.groupby('userId').size()
    
    plt.figure(figsize=(10, 6))
    
    if dark_theme:
        plt.style.use('dark_background')
        sns.set_style("darkgrid")
        colors = ['#4CAF50']
    else:
        plt.style.use('default')
        sns.set_style("whitegrid")
        colors = ['#1f77b4']
    
    sns.histplot(user_activity, bins=50, kde=True, color=colors[0])
    plt.title('Distribution of User Activity', color='white' if dark_theme else 'black')
    plt.xlabel('Number of Ratings per User', color='white' if dark_theme else 'black')
    plt.ylabel('Count', color='white' if dark_theme else 'black')
    
    if dark_theme:
        plt.gca().tick_params(colors='white')
        plt.gca().spines['bottom'].set_color('white')
        plt.gca().spines['left'].set_color('white')
    
    return plt.gcf() 