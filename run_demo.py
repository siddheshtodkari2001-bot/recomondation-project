#!/usr/bin/env python3
"""
Demo script for the Movie Recommendation System
This script provides a quick way to test the core functionality
without running the full Streamlit app.
"""

import pandas as pd
import numpy as np
from utils import (
    load_data, convert_to_implicit_feedback, create_user_item_matrix,
    calculate_sparsity, train_model, get_recommendations, get_popular_items,
    calculate_precision_at_k
)
from config import MODEL_CONFIG, DATA_CONFIG

def run_demo():
    """Run a demonstration of the recommendation system."""
    print("🎬 Movie Recommendation System Demo")
    print("=" * 50)
    
    # Load data
    print("\n📊 Loading MovieLens dataset...")
    data, ratings_df, movies_df = load_data()
    
    print(f"✅ Dataset loaded successfully!")
    print(f"   - Users: {len(ratings_df['userId'].unique())}")
    print(f"   - Movies: {len(ratings_df['movieId'].unique())}")
    print(f"   - Ratings: {len(ratings_df)}")
    
    # Calculate sparsity
    print("\n📈 Calculating matrix sparsity...")
    matrix = create_user_item_matrix(ratings_df)
    sparsity = calculate_sparsity(matrix)
    print(f"   - Matrix sparsity: {sparsity:.2%}")
    
    # Train model
    print("\n🤖 Training SVD model...")
    model_dict = train_model(ratings_df)
    print(f"   - RMSE: {model_dict['rmse']:.3f}")
    
    # Calculate Precision@K
    print("\n📊 Calculating Precision@K...")
    precision_at_10 = calculate_precision_at_k(model_dict, k=10)
    print(f"   - Precision@10: {precision_at_10:.3f}")
    
    # Get recommendations for a sample user
    print("\n🎯 Generating recommendations...")
    sample_user = ratings_df['userId'].iloc[0]
    recommendations = get_recommendations(model_dict, sample_user, movies_df, n_recommendations=5)
    
    print(f"\n📋 Top 5 recommendations for User {sample_user}:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec['title']} (Predicted: ⭐ {rec['predicted_rating']})")
    
    # Get popular items
    print("\n🔥 Top 5 popular movies:")
    popular_items = get_popular_items(ratings_df, movies_df, n_items=5)
    for i, (_, item) in enumerate(popular_items.iterrows(), 1):
        print(f"   {i}. {item['title']} (⭐ {item['avg_rating']:.2f}, {item['rating_count']} ratings)")
    
    # Show some statistics
    print("\n📊 Dataset Statistics:")
    print(f"   - Average rating: {ratings_df['rating'].mean():.2f}")
    print(f"   - Rating std dev: {ratings_df['rating'].std():.2f}")
    print(f"   - Most common rating: {ratings_df['rating'].mode().iloc[0]}")
    
    # Genre analysis
    print("\n🎭 Top 5 genres by average rating:")
    genre_stats = []
    for _, row in movies_df.iterrows():
        genres = row['genres'].split('|')
        for genre in genres:
            genre_stats.append({
                'movieId': row['movieId'],
                'genre': genre.strip()
            })
    
    genre_df = pd.DataFrame(genre_stats)
    genre_ratings = ratings_df.merge(genre_df, on='movieId')
    genre_analysis = genre_ratings.groupby('genre').agg({
        'rating': ['mean', 'count']
    }).reset_index()
    genre_analysis.columns = ['genre', 'avg_rating', 'rating_count']
    genre_analysis = genre_analysis[genre_analysis['rating_count'] >= 10]
    genre_analysis = genre_analysis.sort_values('avg_rating', ascending=False)
    
    for i, (_, genre) in enumerate(genre_analysis.head(5).iterrows(), 1):
        print(f"   {i}. {genre['genre']} (⭐ {genre['avg_rating']:.2f}, {genre['rating_count']} ratings)")
    
    print("\n✅ Demo completed successfully!")
    print("\n🚀 To run the full Streamlit app, use:")
    print("   streamlit run app.py")

if __name__ == "__main__":
    run_demo() 