import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import (
    load_data, convert_to_implicit_feedback, create_user_item_matrix,
    calculate_sparsity, train_model, get_recommendations, get_popular_items,
    calculate_precision_at_k, plot_rating_distribution, plot_user_activity
)

# Set page config
from config import UI_CONFIG

st.set_page_config(
    page_title=UI_CONFIG['page_title'],
    page_icon=UI_CONFIG['page_icon'],
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme toggle in sidebar
st.sidebar.subheader("🎨 Theme Settings")
theme_mode = st.sidebar.selectbox(
    "Choose Theme:",
    ["Light", "Dark"],
    index=1  # Default to dark mode
)

# Store theme preference in session state
if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = theme_mode
elif st.session_state.theme_mode != theme_mode:
    st.session_state.theme_mode = theme_mode
    st.rerun()

# Dynamic CSS based on theme
if theme_mode == "Dark":
    css_theme = """
    <style>
        /* Dark theme styles */
        .main-header {
            font-size: 3rem;
            color: #4CAF50;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #2d3748;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #4CAF50;
            color: white;
        }
        .recommendation-card {
            background-color: #2d3748;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #4a5568;
            margin-bottom: 0.5rem;
            color: white;
        }
        .recommendation-card strong {
            color: #4CAF50;
        }
        .recommendation-card small {
            color: #a0aec0;
        }
        /* Override Streamlit default text colors for dark theme */
        .stMarkdown, .stText, p, div {
            color: white !important;
        }
        /* Make headers more visible */
        h1, h2, h3, h4, h5, h6 {
            color: #4CAF50 !important;
        }
        /* Style sidebar text for dark theme */
        .css-1d391kg {
            color: white !important;
        }
        /* Style metric labels and values */
        .metric-container label, .metric-container div {
            color: white !important;
        }
        /* Style metrics for better visibility */
        .metric-container {
            background-color: #2d3748;
            border-radius: 0.5rem;
            padding: 1rem;
            border: 1px solid #4a5568;
        }
        /* Style dataframes for dark theme */
        .dataframe {
            background-color: #2d3748 !important;
            color: white !important;
        }
        .dataframe th {
            background-color: #4a5568 !important;
            color: white !important;
        }
        .dataframe td {
            background-color: #2d3748 !important;
            color: white !important;
        }
        /* Style charts for dark theme */
        .stPlotlyChart {
            background-color: #2d3748 !important;
        }
        /* Style selectboxes and other widgets */
        .stSelectbox, .stSlider, .stCheckbox {
            color: white !important;
        }
        /* Style the main content area */
        .main .block-container {
            background-color: #1a202c !important;
        }
        /* Style the sidebar */
        .css-1d391kg {
            background-color: #2d3748 !important;
        }
        /* Style buttons */
        .stButton > button {
            background-color: #4CAF50 !important;
            color: white !important;
            border: none !important;
        }
        .stButton > button:hover {
            background-color: #45a049 !important;
        }
    </style>
    """
else:
    css_theme = """
    <style>
        /* Light theme styles */
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
            color: #333;
        }
        .recommendation-card {
            background-color: #ffffff;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #e0e0e0;
            margin-bottom: 0.5rem;
            color: #333;
        }
        .recommendation-card strong {
            color: #1f77b4;
        }
        .recommendation-card small {
            color: #666;
        }
        /* Style metrics for light theme */
        .metric-container {
            background-color: #f0f2f6;
            border-radius: 0.5rem;
            padding: 1rem;
            border: 1px solid #e0e0e0;
        }
        /* Style buttons for light theme */
        .stButton > button {
            background-color: #1f77b4 !important;
            color: white !important;
            border: none !important;
        }
        .stButton > button:hover {
            background-color: #1565c0 !important;
        }
    </style>
    """

st.markdown(css_theme, unsafe_allow_html=True)

@st.cache_data
def load_cached_data():
    """Load data with caching to avoid reloading."""
    with st.spinner("Loading MovieLens dataset..."):
        data, ratings_df, movies_df = load_data()
    return data, ratings_df, movies_df

@st.cache_resource
def train_cached_model(ratings_df, use_implicit=False):
    """Train model with caching."""
    with st.spinner("Training recommendation model..."):
        model_dict = train_model(ratings_df, implicit=use_implicit)
    return model_dict

def main():
    # Header
    st.markdown('<h1 class="main-header">🎬 Movie Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown("### Personalized movie recommendations using collaborative filtering")
    
    # Load data
    data, ratings_df, movies_df = load_cached_data()
    
    # Sidebar
    st.sidebar.header("🎛️ Controls")
    
    # User selection
    user_ids = sorted(ratings_df['userId'].unique())
    selected_user = st.sidebar.selectbox(
        "Select User ID:",
        user_ids,
        index=0
    )
    
    # Number of recommendations
    n_recommendations = st.sidebar.slider(
        "Number of Recommendations:",
        min_value=UI_CONFIG['min_recommendations'],
        max_value=UI_CONFIG['max_recommendations'],
        value=UI_CONFIG['default_recommendations'],
        step=1
    )
    
    # Model options
    st.sidebar.subheader("Model Settings")
    use_implicit = st.sidebar.checkbox("Use Implicit Feedback", value=False)
    
    # Retrain button
    if st.sidebar.button("🔄 Retrain Model", type="primary"):
        st.cache_resource.clear()
        st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Dataset Overview")
        
        # Dataset preview
        st.subheader("Sample Data")
        st.dataframe(data.head(10), use_container_width=True)
        
        # Basic statistics
        st.subheader("📈 Dataset Statistics")
        
        # Create metrics in columns
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Total Users", len(ratings_df['userId'].unique()))
        
        with metric_col2:
            st.metric("Total Movies", len(ratings_df['movieId'].unique()))
        
        with metric_col3:
            st.metric("Total Ratings", len(ratings_df))
        
        with metric_col4:
            # Calculate sparsity
            matrix = create_user_item_matrix(ratings_df, implicit=use_implicit)
            sparsity = calculate_sparsity(matrix)
            st.metric("Matrix Sparsity", f"{sparsity:.2%}")
        
        # Rating distribution
        st.subheader("📊 Rating Distribution")
        fig = plot_rating_distribution(ratings_df, dark_theme=(theme_mode == "Dark"))
        st.pyplot(fig)
        plt.close()
        
        # User activity distribution
        st.subheader("👥 User Activity Distribution")
        fig = plot_user_activity(ratings_df, dark_theme=(theme_mode == "Dark"))
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.header("🎯 Recommendations")
        
        # Train model
        model_dict = train_cached_model(ratings_df, use_implicit)
        
        # Model performance
        st.subheader("📊 Model Performance")
        from config import EVAL_CONFIG
        precision_at_k = calculate_precision_at_k(model_dict, k=EVAL_CONFIG['default_k'])
        
        perf_col1, perf_col2 = st.columns(2)
        with perf_col1:
            st.metric("RMSE", f"{model_dict['rmse']:.3f}")
        with perf_col2:
            st.metric("Precision@10", f"{precision_at_k:.3f}")
        
        # Get recommendations
        try:
            recommendations = get_recommendations(model_dict, selected_user, movies_df, n_recommendations)
            
            st.subheader(f"🎬 Personalized Recommendations for User {selected_user}")
            
            for i, rec in enumerate(recommendations, 1):
                with st.container():
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <strong>{i}. {rec['title']}</strong><br>
                        <small>Genres: {rec['genres']}</small><br>
                        <small>Predicted Rating: ⭐ {rec['predicted_rating']}</small>
                    </div>
                    """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error getting recommendations: {str(e)}")
            st.info("Showing popular items instead...")
            
            # Fallback to popular items
            popular_items = get_popular_items(ratings_df, movies_df, n_recommendations)
            
            st.subheader("🔥 Popular Movies")
            for i, (_, item) in enumerate(popular_items.iterrows(), 1):
                with st.container():
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <strong>{i}. {item['title']}</strong><br>
                        <small>Genres: {item['genres']}</small><br>
                        <small>Rating: ⭐ {item['avg_rating']:.2f} ({item['rating_count']} ratings)</small>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Additional analysis section
    st.header("🔍 Detailed Analysis")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("📈 Top Rated Movies")
        top_rated = ratings_df.groupby('movieId').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        top_rated.columns = ['movieId', 'avg_rating', 'rating_count']
        from config import DATA_CONFIG
        top_rated = top_rated[top_rated['rating_count'] >= DATA_CONFIG['min_ratings_per_movie']]  # Minimum ratings
        top_rated = top_rated.sort_values('avg_rating', ascending=False)
        top_rated = top_rated.merge(movies_df, on='movieId')
        
        st.dataframe(
            top_rated[['title', 'avg_rating', 'rating_count']].head(10),
            use_container_width=True
        )
    
    with col4:
        st.subheader("📊 Rating Statistics by Genre")
        # Extract genres and create genre-based analysis
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
        genre_analysis = genre_analysis[genre_analysis['rating_count'] >= DATA_CONFIG['min_ratings_per_movie']]
        genre_analysis = genre_analysis.sort_values('avg_rating', ascending=False)
        
        st.dataframe(genre_analysis.head(10), use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with Streamlit • MovieLens 100K Dataset • Collaborative Filtering</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 