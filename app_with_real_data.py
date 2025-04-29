import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from predictors import RBMPredictor, DLPredictor, EnsemblePredictor
from datetime import datetime, timedelta
from data_fetcher import JokerDataFetcher

# Page config
st.set_page_config(
    page_title="Joker Predictions",
    page_icon="🎲",
    layout="wide"
)

# Title
st.title("🎲 Joker Prediction App")
st.markdown("""
Welcome to the Joker Prediction App! This page uses simple statistical models for predictions.
For advanced machine learning models, check the 'Advanced Predictions' page in the sidebar.
""")

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .st-emotion-cache-1v0mbdj {
        margin-top: 1rem;
    }
    .st-emotion-cache-1wivap2 {
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Title with emoji and description
st.title("🎲 Joker Prediction Analysis")
st.markdown("""
This app provides predictions and analysis for the Joker lottery game using three different prediction models:
- **RBM Predictor**: Uses Restricted Boltzmann Machine patterns
- **DL Predictor**: Uses Deep Learning patterns from recent draws
- **Ensemble Predictor**: Combines predictions from both models
""")

# Add a sidebar with additional information and controls
with st.sidebar:
    st.header("Controls")
    auto_update = st.checkbox("Auto-update data", value=True)
    update_button = st.button("Update data manually")
    
    st.header("About")
    st.write("This app uses machine learning to analyze Joker lottery patterns and make predictions.")
    
    st.header("How it works")
    st.write("""
    1. Real Joker data is fetched from OPAP API
    2. Three different models analyze patterns
    3. Performance is tracked and visualized
    4. Moving averages show trend lines
    """)
    
    st.header("Disclaimer")
    st.warning("""
    These predictions are for educational purposes only.
    Lottery games are based on chance, and no prediction system can guarantee wins.
    Please gamble responsibly.
    """)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data(force_update=False):
    """Load and preprocess the Joker dataset"""
    fetcher = JokerDataFetcher()
    
    if force_update or auto_update:
        df = fetcher.update_data()
    else:
        df = fetcher.load_existing_data()
    
    if len(df) == 0:
        st.error("No data available. Please check your internet connection and try updating.")
        # Try to generate sample data
        st.info("Generating sample data for demonstration...")
        dates = pd.date_range(start='2024-01-01', end='2024-04-29', freq='3D')
        data = []
        for date in dates:
            numbers = sorted(np.random.choice(range(1, 46), size=5, replace=False))
            joker = np.random.randint(1, 21)
            data.append({
                'date': date,
                'draw_id': len(data) + 1,
                'numbers': numbers,
                'joker': joker
            })
        df = pd.DataFrame(data)
        st.success("Sample data generated successfully!")
    
    return df

def evaluate_predictions(predictor, test_df, name):
    """Evaluate predictor performance on test data"""
    results = []
    correct_numbers = []
    correct_jokers = []
    
    for idx, row in test_df.iterrows():
        actual_numbers = set(row['numbers'])
        actual_joker = row['joker']
        
        predictions = predictor.predict(3)
        best_match = 0
        hit_joker = False
        
        for pred_numbers, pred_joker in predictions:
            matches = len(actual_numbers & set(pred_numbers))
            best_match = max(best_match, matches)
            if pred_joker == actual_joker:
                hit_joker = True
        
        correct_numbers.append(best_match)
        correct_jokers.append(1 if hit_joker else 0)
        
        results.append({
            'draw_date': row['date'],
            'actual_numbers': sorted(actual_numbers),
            'actual_joker': actual_joker,
            'predictions': predictions,
            'best_match': best_match,
            'hit_joker': hit_joker
        })
    
    return {
        'name': name,
        'results': results,
        'avg_numbers': np.mean(correct_numbers),
        'avg_joker': np.mean(correct_jokers),
        'correct_numbers': correct_numbers,
        'correct_jokers': correct_jokers
    }

try:
    # Load data
    with st.spinner('Loading and processing Joker data...'):
        df = load_data(force_update=update_button)
    
    # Show data info
    st.write("## Data Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Draws", len(df))
    with col2:
        st.metric("First Draw", df['date'].min().strftime('%Y-%m-%d'))
    with col3:
        st.metric("Latest Draw", df['date'].max().strftime('%Y-%m-%d'))
    
    # Use last 200 draws for testing
    test_df = df.iloc[-200:]
    train_df = df.iloc[:-200]
    
    st.write(f"Training on {len(train_df)} draws, testing on {len(test_df)} draws")
    st.write(f"Test period: {test_df['date'].min().strftime('%Y-%m-%d')} to {test_df['date'].max().strftime('%Y-%m-%d')}")
    
    # Initialize predictors
    predictors = {
        'RBM Predictor': RBMPredictor(train_df),
        'DL Predictor': DLPredictor(train_df),
        'Ensemble Predictor': EnsemblePredictor(train_df)
    }
    
    # Next Draw Predictions
    st.header("Next Draw Predictions")
    cols = st.columns(3)
    
    for idx, (name, predictor) in enumerate(predictors.items()):
        with cols[idx]:
            st.subheader(name)
            predictions = predictor.predict(3)
            for i, (numbers, joker) in enumerate(predictions, 1):
                with st.container():
                    st.write(f"Prediction {i}:")
                    st.write(f"Numbers: {[int(x) for x in numbers]}")
                    st.write(f"Joker: {int(joker)}")
                    st.write("---")
    
    # Evaluate predictors
    evaluations = {
        name: evaluate_predictions(predictor, test_df, name)
        for name, predictor in predictors.items()
    }
    
    # Create performance plots
    st.header("Performance Analysis")
    
    # Calculate moving averages
    window = 10  # 10-draw moving average
    
    # Numbers matching plot
    fig1 = go.Figure()
    for name, stats in evaluations.items():
        # Raw data
        fig1.add_trace(go.Scatter(
            x=test_df['date'],
            y=stats['correct_numbers'],
            name=f"{name} (Raw)",
            line=dict(dash='dot'),
            opacity=0.3
        ))
        # Moving average
        ma = pd.Series(stats['correct_numbers']).rolling(window=window).mean()
        fig1.add_trace(go.Scatter(
            x=test_df['date'],
            y=ma,
            name=f"{name} ({window}-draw MA)",
            line=dict(width=2)
        ))
    fig1.update_layout(
        title='Number Matching Performance',
        xaxis_title='Draw Date',
        yaxis_title='Correct Numbers',
        hovermode='x unified'
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Joker matching plot
    fig2 = go.Figure()
    for name, stats in evaluations.items():
        # Raw data
        fig2.add_trace(go.Scatter(
            x=test_df['date'],
            y=stats['correct_jokers'],
            name=f"{name} (Raw)",
            line=dict(dash='dot'),
            opacity=0.3
        ))
        # Moving average
        ma = pd.Series(stats['correct_jokers']).rolling(window=window).mean()
        fig2.add_trace(go.Scatter(
            x=test_df['date'],
            y=ma,
            name=f"{name} ({window}-draw MA)",
            line=dict(width=2)
        ))
    fig2.update_layout(
        title='Joker Matching Performance',
        xaxis_title='Draw Date',
        yaxis_title='Correct Joker',
        hovermode='x unified'
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Statistics
    st.header("Predictor Statistics")
    stat_cols = st.columns(3)
    
    for idx, (name, stats) in enumerate(evaluations.items()):
        with stat_cols[idx]:
            st.subheader(name)
            st.metric("Average Numbers Matched", f"{stats['avg_numbers']:.2f}")
            st.metric("Average Joker Hit Rate", f"{stats['avg_joker']:.2f}")
            
            st.write("Recent Results:")
            for result in stats['results'][-3:]:
                with st.expander(f"Draw {result['draw_date'].strftime('%Y-%m-%d')}"):
                    st.write(f"Matched Numbers: {result['best_match']}")
                    st.write(f"Hit Joker: {'Yes' if result['hit_joker'] else 'No'}")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.exception(e)
