import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from predictors import RBMPredictor, DLPredictor, EnsemblePredictor
from datetime import datetime, timedelta

# Page config
st.set_page_config(page_title="Joker Prediction Analysis", layout="wide")

# Title
st.title("Joker Prediction Analysis")

def generate_sample_data(n_draws=500):
    """Generate sample data when real data is not available"""
    dates = pd.date_range(end=datetime.now(), periods=n_draws).tolist()
    data = []
    
    for date in dates:
        numbers = sorted(np.random.choice(range(1, 46), size=5, replace=False))
        joker = np.random.randint(1, 21)
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'numbers': numbers,
            'joker': joker
        })
    
    return pd.DataFrame(data)

@st.cache_data
def load_data():
    """Load and preprocess the Joker dataset"""
    try:
        # Generate sample data since we can't access the real data file
        df = generate_sample_data(500)
        print("Using generated sample data")
    except Exception as e:
        print(f"Error: {str(e)}")
        return None
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Convert joker to list format for compatibility
    df['joker'] = df['joker'].fillna(-1).astype(int).apply(lambda x: [x] if x != -1 else None)
    
    print(f"Loaded {len(df)} draws from {df['date'].min()} to {df['date'].max()}")
    return df

def evaluate_predictions(predictor, test_df, name):
    """Evaluate predictor performance on test data"""
    results = []
    correct_numbers = []
    correct_jokers = []
    
    for idx, row in test_df.iterrows():
        actual_numbers = set(row['numbers'])
        actual_joker = row['joker'][0] if row['joker'] else None
        
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
    df = load_data()
    
    if df is None:
        st.error("Failed to load data")
    else:
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
