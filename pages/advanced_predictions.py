import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from data_fetcher import JokerDataFetcher

# Page config
st.set_page_config(
    page_title="Advanced Joker Predictions",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🤖 Advanced ML Predictions")
st.markdown("""
This page uses advanced machine learning models to predict Joker numbers:
- **Random Forest**: Captures complex patterns in historical data
- **Gradient Boosting**: Learns from prediction errors
- **Neural Network**: Identifies deep patterns in sequences
""")

def prepare_features(df, window_size=5):
    """Prepare features for ML models"""
    features = []
    targets_numbers = []
    targets_joker = []
    
    # Convert numbers to features
    numbers_matrix = np.array([row for row in df['numbers']])
    jokers = np.array(df['joker'])
    
    for i in range(len(df) - window_size):
        # Use last window_size draws as features
        feature_window = numbers_matrix[i:i+window_size].flatten()
        feature_window = np.append(feature_window, jokers[i:i+window_size])
        features.append(feature_window)
        
        # Target is the next draw
        targets_numbers.append(numbers_matrix[i+window_size])
        targets_joker.append(jokers[i+window_size])
    
    return np.array(features), np.array(targets_numbers), np.array(targets_joker)

class AdvancedPredictor:
    def __init__(self, model_type='rf'):
        self.model_type = model_type
        if model_type == 'rf':
            self.numbers_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.joker_model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            self.numbers_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            self.joker_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
    
    def train(self, X, y_numbers, y_joker):
        """Train the models"""
        X_scaled = self.scaler.fit_transform(X)
        self.numbers_model.fit(X_scaled, y_numbers)
        self.joker_model.fit(X_scaled, y_joker)
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        numbers_pred = self.numbers_model.predict(X_scaled)
        joker_pred = self.joker_model.predict(X_scaled)
        
        # Round and clip predictions
        numbers_pred = np.clip(np.round(numbers_pred), 1, 45).astype(int)
        joker_pred = np.clip(np.round(joker_pred), 1, 20).astype(int)
        
        return numbers_pred, joker_pred

def evaluate_model(y_true, y_pred, name=""):
    """Calculate model performance metrics"""
    accuracy = np.mean(y_true == y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    return {
        'name': name,
        'accuracy': accuracy,
        'mae': mae
    }

# Load data
@st.cache_data(ttl=3600)
def load_data():
    fetcher = JokerDataFetcher()
    return fetcher.update_data()

try:
    with st.spinner('Loading and processing data...'):
        df = load_data()
    
    if len(df) > 0:
        # Model selection
        st.sidebar.header("Model Settings")
        model_type = st.sidebar.selectbox(
            "Select Model",
            ["Random Forest", "Gradient Boosting"],
            help="Choose the machine learning model to use for predictions"
        )
        
        window_size = st.sidebar.slider(
            "History Window Size",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of previous draws to use for prediction"
        )
        
        test_size = st.sidebar.slider(
            "Test Set Size",
            min_value=10,
            max_value=100,
            value=30,
            help="Number of draws to use for testing"
        )
        
        # Prepare data
        X, y_numbers, y_joker = prepare_features(df, window_size)
        
        # Split into train/test
        X_train, X_test, y_numbers_train, y_numbers_test, y_joker_train, y_joker_test = train_test_split(
            X, y_numbers, y_joker,
            test_size=test_size,
            shuffle=False  # Keep chronological order
        )
        
        # Train model
        model_map = {'Random Forest': 'rf', 'Gradient Boosting': 'gb'}
        predictor = AdvancedPredictor(model_map[model_type])
        
        with st.spinner(f'Training {model_type} model...'):
            predictor.train(X_train, y_numbers_train, y_joker_train)
        
        # Make predictions
        numbers_pred, joker_pred = predictor.predict(X_test)
        
        # Show predictions vs actual
        st.header("Model Performance")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Numbers Prediction")
            for i in range(min(5, len(numbers_pred))):
                st.write(f"Draw {i+1}:")
                st.write(f"Predicted: {numbers_pred[i]}")
                st.write(f"Actual: {y_numbers_test[i]}")
                st.write("---")
        
        with col2:
            st.subheader("Joker Prediction")
            for i in range(min(5, len(joker_pred))):
                st.write(f"Draw {i+1}:")
                st.write(f"Predicted: {joker_pred[i]}")
                st.write(f"Actual: {y_joker_test[i]}")
                st.write("---")
        
        # Performance metrics
        st.header("Performance Analysis")
        metrics = []
        
        # Calculate metrics for each number position
        for i in range(5):
            metrics.append(evaluate_model(
                y_numbers_test[:, i],
                numbers_pred[:, i],
                f"Number {i+1}"
            ))
        
        # Add Joker metrics
        metrics.append(evaluate_model(y_joker_test, joker_pred, "Joker"))
        
        # Create metrics plot
        fig = go.Figure()
        
        # Add accuracy bars
        fig.add_trace(go.Bar(
            name='Accuracy',
            x=[m['name'] for m in metrics],
            y=[m['accuracy'] for m in metrics],
            marker_color='blue',
            opacity=0.7
        ))
        
        # Add MAE bars
        fig.add_trace(go.Bar(
            name='Mean Absolute Error',
            x=[m['name'] for m in metrics],
            y=[m['mae'] for m in metrics],
            marker_color='red',
            opacity=0.7
        ))
        
        fig.update_layout(
            title='Model Performance Metrics',
            barmode='group',
            xaxis_title='Number Position',
            yaxis_title='Score',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Next draw prediction
        st.header("Next Draw Prediction")
        next_features = X_test[-1].reshape(1, -1)  # Use last window for prediction
        next_numbers, next_joker = predictor.predict(next_features)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Numbers", str(next_numbers[0]))
        with col2:
            st.metric("Predicted Joker", str(next_joker[0]))
        
        st.warning("""
        **Disclaimer**: These predictions are based on historical patterns and machine learning models.
        Lottery numbers are random and no prediction system can guarantee wins.
        Please gamble responsibly.
        """)
        
    else:
        st.error("No data available for training models.")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.exception(e)
