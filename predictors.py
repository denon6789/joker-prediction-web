import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

class BasePredictor:
    def __init__(self, df):
        self.df = df
        self.n_numbers = 5  # Joker has 5 main numbers
        self.n_joker = 20   # Joker number range is 1-20
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data for training"""
        # Convert numbers lists to a matrix
        self.numbers_matrix = np.array([row for row in self.df['numbers']])
        self.jokers = np.array(self.df['joker'])
    
    def predict(self, n_predictions=1):
        """Generate n predictions"""
        raise NotImplementedError("Subclasses must implement predict()")

class RBMPredictor(BasePredictor):
    def __init__(self, df):
        super().__init__(df)
        self._train_model()
    
    def _train_model(self):
        """Train a simple model based on frequency analysis"""
        self.number_freq = {}
        self.joker_freq = {}
        
        # Calculate frequencies
        for numbers in self.numbers_matrix:
            for num in numbers:
                self.number_freq[num] = self.number_freq.get(num, 0) + 1
        
        for joker in self.jokers:
            self.joker_freq[joker] = self.joker_freq.get(joker, 0) + 1
    
    def predict(self, n_predictions=1):
        """Generate predictions based on frequency analysis"""
        predictions = []
        for _ in range(n_predictions):
            # Select numbers with higher weights for more frequent numbers
            weights = [self.number_freq.get(i, 0) + 1 for i in range(1, 46)]
            weights = np.array(weights) / sum(weights)
            numbers = np.random.choice(range(1, 46), size=5, replace=False, p=weights)
            numbers = sorted(numbers.tolist())
            
            # Select joker with higher weights for more frequent numbers
            joker_weights = [self.joker_freq.get(i, 0) + 1 for i in range(1, 21)]
            joker_weights = np.array(joker_weights) / sum(joker_weights)
            joker = np.random.choice(range(1, 21), p=joker_weights)
            
            predictions.append((numbers, joker))
        
        return predictions

class DLPredictor(BasePredictor):
    def __init__(self, df):
        super().__init__(df)
        self._train_model()
    
    def _prepare_sequences(self, window_size=10):
        """Prepare sequences for training"""
        X, y = [], []
        for i in range(len(self.numbers_matrix) - window_size):
            X.append(self.numbers_matrix[i:i+window_size].flatten())
            y.append(np.concatenate([self.numbers_matrix[i+window_size], [self.jokers[i+window_size]]]))
        return np.array(X), np.array(y)
    
    def _train_model(self, window_size=10):
        """Train a neural network model"""
        X, y = self._prepare_sequences(window_size)
        if len(X) == 0:
            # Not enough data, use simpler model
            self.model = None
            return
            
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            max_iter=1000,
            random_state=42
        )
        self.model.fit(X_scaled, y)
    
    def predict(self, n_predictions=1):
        """Generate predictions using the neural network"""
        if self.model is None:
            # Fall back to simpler prediction
            return RBMPredictor(self.df).predict(n_predictions)
            
        predictions = []
        last_sequence = self.numbers_matrix[-10:].flatten()
        
        for _ in range(n_predictions):
            # Get model prediction
            X = self.scaler.transform([last_sequence])
            pred = self.model.predict(X)[0]
            
            # Round to integers and ensure valid ranges
            numbers = np.clip(np.round(pred[:5]), 1, 45).astype(int)
            numbers = sorted(np.unique(numbers))
            # If we don't have enough numbers, add some randomly
            while len(numbers) < 5:
                new_num = np.random.randint(1, 46)
                if new_num not in numbers:
                    numbers.append(new_num)
            numbers = sorted(numbers[:5])
            
            joker = int(np.clip(np.round(pred[5]), 1, 20))
            
            predictions.append((numbers, joker))
        
        return predictions

class EnsemblePredictor(BasePredictor):
    def __init__(self, df):
        super().__init__(df)
        self.predictors = [
            RBMPredictor(df),
            DLPredictor(df)
        ]
    
    def predict(self, n_predictions=1):
        """Generate predictions by combining multiple predictors"""
        all_predictions = []
        for predictor in self.predictors:
            all_predictions.extend(predictor.predict(n_predictions))
        
        # Select the most diverse predictions
        selected = []
        for _ in range(n_predictions):
            if not all_predictions:
                break
            # Select a random prediction
            idx = np.random.randint(len(all_predictions))
            selected.append(all_predictions.pop(idx))
        
        return selected
