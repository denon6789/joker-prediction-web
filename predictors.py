import numpy as np
import pandas as pd

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
            
            predictions.append((numbers, int(joker)))
        
        return predictions

class DLPredictor(BasePredictor):
    def predict(self, n_predictions=1):
        """Generate predictions using recent patterns"""
        predictions = []
        recent_draws = self.df.iloc[-5:]  # Use last 5 draws
        
        for _ in range(n_predictions):
            # Calculate probabilities based on recent numbers
            number_weights = np.ones(45)  # Initialize with base probability
            joker_weights = np.ones(20)   # Initialize with base probability
            
            # Increase weights for recent numbers
            for _, row in recent_draws.iterrows():
                for num in row['numbers']:
                    number_weights[num-1] += 1
                joker_weights[row['joker']-1] += 1
            
            # Normalize weights
            number_weights = number_weights / number_weights.sum()
            joker_weights = joker_weights / joker_weights.sum()
            
            # Generate prediction
            numbers = []
            while len(numbers) < 5:
                num = np.random.choice(range(1, 46), p=number_weights)
                if num not in numbers:
                    numbers.append(num)
            numbers.sort()
            
            joker = np.random.choice(range(1, 21), p=joker_weights)
            
            predictions.append((numbers, int(joker)))
        
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
        
        # Select random predictions from the pool
        selected = []
        for _ in range(n_predictions):
            if not all_predictions:
                break
            idx = np.random.randint(len(all_predictions))
            selected.append(all_predictions.pop(idx))
        
        return selected
