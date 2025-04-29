import numpy as np

class SimplePredictor:
    def __init__(self, df):
        self.df = df
        self.n_numbers = 45
        self.n_joker = 20
    
    def predict(self, n_predictions=1):
        results = []
        for _ in range(n_predictions):
            # Get random numbers
            numbers = sorted(np.random.choice(range(1, self.n_numbers + 1), size=5, replace=False))
            joker = np.random.randint(1, self.n_joker + 1)
            results.append((numbers, joker))
        return results

class RBMPredictor(SimplePredictor):
    def predict(self, n_predictions=1):
        results = []
        for _ in range(n_predictions):
            # Use last draw as base and modify slightly
            last_draw = self.df.iloc[-1]
            base_numbers = last_draw['numbers']
            
            # Modify 1-2 numbers
            numbers = list(base_numbers)
            n_changes = np.random.randint(1, 3)
            for _ in range(n_changes):
                idx = np.random.randint(0, 5)
                numbers[idx] = np.random.randint(1, self.n_numbers + 1)
            numbers = sorted(list(set(numbers)))
            
            # Fill if needed
            while len(numbers) < 5:
                new_num = np.random.randint(1, self.n_numbers + 1)
                if new_num not in numbers:
                    numbers.append(new_num)
            numbers.sort()
            
            # Get joker
            joker = int(np.random.randint(1, self.n_joker + 1))
            results.append((numbers, joker))
        return results

class DLPredictor(SimplePredictor):
    def predict(self, n_predictions=1):
        results = []
        for _ in range(n_predictions):
            # Use patterns from last few draws
            recent_draws = self.df.iloc[-3:]
            pattern_base = []
            for _, row in recent_draws.iterrows():
                pattern_base.extend(row['numbers'])
            
            # Select numbers with some randomness
            probs = np.ones(self.n_numbers) * 0.1
            for num in pattern_base:
                probs[num-1] += 0.3
            probs = probs / probs.sum()
            
            numbers = []
            while len(numbers) < 5:
                num = int(np.random.choice(range(1, self.n_numbers + 1), p=probs))
                if num not in numbers:
                    numbers.append(num)
            numbers.sort()
            
            # Get joker
            joker = int(np.random.randint(1, self.n_joker + 1))
            results.append((numbers, joker))
        return results

class EnsemblePredictor(SimplePredictor):
    def __init__(self, df):
        super().__init__(df)
        self.rbm = RBMPredictor(df)
        self.dl = DLPredictor(df)
    
    def predict(self, n_predictions=1):
        results = []
        for _ in range(n_predictions):
            # Get predictions from base models
            rbm_pred = self.rbm.predict(1)[0]
            dl_pred = self.dl.predict(1)[0]
            
            # Combine predictions
            all_numbers = list(set(rbm_pred[0] + dl_pred[0]))
            numbers = sorted([int(x) for x in np.random.choice(all_numbers, size=min(5, len(all_numbers)), replace=False)])
            
            # Fill if needed
            while len(numbers) < 5:
                new_num = int(np.random.randint(1, self.n_numbers + 1))
                if new_num not in numbers:
                    numbers.append(new_num)
            numbers.sort()
            
            # Average jokers
            joker = int((rbm_pred[1] + dl_pred[1]) / 2)
            results.append((numbers, joker))
        return results
