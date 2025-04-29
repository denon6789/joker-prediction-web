import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

class JokerDataFetcher:
    def __init__(self, data_file='joker_results.csv'):
        self.data_file = data_file
        self.base_url = "https://api.opap.gr/draws/v3.0/5104"
        
    def _get_draw_date(self, draw_id):
        """Get draw date from OPAP API"""
        url = f"{self.base_url}/{draw_id}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                return datetime.fromtimestamp(data['drawTime'] / 1000)
            return None
        except:
            return None
    
    def _get_draws_for_date_range(self, start_date, end_date):
        """Get all draws between two dates"""
        url = f"{self.base_url}/draw-date/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
            return []
        except:
            return []
    
    def _parse_draw(self, draw_data):
        """Parse draw data from OPAP API"""
        try:
            winning_numbers = draw_data['winningNumbers']
            numbers = sorted([int(num) for num in winning_numbers['list']])
            bonus = winning_numbers.get('bonus', [None])[0]
            
            return {
                'date': datetime.fromtimestamp(draw_data['drawTime'] / 1000).strftime('%Y-%m-%d'),
                'draw_id': draw_data['drawId'],
                'numbers': numbers,
                'joker': int(bonus) if bonus else None
            }
        except:
            return None
    
    def load_existing_data(self):
        """Load existing data from CSV file"""
        if os.path.exists(self.data_file):
            df = pd.read_csv(self.data_file)
            # Convert string representation of list to actual list
            df['numbers'] = df['numbers'].apply(lambda x: [int(n) for n in eval(str(x))])
            df['date'] = pd.to_datetime(df['date'])
            df['joker'] = df['joker'].astype(int)
            return df
        return pd.DataFrame(columns=['date', 'draw_id', 'numbers', 'joker'])
    
    def save_data(self, df):
        """Save data to CSV file"""
        # Convert lists to string representation for saving
        df_to_save = df.copy()
        df_to_save['numbers'] = df_to_save['numbers'].apply(str)
        df_to_save.to_csv(self.data_file, index=False)
        print(f"Saved {len(df)} draws to {self.data_file}")
    
    def update_data(self):
        """Update data file with new draws"""
        print("Loading existing data...")
        df = self.load_existing_data()
        
        # Get latest date in our data
        if len(df) > 0:
            latest_date = df['date'].max()
        else:
            latest_date = datetime(2000, 1, 1)
        
        # Get current date
        current_date = datetime.now()
        
        print(f"Fetching draws from {latest_date.strftime('%Y-%m-%d')} to {current_date.strftime('%Y-%m-%d')}...")
        draws = self._get_draws_for_date_range(latest_date, current_date)
        
        new_draws = []
        for draw in draws:
            parsed = self._parse_draw(draw)
            if parsed:
                new_draws.append(parsed)
        
        if new_draws:
            new_df = pd.DataFrame(new_draws)
            new_df['date'] = pd.to_datetime(new_df['date'])
            
            # Remove duplicates based on draw_id
            if len(df) > 0:
                new_df = new_df[~new_df['draw_id'].isin(df['draw_id'])]
            
            if len(new_df) > 0:
                print(f"Found {len(new_df)} new draws")
                df = pd.concat([df, new_df], ignore_index=True)
                df = df.sort_values('date')
                self.save_data(df)
            else:
                print("No new draws found")
        else:
            print("No new draws found")
        
        return df

if __name__ == "__main__":
    # Test the fetcher
    fetcher = JokerDataFetcher()
    df = fetcher.update_data()
    print(f"Total draws: {len(df)}")
    if len(df) > 0:
        print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        print("Sample row:")
        print(df.iloc[0])
