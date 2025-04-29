import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import traceback
import time

class JokerDataFetcher:
    def __init__(self, data_file='joker_results.csv'):
        self.data_file = data_file
        self.base_url = "https://api.opap.gr/games/v1.0/5104"
        self.draw_url = "https://api.opap.gr/draws/v3.0/5104"
        self.page_size = 50
    
    def _get_last_draw_id(self):
        """Get the last draw ID from OPAP API"""
        try:
            url = f"{self.base_url}/last-result-and-active"
            print(f"Fetching last draw from: {url}")
            response = requests.get(url, timeout=10)
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'last' in data and 'drawId' in data['last']:
                    return data['last']['drawId']
            print(f"Error response: {response.text}")
            return None
        except Exception as e:
            print(f"Error getting last draw: {str(e)}")
            return None
    
    def _get_draw(self, draw_id):
        """Get a specific draw by ID"""
        try:
            url = f"{self.draw_url}/{draw_id}"
            print(f"Fetching draw {draw_id}")
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"Error getting draw {draw_id}: {str(e)}")
            return None
    
    def _parse_draw(self, draw_data):
        """Parse draw data from OPAP API"""
        try:
            if not draw_data or 'winningNumbers' not in draw_data:
                return None
                
            winning_numbers = draw_data['winningNumbers']
            if 'list' not in winning_numbers or 'bonus' not in winning_numbers:
                return None
                
            numbers = sorted([int(num) for num in winning_numbers['list']])
            bonus = winning_numbers['bonus'][0] if winning_numbers['bonus'] else None
            
            if len(numbers) != 5 or not bonus:
                return None
            
            return {
                'date': datetime.fromtimestamp(draw_data['drawTime'] / 1000).strftime('%Y-%m-%d'),
                'draw_id': draw_data['drawId'],
                'numbers': numbers,
                'joker': int(bonus)
            }
        except Exception as e:
            print(f"Error parsing draw: {str(e)}")
            return None
    
    def load_existing_data(self):
        """Load existing data from CSV file"""
        try:
            if os.path.exists(self.data_file):
                print(f"Loading data from {self.data_file}")
                df = pd.read_csv(self.data_file)
                # Convert string representation of list to actual list
                df['numbers'] = df['numbers'].apply(lambda x: [int(n) for n in eval(str(x))])
                df['date'] = pd.to_datetime(df['date'])
                df['joker'] = df['joker'].astype(int)
                print(f"Loaded {len(df)} draws")
                return df
            print(f"No existing data file found at {self.data_file}")
            return pd.DataFrame(columns=['date', 'draw_id', 'numbers', 'joker'])
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            traceback.print_exc()
            return pd.DataFrame(columns=['date', 'draw_id', 'numbers', 'joker'])
    
    def save_data(self, df):
        """Save data to CSV file"""
        try:
            # Convert lists to string representation for saving
            df_to_save = df.copy()
            df_to_save['numbers'] = df_to_save['numbers'].apply(str)
            df_to_save.to_csv(self.data_file, index=False)
            print(f"Saved {len(df)} draws to {self.data_file}")
        except Exception as e:
            print(f"Error saving data: {str(e)}")
            traceback.print_exc()
    
    def fetch_historical_data(self, start_year=2000):
        """Fetch all historical data by iterating through draw IDs"""
        print(f"Fetching historical data...")
        
        # Load existing data first
        df = self.load_existing_data()
        
        # Get the last draw ID
        last_draw_id = self._get_last_draw_id()
        if not last_draw_id:
            print("Could not get last draw ID!")
            return df
        
        print(f"Last draw ID: {last_draw_id}")
        
        # Start from the earliest possible draw ID
        # OPAP Joker started in November 1997 with draw ID around 1
        all_draws = []
        current_id = 1
        
        while current_id <= last_draw_id:
            draw_data = self._get_draw(current_id)
            if draw_data:
                parsed = self._parse_draw(draw_data)
                if parsed:
                    draw_date = datetime.strptime(parsed['date'], '%Y-%m-%d')
                    if draw_date.year >= start_year:
                        all_draws.append(parsed)
                        if len(all_draws) % 10 == 0:  # Show progress every 10 draws
                            print(f"Found {len(all_draws)} valid draws so far...")
            
            current_id += 1
            if current_id % 10 == 0:  # Add delay every 10 requests
                time.sleep(1)
        
        if all_draws:
            print(f"\nProcessing {len(all_draws)} total draws...")
            new_df = pd.DataFrame(all_draws)
            new_df['date'] = pd.to_datetime(new_df['date'])
            
            # Combine with existing data, removing duplicates
            if len(df) > 0:
                new_df = new_df[~new_df['draw_id'].isin(df['draw_id'])]
                if len(new_df) > 0:
                    df = pd.concat([df, new_df], ignore_index=True)
            else:
                df = new_df
            
            # Sort by date and save
            df = df.sort_values('date')
            self.save_data(df)
            print(f"\nFinal dataset: {len(df)} draws from {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        else:
            print("No draws found!")
        
        return df
    
    def update_data(self):
        """Update data file with new draws"""
        print("Loading existing data...")
        df = self.load_existing_data()
        
        if len(df) == 0:
            print("No existing data, fetching full historical data...")
            return self.fetch_historical_data()
        
        # Get latest draw ID from our data
        latest_draw_id = df['draw_id'].max()
        
        # Get current last draw ID
        last_draw_id = self._get_last_draw_id()
        if not last_draw_id:
            print("Could not get last draw ID!")
            return df
        
        print(f"Fetching draws from ID {latest_draw_id + 1} to {last_draw_id}")
        
        new_draws = []
        current_id = latest_draw_id + 1
        
        while current_id <= last_draw_id:
            draw_data = self._get_draw(current_id)
            if draw_data:
                parsed = self._parse_draw(draw_data)
                if parsed:
                    new_draws.append(parsed)
            current_id += 1
            if current_id % 10 == 0:  # Add delay every 10 requests
                time.sleep(1)
        
        if new_draws:
            print(f"Processing {len(new_draws)} new draws")
            new_df = pd.DataFrame(new_draws)
            new_df['date'] = pd.to_datetime(new_df['date'])
            
            # Add new draws
            df = pd.concat([df, new_df], ignore_index=True)
            df = df.sort_values('date')
            self.save_data(df)
            print(f"Added {len(new_draws)} new draws")
        else:
            print("No new draws found")
        
        print(f"Total draws: {len(df)}")
        print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        
        return df

if __name__ == "__main__":
    # Test the fetcher
    fetcher = JokerDataFetcher()
    df = fetcher.fetch_historical_data(start_year=2000)
    print(f"Total draws: {len(df)}")
    if len(df) > 0:
        print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        print("Sample row:")
        print(df.iloc[0])
