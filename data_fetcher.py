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
        self.base_url = "https://api.opap.gr/draws/v3.0/5104"
        self.page_size = 50
    
    def _get_draws_for_date_range(self, start_date, end_date):
        """Get all draws between two dates using pagination"""
        all_draws = []
        page = 1
        
        while True:
            try:
                url = (f"{self.base_url}/draw-date/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
                      f"?page={page}&pageSize={self.page_size}")
                
                print(f"Fetching page {page} from: {url}")
                response = requests.get(url, timeout=10)
                print(f"Response status: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    if not data:  # No more draws
                        break
                    
                    all_draws.extend(data)
                    print(f"Found {len(data)} draws on page {page}")
                    
                    if len(data) < self.page_size:  # Last page
                        break
                    
                    page += 1
                    time.sleep(0.5)  # Small delay between requests
                else:
                    print(f"Error response: {response.text}")
                    break
                    
            except Exception as e:
                print(f"Error getting draws: {str(e)}")
                traceback.print_exc()
                break
        
        print(f"Total draws found: {len(all_draws)}")
        return all_draws
    
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
        """Fetch all historical data from start_year to now"""
        print(f"Fetching historical data from {start_year} to present...")
        
        # Load existing data first
        df = self.load_existing_data()
        
        # Calculate date ranges
        start_date = datetime(start_year, 1, 1)
        current_date = datetime.now()
        
        # Fetch data in 30-day chunks to avoid timeouts
        chunk_delta = timedelta(days=30)
        chunk_start = start_date
        
        all_draws = []
        
        while chunk_start < current_date:
            chunk_end = min(chunk_start + chunk_delta, current_date)
            print(f"\nFetching chunk: {chunk_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")
            
            # Try to get draws for this chunk
            draws = self._get_draws_for_date_range(chunk_start, chunk_end)
            if draws:
                for draw in draws:
                    parsed = self._parse_draw(draw)
                    if parsed:
                        all_draws.append(parsed)
                print(f"Successfully parsed {len(draws)} draws in this chunk")
            
            # Add a small delay between chunks
            time.sleep(1)
            chunk_start = chunk_end
        
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
        
        # Get latest date in our data
        latest_date = df['date'].max()
        current_date = datetime.now()
        
        # If our data is more than a month old, do a full update
        if latest_date < current_date - timedelta(days=30):
            print("Data is more than a month old, fetching full historical data...")
            return self.fetch_historical_data()
        
        # Otherwise just get recent draws
        print(f"Fetching draws from {latest_date.strftime('%Y-%m-%d')} to {current_date.strftime('%Y-%m-%d')}...")
        draws = self._get_draws_for_date_range(latest_date, current_date)
        
        new_draws = []
        for draw in draws:
            parsed = self._parse_draw(draw)
            if parsed:
                new_draws.append(parsed)
        
        if new_draws:
            print(f"Processing {len(new_draws)} new draws")
            new_df = pd.DataFrame(new_draws)
            new_df['date'] = pd.to_datetime(new_df['date'])
            
            # Remove duplicates based on draw_id
            new_df = new_df[~new_df['draw_id'].isin(df['draw_id'])]
            
            if len(new_df) > 0:
                print(f"Adding {len(new_df)} new draws")
                df = pd.concat([df, new_df], ignore_index=True)
                df = df.sort_values('date')
                self.save_data(df)
            else:
                print("No new unique draws found")
        else:
            print("No new draws found")
        
        if len(df) == 0:
            print("WARNING: No data available!")
        else:
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
