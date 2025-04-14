import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Create directory for sample data
os.makedirs('sample_data', exist_ok=True)

# -------------------- 1. Customer Churn Dataset -------------------- #
def create_churn_data(num_samples=100):
    np.random.seed(42)  # For reproducibility
    
    # Generate data
    data = {
        'customer_id': [f'CUST-{i:04d}' for i in range(1, num_samples+1)],
        'gender': np.random.choice(['Male', 'Female'], size=num_samples),
        'age': np.random.randint(18, 85, size=num_samples),
        'tenure': np.random.randint(1, 72, size=num_samples),  # Months
        'monthly_charges': np.round(np.random.uniform(20, 120, size=num_samples), 2),
        'total_charges': np.zeros(num_samples),
        'phone_service': np.random.choice(['Yes', 'No'], size=num_samples, p=[0.9, 0.1]),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], size=num_samples),
        'contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], size=num_samples),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], size=num_samples),
        'churn': np.zeros(num_samples, dtype=int),
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Calculate total charges based on tenure and monthly charges (with some noise)
    df['total_charges'] = df['tenure'] * df['monthly_charges'] * (0.9 + 0.2 * np.random.random(num_samples))
    df['total_charges'] = df['total_charges'].round(2)
    
    # Determine churn based on a simple rule (just for sample data)
    # Higher chance of churn if: month-to-month contract, high monthly charges, low tenure
    df['churn_score'] = (
        (df['contract'] == 'Month-to-month') * 0.6 +
        (df['monthly_charges'] > 80) * 0.3 +
        (df['tenure'] < 12) * 0.4 -
        (df['tenure'] > 36) * 0.3
    ) + np.random.random(num_samples) * 0.3
    
    df['churn'] = (df['churn_score'] > 0.7).astype(int)
    df.drop('churn_score', axis=1, inplace=True)
    
    # Save to Excel
    file_path = 'sample_data/churn_data_sample.xlsx'
    df.to_excel(file_path, index=False)
    print(f"Customer churn dataset saved to {file_path}")
    
    return df

# -------------------- 2. Fraud Detection Dataset -------------------- #
def create_fraud_data(num_samples=200):
    np.random.seed(42)  # For reproducibility
    
    # Generate transaction timestamps over the last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    timestamps = [start_date + timedelta(seconds=random.randint(0, 30*24*60*60)) for _ in range(num_samples)]
    timestamps.sort()
    
    # Generate data
    transaction_types = ['purchase', 'withdrawal', 'transfer', 'payment']
    merchants = ['Online Store', 'Grocery', 'Restaurant', 'Electronics', 'Travel', 
                'Gas Station', 'Unknown', 'Entertainment', 'Utility', 'Healthcare']
    locations = ['New York', 'Los Angeles', 'Chicago', 'Online', 'International', 
                'Miami', 'Dallas', 'Seattle', 'Boston', 'San Francisco']
    
    data = {
        'transaction_id': [f'TXN-{i:06d}' for i in range(1, num_samples+1)],
        'time': timestamps,
        'amount': np.round(np.random.exponential(scale=50, size=num_samples) + 5, 2),
        'transaction_type': np.random.choice(transaction_types, size=num_samples),
        'merchant': np.random.choice(merchants, size=num_samples),
        'location': np.random.choice(locations, size=num_samples),
        'hour': [ts.hour for ts in timestamps],
        'day_of_week': [ts.strftime('%A') for ts in timestamps],
        'previous_transaction': np.zeros(num_samples),
        'is_fraud': np.zeros(num_samples, dtype=int),
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Set previous transaction amount (shift by 1, first will be 0)
    df['previous_transaction'] = df['amount'].shift(1).fillna(0)
    
    # Generate fraud cases (around 5% of transactions)
    fraud_indices = np.random.choice(range(num_samples), size=int(num_samples * 0.05), replace=False)
    
    # Fraud rules: Large amounts, unusual locations, or unusual hours
    for idx in fraud_indices:
        # Make some modifications to make it look like fraud
        if np.random.random() < 0.33:
            # Unusually large amount
            df.at[idx, 'amount'] = df.at[idx, 'amount'] * 5 + 200
        elif np.random.random() < 0.5:
            # Unusual location
            df.at[idx, 'location'] = 'International'
        else:
            # Unusual hour
            df.at[idx, 'hour'] = np.random.choice([1, 2, 3, 4])
        
        df.at[idx, 'is_fraud'] = 1
    
    # Convert time to string for Excel
    df['time'] = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Save to Excel
    file_path = 'sample_data/fraud_data_sample.xlsx'
    df.to_excel(file_path, index=False)
    print(f"Fraud detection dataset saved to {file_path}")
    
    return df

# -------------------- 3. Sales Forecasting Dataset -------------------- #
def create_sales_data(num_samples=365):
    np.random.seed(42)  # For reproducibility
    
    # Create date range for one year
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(num_samples)]
    
    # Product categories
    products = ['Electronics', 'Clothing', 'Food', 'Home Goods', 'Books']
    stores = ['Store A', 'Store B', 'Store C', 'Online']
    
    # Generate rows for each product and store combination
    rows = []
    for date in dates:
        for product in products:
            for store in stores:
                # Basic seasonal pattern
                season_factor = 1.0
                month = date.month
                
                # Higher sales in summer and winter holiday season
                if month in [6, 7, 8]:  # Summer
                    season_factor = 1.2
                elif month in [11, 12]:  # Holiday season
                    season_factor = 1.5
                
                # Weekend effect
                day_of_week = date.weekday()  # 0=Monday, 6=Sunday
                weekend_factor = 1.3 if day_of_week >= 5 else 1.0
                
                # Base price varies by product
                base_price = {
                    'Electronics': 100,
                    'Clothing': 50,
                    'Food': 20,
                    'Home Goods': 70,
                    'Books': 15
                }[product]
                
                # Apply some price variation
                price = round(base_price * (0.9 + 0.2 * np.random.random()), 2)
                
                # Promotion randomly for 15% of entries
                promotion = int(np.random.random() < 0.15)
                
                # Holiday indicator
                holidays = [
                    datetime(2023, 1, 1),   # New Year
                    datetime(2023, 7, 4),   # Independence Day
                    datetime(2023, 11, 24), # Black Friday
                    datetime(2023, 12, 25)  # Christmas
                ]
                holiday = int(date in holidays)
                
                # Calculate sales amount based on factors
                base_sales = base_price * (0.1 + 0.05 * np.random.random())
                sales_amount = base_sales * season_factor * weekend_factor
                
                # Promotion effect
                if promotion:
                    sales_amount *= 1.4
                
                # Holiday effect
                if holiday:
                    sales_amount *= 2.0
                
                # Add store-specific factors
                store_factors = {
                    'Store A': 1.2,
                    'Store B': 1.0,
                    'Store C': 0.8,
                    'Online': 1.5
                }
                sales_amount *= store_factors[store]
                
                # Add some random noise
                sales_amount *= (0.8 + 0.4 * np.random.random())
                
                # Calculate discount percentage
                discount = 0
                if promotion:
                    discount = int(np.random.choice([10, 15, 20, 25, 30]))
                
                # Add row to data
                rows.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'product': product,
                    'store': store,
                    'price': price,
                    'promotion': promotion,
                    'holiday': holiday,
                    'day_of_week': date.strftime('%A'),
                    'month': date.strftime('%B'),
                    'season': 'Winter' if month in [12, 1, 2] else
                             'Spring' if month in [3, 4, 5] else
                             'Summer' if month in [6, 7, 8] else 'Fall',
                    'discount': discount,
                    'sales_amount': round(sales_amount, 2)
                })
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Sample a subset to keep file size manageable
    if len(df) > 2000:
        df = df.sample(2000, random_state=42)
    
    # Save to Excel
    file_path = 'sample_data/sales_data_sample.xlsx'
    df.to_excel(file_path, index=False)
    print(f"Sales forecasting dataset saved to {file_path}")
    
    return df

# Generate all sample datasets
if __name__ == "__main__":
    print("Generating sample datasets for ML prediction system...")
    create_churn_data(num_samples=200)
    create_fraud_data(num_samples=300)
    create_sales_data()
    print("All sample datasets created successfully!") 