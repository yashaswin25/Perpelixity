import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Create sample data for classification (Customer Churn Prediction)
def create_classification_data(n_samples=1000):
    # Generate customer data
    customer_ids = range(1, n_samples + 1)
    tenure = np.random.randint(1, 72, n_samples)  # months
    monthly_charges = np.random.normal(65, 30, n_samples).round(2)
    total_charges = (tenure * monthly_charges).round(2)
    
    # Generate categorical features
    contract_types = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.3, 0.2])
    payment_methods = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples)
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.3, 0.4, 0.3])
    
    # Generate additional features
    online_security = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2])
    online_backup = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2])
    tech_support = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2])
    
    # Generate target variable (Churn) based on features
    churn_prob = (
        0.1 +  # Base probability
        0.3 * (contract_types == 'Month-to-month') +  # Higher churn for month-to-month
        0.2 * (internet_service == 'Fiber optic') +  # Higher churn for fiber optic
        0.1 * (monthly_charges > 70) -  # Higher churn for expensive plans
        0.2 * (online_security == 'Yes') -  # Lower churn with security
        0.2 * (tech_support == 'Yes')  # Lower churn with tech support
    )
    churn = np.random.binomial(1, churn_prob)
    
    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'tenure': tenure,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'contract_type': contract_types,
        'payment_method': payment_methods,
        'internet_service': internet_service,
        'online_security': online_security,
        'online_backup': online_backup,
        'tech_support': tech_support,
        'churn': churn
    })
    
    return df

# Create sample data for regression (House Price Prediction)
def create_regression_data(n_samples=1000):
    # Generate house features
    house_ids = range(1, n_samples + 1)
    area = np.random.normal(1500, 500, n_samples).round(0)  # square feet
    bedrooms = np.random.randint(1, 6, n_samples)
    bathrooms = np.random.randint(1, 4, n_samples)
    year_built = np.random.randint(1950, 2023, n_samples)
    
    # Generate categorical features
    location = np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples, p=[0.4, 0.4, 0.2])
    condition = np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], n_samples, p=[0.2, 0.4, 0.3, 0.1])
    
    # Generate target variable (Price) based on features
    base_price = 200000
    price = (
        base_price +
        area * 100 +  # $100 per square foot
        bedrooms * 50000 +  # $50k per bedroom
        bathrooms * 30000 +  # $30k per bathroom
        (2023 - year_built) * -1000 +  # $1k less per year old
        (location == 'Urban') * 50000 +  # Urban premium
        (location == 'Rural') * -30000 +  # Rural discount
        (condition == 'Excellent') * 50000 +  # Condition premium
        (condition == 'Poor') * -40000  # Condition discount
    )
    price = price + np.random.normal(0, 50000, n_samples)  # Add some noise
    price = price.round(-3)  # Round to nearest thousand
    
    # Create DataFrame
    df = pd.DataFrame({
        'house_id': house_ids,
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'year_built': year_built,
        'location': location,
        'condition': condition,
        'price': price
    })
    
    return df

# Create and save the sample data
def main():
    # Create classification data
    classification_df = create_classification_data()
    
    # Create regression data
    regression_df = create_regression_data()
    
    # Save to Excel file with two sheets
    with pd.ExcelWriter('sample_data.xlsx') as writer:
        classification_df.to_excel(writer, sheet_name='Classification', index=False)
        regression_df.to_excel(writer, sheet_name='Regression', index=False)
    
    print("Sample data has been created and saved to 'sample_data.xlsx'")
    print("\nClassification Data Preview:")
    print(classification_df.head())
    print("\nRegression Data Preview:")
    print(regression_df.head())

if __name__ == "__main__":
    main() 