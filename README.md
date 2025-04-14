# ML Prediction System

A Streamlit-based machine learning application that allows users to perform classification and regression predictions on their data.

## Features

- **Simple Interface**: Easy-to-use web interface for data upload and model training
- **Two Prediction Types**:
  - Classification: For predicting categorical outcomes
  - Regression: For predicting continuous values
- **Binary Target Creation**: Convert continuous data into binary targets for classification
- **Feature Selection**: Flexible feature selection for model training
- **Visualizations**: 
  - Classification: Prediction distribution and feature importance
  - Regression: Value distribution and feature importance
- **Results Export**: Download predictions in Excel format

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
streamlit run ml_prediction_system.py
```

2. Access the application at `http://localhost:8502`

3. Using the Application:

### For Classification:

1. Select "Classification" in the sidebar
2. Upload your Excel file
3. If you have continuous data:
   - Select a numeric column to convert to binary
   - Set a threshold value
   - The application will create a new binary column
4. Select features for prediction
5. Click "Process and Predict"
6. View results and download predictions

### For Regression:

1. Select "Regression" in the sidebar
2. Upload your Excel file
3. Select a numeric target column
4. Select features for prediction
5. Click "Process and Predict"
6. View results and download predictions

## Data Requirements

- **File Format**: Excel (.xlsx or .xls)
- **Classification**:
  - At least one numeric column for binary target creation
  - Multiple feature columns
- **Regression**:
  - One numeric target column
  - Multiple feature columns

## Output

The application provides:

1. **Prediction Results**:
   - Original data with predictions
   - For classification: Probability scores
   - For regression: Predicted values

2. **Visualizations**:
   - Classification:
     - Prediction distribution (pie chart)
     - Feature importance (bar chart)
   - Regression:
     - Value distribution (histogram)
     - Feature importance (bar chart)

3. **Download Options**:
   - Excel file with all predictions
   - Includes original data and prediction columns

## Dependencies

- streamlit>=1.20.0
- pandas>=1.5.0
- numpy>=1.20.0
- scikit-learn>=1.0.0
- matplotlib>=3.5.0
- seaborn>=0.11.0
- joblib>=1.1.0
- openpyxl>=3.0.0
- xlsxwriter>=3.0.0

## Example Use Cases

1. **Customer Churn Prediction**:
   - Convert monthly charges to binary target
   - Use customer features to predict churn probability

2. **Sales Forecasting**:
   - Use historical sales data
   - Predict future sales amounts

3. **Risk Assessment**:
   - Convert risk scores to binary categories
   - Predict risk levels based on various factors

## Notes

- The application uses RandomForest models for both classification and regression
- Feature importance is calculated and displayed for both prediction types
- Missing values are handled automatically using median/mode imputation
- Categorical features are automatically one-hot encoded
- Numeric features are standardized

## Troubleshooting

1. **No numeric columns found**:
   - Ensure your Excel file contains numeric data
   - Check column data types

2. **Visualization errors**:
   - Try selecting different features
   - Ensure sufficient data for visualization

3. **Prediction errors**:
   - Check data quality
   - Ensure appropriate feature selection
   - Verify target column suitability 