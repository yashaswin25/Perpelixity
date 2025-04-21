# ML Prediction System (AutoML Version)

This is the AutoML version of the ML Prediction System, which uses automated machine learning to find the best model and hyperparameters for your data.

## Key Differences from Original Version

1. **Automated Model Selection**:
   - Uses GridSearchCV to automatically find the best hyperparameters
   - Optimizes model performance through extensive parameter search
   - Tries multiple configurations to find the optimal settings

2. **Training Time Control**:
   - Configurable training time limit
   - Shows training progress and time taken
   - Displays model configuration details

3. **Enhanced Model Information**:
   - Shows which models were tried
   - Displays model performance metrics
   - Provides detailed model configuration

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: TPOT requires several dependencies that will be installed automatically with the requirements.txt file.

## Usage

1. Run the application:
```bash
streamlit run ml_prediction_system_automl.py
```

2. Access the application at `http://localhost:8502`

3. Using the Application:

### For Classification:

1. Select "Classification" in the sidebar
2. Set the desired training time (30-300 seconds)
3. Upload your Excel file
4. Create a binary target column if needed
5. Select features for prediction
6. Click "Process and Predict"
7. View results and model information

### For Regression:

1. Select "Regression" in the sidebar
2. Set the desired training time (30-300 seconds)
3. Upload your Excel file
4. Select target column
5. Select features for prediction
6. Click "Process and Predict"
7. View results and model information

## Features

- **AutoML Capabilities**:
  - Automatic model selection
  - Hyperparameter optimization
  - Ensemble model creation
  - Feature preprocessing

- **User Interface**:
  - Training time control
  - Model configuration display
  - Performance metrics
  - Visualizations

- **Data Handling**:
  - Excel file support
  - Missing value handling
  - Feature preprocessing
  - Binary target creation

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
- scikit-learn>=1.0.0
- pandas>=1.5.0
- numpy>=1.20.0
- matplotlib>=3.5.0
- seaborn>=0.11.0

## Notes

- The AutoML version may take longer to train than the original version
- Training time can be adjusted based on your needs
- More computational resources may be required
- Results may vary based on the training time allowed

## Troubleshooting

1. **Installation Issues**:
   - Ensure SWIG is installed correctly
   - Check system requirements for auto-sklearn
   - Verify all dependencies are installed

2. **Training Issues**:
   - Increase training time if results are not satisfactory
   - Check data quality and preprocessing
   - Ensure sufficient computational resources

3. **Memory Issues**:
   - Reduce training time
   - Use smaller datasets
   - Increase system memory if possible