import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneOut, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import time
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# Set page config
st.set_page_config(
    page_title="ML Prediction System (AutoML)",
    page_icon="🤖",
    layout="wide"
)

# Define available tasks and their properties
TASKS = {
    "classification": {
        "name": "Classification",
        "type": "classification",
        "description": "Predict categorical outcomes using AutoML"
    },
    "regression": {
        "name": "Regression",
        "type": "regression",
        "description": "Predict continuous values using AutoML"
    }
}

# Directory for storing models
MODEL_DIR = "models_automl"
os.makedirs(MODEL_DIR, exist_ok=True)

# Helper functions
def validate_dataframe(df):
    """Validate the input dataframe"""
    if df is None:
        raise ValueError("DataFrame is None")
    if df.empty:
        raise ValueError("DataFrame is empty")
    if len(df.columns) < 2:
        raise ValueError("DataFrame must have at least 2 columns (features and target)")
    return True

def train_automl_model(X, y, task_type, time_limit=60):
    """Train a model using scikit-learn's models with dynamic handling for different dataset sizes"""
    try:
        # Store original feature names before preprocessing
        feature_names = X.columns if isinstance(X, pd.DataFrame) else None
        
        # Validate input data
        if X is None or y is None:
            raise ValueError("Input data is None")
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Input data is empty")
        if len(X) != len(y):
            raise ValueError("X and y have different lengths")
        
        st.write("Dataset Information:")
        st.write(f"Number of samples: {len(X)}")
        st.write(f"Number of features: {X.shape[1]}")
        st.write("Features:", list(feature_names) if feature_names is not None else "Unknown")
        
        # Determine if dataset is small (less than 100 samples)
        is_small_dataset = len(X) < 100
        
        if task_type == "regression":
            if is_small_dataset:
                st.info("Small dataset detected. Using simplified models to prevent overfitting.")
                models = [
                    ('rf', RandomForestRegressor(
                        n_estimators=10,
                        max_depth=3,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        random_state=42
                    )),
                    ('lr', LinearRegression()),
                    ('ridge', Ridge(alpha=1.0)),
                    ('lasso', Lasso(alpha=1.0))
                ]
            else:
                st.info("Large dataset detected. Using full models with cross-validation.")
                models = [
                    ('rf', RandomForestRegressor(
                        n_estimators=100,
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        random_state=42
                    )),
                    ('lr', LinearRegression()),
                    ('ridge', Ridge(alpha=1.0)),
                    ('lasso', Lasso(alpha=1.0))
                ]
            
            best_score = -np.inf
            best_model = None
            best_model_name = ""
            
            # Try each model
            for model_name, model in models:
                try:
                    st.write(f"\nTraining {model_name}...")
                    
                    if is_small_dataset:
                        # For small datasets, use the entire dataset
                        model.fit(X, y)
                        predictions = model.predict(X)
                    else:
                        # For large datasets, use cross-validation
                        scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                        model.fit(X, y)
                        predictions = model.predict(X)
                        st.write(f"Cross-validation R² scores: {scores}")
                        st.write(f"Mean R²: {scores.mean():.4f} (±{scores.std() * 2:.4f})")
                    
                    # Calculate metrics
                    r2 = r2_score(y, predictions)
                    mae = mean_absolute_error(y, predictions)
                    
                    st.write(f"{model_name} Training Metrics:")
                    st.write(f"R² Score: {r2:.4f}")
                    st.write(f"Mean Absolute Error: {mae:.4f}")
                    
                    # Create a comparison DataFrame
                    results_df = pd.DataFrame({
                        'Actual': y,
                        'Predicted': predictions,
                        'Absolute Error': np.abs(y - predictions)
                    })
                    st.write(f"\n{model_name} Predictions vs Actual:")
                    st.dataframe(results_df)
                    
                    if r2 > best_score:
                        best_score = r2
                        best_model = model
                        best_model_name = model_name
                
                except Exception as e:
                    st.warning(f"Warning: Could not train {model_name}: {str(e)}")
                    continue
            
            if best_model is None:
                raise ValueError("No model could be successfully trained")
            
            st.success(f"""
            Best model: {best_model_name}
            R² Score: {best_score:.4f}
            Mean Absolute Error: {mae:.4f}
            """)
            
            # Feature importance for Random Forest
            if best_model_name == 'rf' and feature_names is not None:
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': best_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                st.write("\nFeature Importance:")
                st.dataframe(importance_df)
                
                # Plot feature importance
                fig, ax = plt.subplots(figsize=(10, 6))
                importance_df.plot(kind='bar', x='Feature', y='Importance', ax=ax)
                plt.title('Feature Importance')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
            return best_model
            
        else:  # classification
            if is_small_dataset:
                st.info("Small dataset detected. Using simplified models to prevent overfitting.")
                models = [
                    ('rf', RandomForestClassifier(
                        n_estimators=10,
                        max_depth=3,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        random_state=42
                    )),
                    ('lr', LogisticRegression(max_iter=1000)),
                    ('svc', SVC(kernel='linear', probability=True))
                ]
            else:
                st.info("Large dataset detected. Using full models with cross-validation.")
                models = [
                    ('rf', RandomForestClassifier(
                        n_estimators=100,
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        random_state=42
                    )),
                    ('lr', LogisticRegression(max_iter=1000)),
                    ('svc', SVC(kernel='linear', probability=True))
                ]
            
            best_score = -np.inf
            best_model = None
            best_model_name = ""
            
            # Try each model
            for model_name, model in models:
                try:
                    st.write(f"\nTraining {model_name}...")
                    
                    if is_small_dataset:
                        # For small datasets, use the entire dataset
                        model.fit(X, y)
                        predictions = model.predict(X)
                    else:
                        # For large datasets, use cross-validation
                        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
                        model.fit(X, y)
                        predictions = model.predict(X)
                        st.write(f"Cross-validation accuracy scores: {scores}")
                        st.write(f"Mean accuracy: {scores.mean():.4f} (±{scores.std() * 2:.4f})")
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y, predictions)
                    
                    st.write(f"{model_name} Training Metrics:")
                    st.write(f"Accuracy: {accuracy:.4f}")
                    
                    # Create a confusion matrix
                    cm = confusion_matrix(y, predictions)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    plt.title('Confusion Matrix')
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    st.pyplot(fig)
                    
                    if accuracy > best_score:
                        best_score = accuracy
                        best_model = model
                        best_model_name = model_name
                
                except Exception as e:
                    st.warning(f"Warning: Could not train {model_name}: {str(e)}")
                    continue
            
            if best_model is None:
                raise ValueError("No model could be successfully trained")
            
            st.success(f"""
            Best model: {best_model_name}
            Accuracy: {best_score:.4f}
            """)
            
            # Feature importance for Random Forest
            if best_model_name == 'rf' and feature_names is not None:
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': best_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                st.write("\nFeature Importance:")
                st.dataframe(importance_df)
                
                # Plot feature importance
                fig, ax = plt.subplots(figsize=(10, 6))
                importance_df.plot(kind='bar', x='Feature', y='Importance', ax=ax)
                plt.title('Feature Importance')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
            return best_model
            
    except Exception as e:
        st.error(f"Error during model training: {str(e)}")
        st.error("Debugging information:")
        st.write("X shape:", X.shape if X is not None else "None")
        st.write("y shape:", y.shape if y is not None else "None")
        st.write("X type:", type(X))
        st.write("First few rows of X:", X[:5] if X is not None else "None")
        return None

def create_preprocessing_pipeline(X):
    """Create a preprocessing pipeline based on data types"""
    try:
        # Ensure we have valid data
        if X is None or X.empty:
            raise ValueError("Input data is empty or None")
        
        st.write("Original Data Types:")
        st.write(X.dtypes)
        
        # First convert date strings to datetime objects
        date_columns = []
        for col in X.columns:
            try:
                if X[col].dtype == 'object':
                    # Try to parse as datetime with a specific format
                    pd.to_datetime(X[col], format='%d-%m-%Y', errors='raise')
                    date_columns.append(col)
            except:
                continue
        
        # Create date features first
        if date_columns:
            st.write("Processing date columns:", date_columns)
            for col in date_columns:
                try:
                    dates = pd.to_datetime(X[col], format='%d-%m-%Y')
                    # Extract numeric features from dates
                    X[f'{col}_year'] = dates.dt.year
                    X[f'{col}_month'] = dates.dt.month
                    X[f'{col}_day'] = dates.dt.day
                    # Drop original date column
                    X = X.drop(columns=[col])
                except Exception as e:
                    st.warning(f"Could not process date column {col}: {str(e)}")
        
        # Now identify feature types after date processing
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        
        st.write("\nProcessed Features:")
        st.write("Numeric features:", list(numeric_features))
        st.write("Categorical features:", list(categorical_features))
        
        transformers = []
        
        # Add numeric transformer if we have numeric features
        if len(numeric_features) > 0:
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),  # Changed to mean for small datasets
                ('scaler', StandardScaler())  # Changed back to StandardScaler for small datasets
            ])
            transformers.append(('num', numeric_transformer, numeric_features))
        
        # Add categorical transformer if we have categorical features
        if len(categorical_features) > 0:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', categorical_transformer, categorical_features))
        
        preprocessor = ColumnTransformer(transformers=transformers)
        return preprocessor
    except Exception as e:
        st.error(f"Error in preprocessing pipeline: {str(e)}")
        st.error("Please check your data format and try again.")
        return None

def extract_date_features(X):
    """Extract useful features from datetime columns"""
    if isinstance(X, pd.DataFrame):
        X = pd.to_datetime(X.iloc[:, 0])
    else:
        X = pd.to_datetime(X)
    
    return pd.DataFrame({
        'year': X.dt.year,
        'month': X.dt.month,
        'day': X.dt.day,
        'dayofweek': X.dt.dayofweek,
        'quarter': X.dt.quarter
    })

def get_download_link(df, filename="predictions.xlsx"):
    """Generate a download link for a dataframe"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    b64 = base64.b64encode(output.getvalue()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download Excel file</a>'
    return href

def visualize_predictions(df, task_type, prediction_col, model, feature_names):
    """Create visualizations based on predictions and task type"""
    try:
        if task_type == "classification":
            # Distribution of predictions
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Pie chart of prediction distribution
            prediction_counts = df[prediction_col].value_counts()
            ax1.pie(prediction_counts, labels=prediction_counts.index, autopct='%1.1f%%')
            ax1.set_title('Prediction Distribution')
            
            # Bar chart for top factors
            if hasattr(model, 'feature_importances_'):
                importances = pd.Series(model.feature_importances_, index=feature_names)
                importances = importances.sort_values(ascending=False).head(10)
                importances.plot(kind='bar', ax=ax2)
                ax2.set_title('Top 10 Important Features')
                ax2.set_ylabel('Importance')
            else:
                ax2.text(0.5, 0.5, 'Feature importance not available', 
                         horizontalalignment='center', verticalalignment='center')
            
            st.pyplot(fig)
                
        else:  # regression
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
            
            # Scatter plot of actual vs predicted
            ax1.scatter(df[prediction_col.replace('predicted_', '')], df[prediction_col], alpha=0.5)
            ax1.plot([df[prediction_col].min(), df[prediction_col].max()], 
                    [df[prediction_col].min(), df[prediction_col].max()], 'r--')
            ax1.set_xlabel('Actual Values')
            ax1.set_ylabel('Predicted Values')
            ax1.set_title('Actual vs Predicted Values')
            
            # Residual plot
            residuals = df[prediction_col.replace('predicted_', '')] - df[prediction_col]
            ax2.scatter(df[prediction_col], residuals, alpha=0.5)
            ax2.axhline(y=0, color='r', linestyle='--')
            ax2.set_xlabel('Predicted Values')
            ax2.set_ylabel('Residuals')
            ax2.set_title('Residual Plot')
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importances = pd.Series(model.feature_importances_, index=feature_names)
                importances = importances.sort_values(ascending=False).head(10)
                importances.plot(kind='bar', ax=ax3)
                ax3.set_title('Top 10 Important Features')
                ax3.set_ylabel('Importance')
            else:
                ax3.text(0.5, 0.5, 'Feature importance not available', 
                         horizontalalignment='center', verticalalignment='center')
                
            st.pyplot(fig)
            
            # Calculate and display regression metrics
            actual = df[prediction_col.replace('predicted_', '')]
            predicted = df[prediction_col]
            r2 = r2_score(actual, predicted)
            mae = mean_absolute_error(actual, predicted)
            
            st.write("**Regression Metrics:**")
            st.write(f"- R² Score: {r2:.4f}")
            st.write(f"- Mean Absolute Error: {mae:.4f}")
            
    except Exception as e:
        st.error(f"Error creating visualizations: {str(e)}")

def create_binary_target(df, column_name, threshold, above_threshold_label="Yes", below_threshold_label="No"):
    """Create a binary target column from a continuous column"""
    df[f"{column_name}_binary"] = df[column_name].apply(
        lambda x: above_threshold_label if x > threshold else below_threshold_label
    )
    return f"{column_name}_binary"

def process_data_for_training(df, target_column, selected_features):
    """Process data before training"""
    try:
        # Select features and target
        X = df[selected_features].copy()
        y = df[target_column].copy()
        
        # Create and fit preprocessing pipeline
        preprocessor = create_preprocessing_pipeline(X)
        if preprocessor is None:
            raise ValueError("Failed to create preprocessing pipeline")
        
        # Transform the features
        X_processed = preprocessor.fit_transform(X)
        
        # Get feature names after preprocessing if possible
        if hasattr(preprocessor, 'get_feature_names_out'):
            feature_names = preprocessor.get_feature_names_out()
            X_processed = pd.DataFrame(X_processed, columns=feature_names)
        
        return X_processed, y
        
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None, None

# Main application
def main():
    st.title("Machine Learning Prediction System (AutoML)")
    st.write("Upload an Excel file and select a prediction type to get started.")
    
    # Sidebar for task selection
    with st.sidebar:
        st.header("Prediction Type")
        task_id = st.selectbox(
            "Select prediction type:",
            options=list(TASKS.keys()),
            format_func=lambda x: TASKS[x]["name"]
        )
        
        task = TASKS[task_id]
        st.write(f"**Description**: {task['description']}")
        
        # Add time limit selection for AutoML
        time_limit = st.slider(
            "AutoML Training Time (minutes):",
            min_value=1,
            max_value=30,
            value=5,
            step=1
        )
    
    # File upload
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # Load the data with explicit error handling
            try:
                df = pd.read_excel(uploaded_file)
                if df is None or df.empty:
                    st.error("The uploaded file is empty or invalid.")
                    return
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                st.error("Please make sure your Excel file is not empty and contains valid data.")
                return
            
            st.write("Data Preview:")
            st.dataframe(df.head())
            
            # Show basic info
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Rows:** {df.shape[0]}")
                st.write(f"**Columns:** {df.shape[1]}")
            with col2:
                missing_percentage = (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                st.write(f"**Missing values:** {missing_percentage:.2f}%")
            
            # Feature Selection
            st.subheader("Feature Selection")
            
            if task["type"] == "classification":
                # For classification, let user create a binary target
                st.write("**Create Binary Target Column**")
                numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                
                if len(numeric_columns) > 0:
                    base_column = st.selectbox(
                        "Select a numeric column to create binary target:",
                        options=numeric_columns
                    )
                    
                    if base_column:
                        # Show statistics of the selected column
                        col_stats = df[base_column].describe()
                        st.write("**Column Statistics:**")
                        st.write(col_stats)
                        
                        # Let user set threshold
                        threshold = st.number_input(
                            "Set threshold for binary classification:",
                            value=float(col_stats['mean']),
                            step=float(col_stats['std']/10) if col_stats['std'] > 0 else 0.1
                        )
                        
                        # Create binary target
                        target_column = create_binary_target(df, base_column, threshold)
                        st.success(f"Created binary target column: {target_column}")
                        st.write(f"Distribution of new target column:")
                        st.write(df[target_column].value_counts())
                    else:
                        st.error("Please select a numeric column for classification.")
                        return
                else:
                    st.error("No numeric columns found in the dataset. Please upload a dataset with numeric columns.")
                    return
            else:
                # For regression, just select target column
                if len(df.columns) > 0:
                    target_column = st.selectbox(
                        "Select the target column:",
                        options=df.columns,
                        index=0 if len(df.columns) > 0 else None
                    )
                else:
                    st.error("No columns found in the dataset.")
                    return
            
            # Validate target column based on task type
            if task["type"] == "classification":
                unique_values = df[target_column].nunique()
                if unique_values > 10:
                    st.warning(f"Warning: The selected target column has {unique_values} unique values. For classification, it's recommended to have a limited number of categories.")
            else:  # regression
                if not pd.api.types.is_numeric_dtype(df[target_column]):
                    st.warning("Warning: The selected target column is not numeric. For regression tasks, the target should be numeric.")
            
            # Let user select features for prediction
            available_features = [col for col in df.columns if col != target_column and not col.endswith('_binary')]
            if not available_features:
                st.error("No features available for selection after excluding target column.")
                return
                
            user_selected_features = st.multiselect(
                "Select features to use for prediction:",
                options=available_features,
                default=available_features[:min(10, len(available_features))]
            )

            # Process and predict button
            if st.button("Process and Predict"):
                if not user_selected_features:
                    st.error("Please select at least one feature column.")
                else:
                    with st.spinner(f"Training model..."):
                        try:
                            # Process data
                            X_processed, y = process_data_for_training(df, target_column, user_selected_features)
                            
                            if X_processed is not None and y is not None:
                                # Train model
                                model = train_automl_model(X_processed, y, task["type"], time_limit)
                                
                                if model is None:
                                    st.error("Failed to train the model.")
                            else:
                                st.error("Failed to process the data.")
                        except Exception as e:
                            st.error(f"Error during model training and prediction: {str(e)}")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.error("Please check if your Excel file is properly formatted and not corrupted.")

if __name__ == "__main__":
    main() 