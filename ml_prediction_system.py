import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Set page config
st.set_page_config(
    page_title="ML Prediction System",
    page_icon="🤖",
    layout="wide"
)

# Define available tasks and their properties
TASKS = {
    "classification": {
        "name": "Classification",
        "type": "classification",
        "description": "Predict categorical outcomes"
    },
    "regression": {
        "name": "Regression",
        "type": "regression",
        "description": "Predict continuous values"
    }
}

# Directory for storing models
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Helper functions
def train_model(X, y, task_type):
    """Train a new model based on task type"""
    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:  # regression
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model.fit(X, y)
    return model

def create_preprocessing_pipeline(X, task_type):
    """Create a preprocessing pipeline based on data types"""
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram of predictions
        ax1.hist(df[prediction_col], bins=20)
        ax1.set_title('Distribution of Predicted Values')
        ax1.set_xlabel('Predicted Value')
        ax1.set_ylabel('Frequency')
        
        # Feature importance
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

def create_binary_target(df, column_name, threshold, above_threshold_label="Yes", below_threshold_label="No"):
    """Create a binary target column from a continuous column"""
    df[f"{column_name}_binary"] = df[column_name].apply(
        lambda x: above_threshold_label if x > threshold else below_threshold_label
    )
    return f"{column_name}_binary"

def train_automl(df, task_id, user_selected_features, target_column):
    """Train a model using AutoML approach"""
    task = TASKS[task_id]
    task_type = task["type"]
    
    # Use user specified target column or the default one from task definition
    if target_column is None or target_column not in df.columns:
        st.warning(f"Target column not found. Using default target name: {task['target_column']}")
        if task["target_column"] not in df.columns:
            st.error(f"Default target column '{task['target_column']}' not found in data. Please select a target column.")
            return None, None
        target_column = task["target_column"]
    
    y = df[target_column]
    X = df[user_selected_features]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create preprocessing pipeline
    pipeline = create_preprocessing_pipeline(X, task_type)
    
    # Fit preprocessing pipeline
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)
    
    # Train model
    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_processed, y_train)
        y_pred = model.predict(X_test_processed)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Model trained with accuracy: {accuracy:.2f}")
    else:  # regression
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_processed, y_train)
        y_pred = model.predict(X_test_processed)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        st.write(f"Model trained with R² score: {r2:.2f}, Mean Absolute Error: {mae:.2f}")
    
    return model, pipeline, user_selected_features

# Main application
def main():
    st.title("Machine Learning Prediction System")
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
    
    # File upload
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        # Load the data
        try:
            df = pd.read_excel(uploaded_file)
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
                numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
                
                if len(numeric_columns) > 0:
                    base_column = st.selectbox(
                        "Select a numeric column to create binary target:",
                        options=numeric_columns
                    )
                    
                    # Show statistics of the selected column
                    col_stats = df[base_column].describe()
                    st.write("**Column Statistics:**")
                    st.write(col_stats)
                    
                    # Let user set threshold
                    threshold = st.number_input(
                        "Set threshold for binary classification:",
                        value=float(col_stats['mean']),
                        step=float(col_stats['std']/10)
                    )
                    
                    # Create binary target
                    target_column = create_binary_target(df, base_column, threshold)
                    st.success(f"Created binary target column: {target_column}")
                    st.write(f"Distribution of new target column:")
                    st.write(df[target_column].value_counts())
                else:
                    st.error("No numeric columns found in the dataset. Please upload a dataset with numeric columns.")
                    return
            else:
                # For regression, just select target column
                target_column = st.selectbox(
                    "Select the target column:",
                    options=df.columns,
                    index=0
                )
            
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
                    with st.spinner("Processing data and generating predictions..."):
                        # Train model
                        model, pipeline, features = train_automl(df, task_id, user_selected_features, target_column)
                        
                        if model is not None and pipeline is not None:
                            # Prepare input data
                            X_pred = df[user_selected_features]
                            
                            # Transform prediction data
                            X_processed = pipeline.transform(X_pred)
                            
                            # Generate predictions
                            if task["type"] == "classification":
                                predictions = model.predict(X_processed)
                                pred_col = f"predicted_{target_column}"
                                
                                # Get prediction probabilities if available
                                try:
                                    proba = model.predict_proba(X_processed)
                                    # Add probability of positive class
                                    df[f"{pred_col}_probability"] = proba[:, 1]
                                except:
                                    pass
                            else:  # regression
                                predictions = model.predict(X_processed)
                                pred_col = f"predicted_{target_column}"
                            
                            # Add predictions to dataframe
                            df[pred_col] = predictions
                            
                            st.success("Model training and prediction complete!")
                            
                            # Display results
                            st.subheader("Prediction Results")
                            st.dataframe(df)
                            
                            # Show download link
                            st.markdown(get_download_link(df), unsafe_allow_html=True)
                            
                            # Create visualizations
                            st.subheader("Visualizations")
                            try:
                                # Get feature names for importance plotting
                                if hasattr(pipeline, 'get_feature_names_out'):
                                    feature_names = pipeline.get_feature_names_out()
                                else:
                                    feature_names = user_selected_features
                                
                                visualize_predictions(df, task["type"], pred_col, model, feature_names)
                            except Exception as e:
                                st.error(f"Error creating visualizations: {str(e)}")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main() 