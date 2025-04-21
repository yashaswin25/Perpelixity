import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import time
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning imports
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, confusion_matrix
from sklearn.svm import SVC

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

def train_automl_model(X, y, task_type, time_limit=5):
    """Train a model using GridSearchCV for automated hyperparameter tuning"""
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

        # Display dataset information
        st.write("Dataset Information:")
        st.write(f"Number of samples: {len(X)}")
        st.write(f"Number of features: {X.shape[1]}")
        st.write("Features:", list(feature_names) if feature_names is not None else "Unknown")

        # Determine if dataset is small (less than 100 samples)
        is_small_dataset = len(X) < 100

        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if task_type == "regression":
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Configure models and parameters based on dataset size
            if is_small_dataset:
                st.info("Small dataset detected. Using simplified models to prevent overfitting.")
                # For small datasets, use simpler models and parameters
                model = RandomForestRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [10, 50, 100],
                    'max_depth': [3, 5, 7, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            else:
                st.info("Large dataset detected. Using more complex models and parameters.")
                # For larger datasets, use more complex models
                model = RandomForestRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }

            # Start time tracking
            start_time = time.time()

            # Setup progress updates
            def update_progress():
                elapsed_time = time.time() - start_time
                # Update every second
                if int(elapsed_time) % 2 == 0:
                    progress = min(elapsed_time / (time_limit * 60), 0.99)
                    progress_bar.progress(progress)
                    status_text.text(f"Training in progress... Elapsed time: {elapsed_time:.1f} seconds")

            # Train with GridSearchCV
            try:
                # Setup GridSearchCV
                status_text.text("Starting AutoML training...")

                # Configure GridSearchCV
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=5,
                    scoring='r2',
                    n_jobs=-1,
                    verbose=1
                )

                # Fit the model
                with st.spinner("Training models with GridSearchCV..."):
                    # Start a background thread to update progress
                    import threading
                    stop_event = threading.Event()

                    def update_progress_thread():
                        while not stop_event.is_set():
                            update_progress()
                            time.sleep(0.5)

                    progress_thread = threading.Thread(target=update_progress_thread)
                    progress_thread.start()

                    # Fit the model
                    grid_search.fit(X_train, y_train)

                    # Stop the progress thread
                    stop_event.set()
                    progress_thread.join()

                    # Final update
                    progress_bar.progress(1.0)

                # Get the best model
                best_model = grid_search.best_estimator_

                # Make predictions
                train_predictions = best_model.predict(X_train)
                test_predictions = best_model.predict(X_test)

                # Calculate metrics
                train_r2 = r2_score(y_train, train_predictions)
                test_r2 = r2_score(y_test, test_predictions)
                train_mae = mean_absolute_error(y_train, train_predictions)
                test_mae = mean_absolute_error(y_test, test_predictions)

                # Display results
                st.success(f"""
                AutoML training complete!

                Best parameters: {grid_search.best_params_}

                Training metrics:
                - R² Score: {train_r2:.4f}
                - Mean Absolute Error: {train_mae:.4f}

                Test metrics:
                - R² Score: {test_r2:.4f}
                - Mean Absolute Error: {test_mae:.4f}
                """)

                # Create a comparison DataFrame for test data
                results_df = pd.DataFrame({
                    'Actual': y_test,
                    'Predicted': test_predictions,
                    'Absolute Error': np.abs(y_test - test_predictions)
                })
                st.write("Test Set Predictions vs Actual:")
                st.dataframe(results_df.head(20))

                # Visualize results
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # Scatter plot of actual vs predicted
                ax1.scatter(y_test, test_predictions, alpha=0.5)
                ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                ax1.set_xlabel('Actual Values')
                ax1.set_ylabel('Predicted Values')
                ax1.set_title('Actual vs Predicted Values')

                # Residual plot
                residuals = y_test - test_predictions
                ax2.scatter(test_predictions, residuals, alpha=0.5)
                ax2.axhline(y=0, color='r', linestyle='--')
                ax2.set_xlabel('Predicted Values')
                ax2.set_ylabel('Residuals')
                ax2.set_title('Residual Plot')

                st.pyplot(fig)

                # Feature importance
                if hasattr(best_model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': best_model.feature_importances_
                    }).sort_values('Importance', ascending=False)

                    st.write("### Feature Importance")
                    st.dataframe(importance_df)

                    # Plot feature importance
                    fig, ax = plt.subplots(figsize=(10, 6))
                    importance_df.head(10).plot(kind='bar', x='Feature', y='Importance', ax=ax)
                    plt.title('Top 10 Feature Importance')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)

                return best_model

            except Exception as e:
                st.error(f"Error during AutoML training: {str(e)}")
                # Fallback to a simple model
                st.warning("Falling back to a simple RandomForest model...")
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                return model

        else:  # classification
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Configure models and parameters based on dataset size
            if is_small_dataset:
                st.info("Small dataset detected. Using simplified models to prevent overfitting.")
                # For small datasets, use simpler models and parameters
                model = RandomForestClassifier(random_state=42)
                param_grid = {
                    'n_estimators': [10, 50, 100],
                    'max_depth': [3, 5, 7, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            else:
                st.info("Large dataset detected. Using more complex models and parameters.")
                # For larger datasets, use more complex models
                model = RandomForestClassifier(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }

            # Start time tracking
            start_time = time.time()

            # Setup progress updates
            def update_progress():
                elapsed_time = time.time() - start_time
                # Update every second
                if int(elapsed_time) % 2 == 0:
                    progress = min(elapsed_time / (time_limit * 60), 0.99)
                    progress_bar.progress(progress)
                    status_text.text(f"Training in progress... Elapsed time: {elapsed_time:.1f} seconds")

            # Train with GridSearchCV
            try:
                # Setup GridSearchCV
                status_text.text("Starting AutoML training...")

                # Configure GridSearchCV
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=5,
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=1
                )

                # Fit the model
                with st.spinner("Training models with GridSearchCV..."):
                    # Start a background thread to update progress
                    import threading
                    stop_event = threading.Event()

                    def update_progress_thread():
                        while not stop_event.is_set():
                            update_progress()
                            time.sleep(0.5)

                    progress_thread = threading.Thread(target=update_progress_thread)
                    progress_thread.start()

                    # Fit the model
                    grid_search.fit(X_train, y_train)

                    # Stop the progress thread
                    stop_event.set()
                    progress_thread.join()

                    # Final update
                    progress_bar.progress(1.0)

                # Get the best model
                best_model = grid_search.best_estimator_

                # Make predictions
                train_predictions = best_model.predict(X_train)
                test_predictions = best_model.predict(X_test)

                # Calculate metrics
                train_accuracy = accuracy_score(y_train, train_predictions)
                test_accuracy = accuracy_score(y_test, test_predictions)

                # Display results
                st.success(f"""
                AutoML training complete!

                Best parameters: {grid_search.best_params_}

                Training metrics:
                - Accuracy: {train_accuracy:.4f}

                Test metrics:
                - Accuracy: {test_accuracy:.4f}
                """)

                # Create confusion matrices
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # Training confusion matrix
                cm_train = confusion_matrix(y_train, train_predictions)
                sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=ax1)
                ax1.set_title('Training Confusion Matrix')
                ax1.set_xlabel('Predicted')
                ax1.set_ylabel('Actual')

                # Test confusion matrix
                cm_test = confusion_matrix(y_test, test_predictions)
                sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=ax2)
                ax2.set_title('Test Confusion Matrix')
                ax2.set_xlabel('Predicted')
                ax2.set_ylabel('Actual')

                st.pyplot(fig)

                # Feature importance
                if hasattr(best_model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': best_model.feature_importances_
                    }).sort_values('Importance', ascending=False)

                    st.write("### Feature Importance")
                    st.dataframe(importance_df)

                    # Plot feature importance
                    fig, ax = plt.subplots(figsize=(10, 6))
                    importance_df.head(10).plot(kind='bar', x='Feature', y='Importance', ax=ax)
                    plt.title('Top 10 Feature Importance')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)

                return best_model

            except Exception as e:
                st.error(f"Error during AutoML training: {str(e)}")
                # Fallback to a simple model
                st.warning("Falling back to a simple RandomForest model...")
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                return model

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
                    with st.spinner(f"Processing data and training model..."):
                        try:
                            # Process data
                            X_processed, y = process_data_for_training(df, target_column, user_selected_features)

                            if X_processed is not None and y is not None:
                                # Train model
                                model = train_automl_model(X_processed, y, task["type"], time_limit)

                                if model is not None:
                                    # Generate predictions on the full dataset
                                    st.subheader("Generating predictions on the full dataset...")

                                    # Create a copy of the original dataframe for predictions
                                    df_with_predictions = df.copy()

                                    # Generate predictions
                                    try:
                                        # Get the processed features for prediction
                                        X_for_prediction = X_processed

                                        # Make predictions
                                        predictions = model.predict(X_for_prediction)

                                        # Add predictions to the dataframe
                                        pred_col_name = f"predicted_{target_column}"
                                        df_with_predictions[pred_col_name] = predictions

                                        # For classification, try to get probabilities if available
                                        if task["type"] == "classification" and hasattr(model, 'predict_proba'):
                                            try:
                                                probas = model.predict_proba(X_for_prediction)
                                                # Add probability of positive class (assuming binary classification)
                                                if probas.shape[1] >= 2:  # At least two classes
                                                    df_with_predictions[f"{pred_col_name}_probability"] = probas[:, 1]
                                            except Exception as e:
                                                st.warning(f"Could not generate prediction probabilities: {str(e)}")

                                        # Display results
                                        st.subheader("Prediction Results")
                                        st.dataframe(df_with_predictions)

                                        # Provide download link
                                        st.markdown("### Download Predictions")
                                        st.markdown(get_download_link(df_with_predictions, "automl_predictions.xlsx"), unsafe_allow_html=True)

                                        # Save model
                                        model_filename = f"automl_{task['type']}_{int(time.time())}.joblib"
                                        model_path = os.path.join(MODEL_DIR, model_filename)
                                        joblib.dump(model, model_path)
                                        st.success(f"Model saved as {model_filename}")

                                    except Exception as e:
                                        st.error(f"Error generating predictions: {str(e)}")
                                else:
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