# This scripts purpose is to predict customer lifetime value for one year
# ahead. As well as creating visualizations of data distrisbution and feature
# importances, and preparing the final outputs:
# final_customer_insigths.csv
# final_insights_barplot.png
#
# Author: Per Idar Rød.
# post@peridar.net


import os
import pickle
import math
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (StandardScaler,
                                   OneHotEncoder,
                                   FunctionTransformer)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (train_test_split,
                                     cross_val_score,
                                     RandomizedSearchCV,
                                     GridSearchCV)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


# Setting paths
csv_dir = './csv_files/'
model_dir = './models/'
plot_dir = './plots/'
os.makedirs(csv_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)


def check_data_distribution(data, feature_columns, categorical_features):
    """
    Check the distribution of the data by plotting histograms and box plots for numerical features.

    Args:
        data (pd.DataFrame): The input data.
        feature_columns (list): List of feature columns to analyze.
        categorical_features (list): List of categorical feature columns.

    Returns:
        None
    """

    print("\nData distribution - Summary Statistics:\n",
          data[feature_columns].describe())

    numerical_features = [
        col for col in feature_columns if col not in categorical_features]

    num_features = len(numerical_features)
    num_cols = 3
    num_rows = math.ceil(num_features / num_cols)

    # Histogram with KDE
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
    for i, col in enumerate(numerical_features):
        row, col_idx = divmod(i, 3)
        sns.histplot(data[col], bins=30, kde=True, ax=axs[row, col_idx])
        axs[row, col_idx].set_title(f'Distribution of {col}')
    fig.tight_layout()
    fig.savefig(f"{plot_dir}Histogram_with_KDE.png")
    print("\nSaved Histogram to ./plots")
    plt.close(fig)

    # Box-and-whisker plots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
    for i, col in enumerate(numerical_features):
        row, col_idx = divmod(i, 3)
        sns.boxplot(y=data[col], ax=axs[row, col_idx])
        axs[row, col_idx].set_title(f'Box plot of {col}')
    fig.tight_layout()
    fig.savefig(f"{plot_dir}Box_and_whisker.png")
    print("\nSaved Box-and-whisker plot to ./plots")
    plt.close(fig)


def plot_feature_importances(importances, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=list(importances.values()), y=list(
        importances.keys()), color='darkblue')
    ax.set_title(title)
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    fig.tight_layout()
    fig.savefig(f"{plot_dir}{title}.png", dpi=300,
                format='png', bbox_inches='tight')
    plt.close(fig)


def preprocess_data(data, feature_columns, categorical_features):
    """Function for preprocessing of data.
    Returns preprocessor
    """
    numerical_features = [
        col for col in feature_columns if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            # ('log', log_transformer, numerical_features),
            ('scaler', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ], remainder='passthrough'
    )

    return preprocessor


def run_models(
        data,
        feature_columns,
        categorical_features,
        use_gridsearch=True):
    """
    Run machine learning models on the provided data,
    including hyperparameter tuning,
    model evaluation, and saving the results.

    Args:
        data (pd.DataFrame): The input data.
        feature_columns (list): List of feature columns to be
        used in the model.
        categorical_features (list): List of categorical feature columns.
        use_gridsearch (bool): Whether to use GridSearch for
        hyperparameter tuning.

    Returns:
        pd.DataFrame: DataFrame containing the evaluation metrics
        of the trained models.
    """

    # Split the data
    X = data[feature_columns]
    y = data['FutureSpend'].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1)

    # Scale the target variable
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    # Preproccess the features
    preprocessor = preprocess_data(data, feature_columns, categorical_features)

    # Define models
    models = {
        'Linear Regression': LinearRegression(n_jobs=-1),
        'Random Forest': RandomForestRegressor(
            random_state=1,
            # verbose=1,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            random_state=1,
            # verbose=1,
        )}

    # Hyperparameter grids
    param_grids = {
        'Random Forest': {
            'model__n_estimators': [250, 300, 350],
            'model__max_depth': [10, 15, 20],
            'model__min_samples_split': [4, 5, 6]
        },
        'Gradient Boosting': {
            'model__n_estimators': [250, 300, 350],
            'model__learning_rate': [0.09, 0.1, 0.11],
            'model__max_depth': [2, 4, 6],
            'model__min_samples_split': [2, 3, 4]

        }
    }

    results = []
    # Pipeline setup and running models
    for name, model in models.items():
        model_path = f'{model_dir}{name}.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                pipeline, y_scaler = pickle.load(f)
            print(f"Loaded {name} and scaler from pickle.\n")
        else:
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            if use_gridsearch and name in param_grids:
                random_search = RandomizedSearchCV(
                    pipeline, param_grids[name],
                    cv=5, scoring='neg_mean_squared_error',
                    n_jobs=-1)

                random_search.fit(X_train, y_train_scaled.ravel())
                pipeline = random_search.best_estimator_
                best_params = random_search.best_params_
                try:
                    best_params_df = pd.DataFrame([best_params])
                    best_params_df.to_csv(
                        f"{csv_dir}best_parameters_{name}.csv")
                except Exception as e:
                    print(f"Saving csv failed..{e}")

                print(
                    f"Best parameters for {name}: {random_search.best_params_}")

            else:
                pipeline.fit(X_train, y_train_scaled.ravel())

            with open(model_path, 'wb') as f:
                pickle.dump((pipeline, y_scaler), f)
            print(f"Saved {name} model and scaler to pickle.")

        # Predictions and evaluations
        y_pred_scaled = pipeline.predict(X_test)
        y_pred = y_scaler.inverse_transform(
            y_pred_scaled.reshape(-1, 1)).flatten()
        y_test_original = y_scaler.inverse_transform(
            y_test_scaled.reshape(-1, 1)).flatten()

        mse = mean_squared_error(y_test_original, y_pred)
        r2 = r2_score(y_test_original, y_pred)
        result = {'Model': name, 'Test MSE': mse, 'R-squared': r2}

        # Feature importances for applicable models
        if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
            feature_importances = pipeline.named_steps['model']\
                .feature_importances_
            feature_names = [col for col in feature_columns
                             if col not in categorical_features] + \
                list(pipeline.named_steps['preprocessor'].named_transformers_[
                    'cat'].get_feature_names_out())
            result['Feature Importances'] = dict(
                zip(feature_names, feature_importances))

        results.append(result)

    training_results_df = pd.DataFrame(results)
    print("\nResults DF:\n", training_results_df)
    training_results_df.to_csv(
        f"{csv_dir}model_evaluation_metrics.csv", index=False)
    print(
        f"\nModel evaluation metrics saved to {csv_dir}model_evaluation_metrics.csv")

    for index, row in training_results_df.iterrows():
        if row['Model'] in ['Random Forest', 'Gradient Boosting']:
            plot_feature_importances(
                row['Feature Importances'],
                f"{row['Model']} Feature Importances")

    return training_results_df


def make_predictions(data, feature_columns):
    """
    Make predictions using pre-trained models and inverse
    transform the scaled predictions.

    Args:
        data (pd.DataFrame): The input data.
        feature_columns (list): List of feature columns to be used
        for making predictions.

    Returns:
        pd.DataFrame: DataFrame with predictions added.
    """

    model_names = ['Linear Regression', 'Random Forest', 'Gradient Boosting']
    predictions = {}

    for name in model_names:
        model_path = f'{model_dir}{name}.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                pipeline, y_scaler = pickle.load(f)
            print(f"Loaded {name} and scaler from pickle.\n")

            X = data[feature_columns]
            y_pred_scaled = pipeline.predict(X).reshape(-1, 1)
            y_pred = y_scaler.inverse_transform(
                y_pred_scaled).flatten()

            # Clip negative predictions
            y_pred = [max(0, pred) for pred in y_pred]

            predictions[name] = y_pred
            data[f"{name}_Predictions"] = y_pred

        # Ensuring predictions are numeric
        data[f"{name}_Predictions"] = pd.to_numeric(
            data[f"{name}_Predictions"])

    return data


def cross_val_performance(data, feature_columns, categorical_features):
    print("Starting Cross-validation!")
    X = data[feature_columns]
    y = data['FutureSpend']

    preprocessor = preprocess_data(data, feature_columns, categorical_features)

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=1, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(random_state=1)
    }

    results = []
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        scores = cross_val_score(pipeline, X, y, cv=5,
                                 scoring='neg_mean_squared_error')
        mse_scores = -scores
        result = {
            'Model': name,
            'Cross-Validation MSE Scores': ', '.join(
                f'{score:.2f}' for score in mse_scores),
            'Mean CV MSE': mse_scores.mean(),
            'Std CV MSE': mse_scores.std()
        }
        results.append(result)

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        f"{csv_dir}cross_val_model_evaluation_metrics.csv", index=False)
    print(
        f"\nCross-validation model evaluation metrics saved to {csv_dir}cross_val_model_evaluation_metrics.csv")

    best_model_name = results_df.loc[
        results_df['Mean CV MSE'].idxmin()]['Model']
    print(f"\nBest model based on CV: {best_model_name}")

    print("Cross Validation Results:\n", results_df)

    return results_df


def prepare_final_output(results):
    final_columns = [
        'Customer ID', 'Country',
        'Quantity', 'Price',
        'ClusterGroup', 'R_Score', 'F_Score', 'M_Score',
        'RFM_Level', 'Linear Regression_Predictions',
        'Gradient Boosting_Predictions',
        'Random Forest_Predictions'
    ]

    # rule_columns = [col for col in results.columns if col.startswith('Rule_')]
    # final_columns.extend(rule_columns)
    results['Quantity'] = results['Quantity'].round(2)
    final_data = results[final_columns]

    final_data.to_csv(f"{csv_dir}final_customer_insights.csv", index=False)
    print(
        f"Final customer insights saved to {csv_dir}final_customer_insights.csv")


def visualize_predictions(validation_data, predictions):
    """
    Visualize the comparison of total income from the past two years with
    predicted future spend.

    Args:
        validation_data (pd.DataFrame): The validation data with predictions.
        predictions (pd.DataFrame): The DataFrame containing predictions.

    Returns:
        None
    """

    # Combining predictions with validation data
    validation_data = validation_data.copy()
    validation_data['Random Forest_Predictions'] = predictions['Random Forest_Predictions']
    validation_data['Linear Regression_Predictions'] = predictions['Linear Regression_Predictions']
    validation_data['Gradient Boosting_Predictions'] = predictions['Gradient Boosting_Predictions']

    total_income_past = validation_data['Price'].sum()
    total_predicted_future_spend = validation_data['Random Forest_Predictions'].sum(
    )

    # Print the sums for confirmation
    print(f"Total Income in validation data: {total_income_past}")
    print(f"Total Predicted Future Spend: {total_predicted_future_spend}")

    # Creating a DataFrame for plotting
    comparison_data = pd.DataFrame({
        'Period': ['Past Two Years', 'Predicted Next Year'],
        'Total Spend': [total_income_past, total_predicted_future_spend]
    })

    # Check the basic statistics of the predictions
    print("Linear Regression:\n",
          validation_data['Linear Regression_Predictions'].describe())
    print("Random Forest:\n",
          validation_data['Random Forest_Predictions'].describe())
    print("Gradient Boosting:\n",
          validation_data['Gradient Boosting_Predictions'].describe())

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Period', y='Total Spend',
                data=comparison_data, ax=ax, color='darkblue')
    ax.set_title(
        'Comparison of Total Income: Past Two Years vs. Predicted Next Year')
    ax.set_ylabel('Total Spend')
    ax.set_xlabel('Period')

    # Save and show the plot
    fig.tight_layout()
    plt.savefig(f"{plot_dir}total_income_comparison.png")
    plt.show()


def cap_outliers(data, feature, lower_percentile=0.015, upper_percentile=0.98):
    """
    Cap outliers in a given feature based on specified percentiles.

    Args:
        data (pd.DataFrame): The input data.
        feature (str): The feature to cap outliers for.
        lower_percentile (float): The lower percentile for capping.
        upper_percentile (float): The upper percentile for capping.

    Returns:
        pd.DataFrame: The data with outliers capped.
    """
    lower_threshold = data[feature].quantile(lower_percentile)
    upper_threshold = data[feature].quantile(upper_percentile)

    data[feature] = np.where(
        data[feature] < lower_threshold, lower_threshold, data[feature])
    data[feature] = np.where(
        data[feature] > upper_threshold, upper_threshold, data[feature])

    print("Outliers capped!")

    return data


def main():
    """
    Main function to execute the data pipeline, including preprocessing,
    feature engineering, model training, and evaluation.

    Returns:
        None
    """
    # Load the data
    train_data = pd.read_csv(f"{csv_dir}train_data.csv")
    validation_data = pd.read_csv(f"{csv_dir}validation_data.csv")

    # Generating target column
    train_data['FutureSpend'] = train_data['Quantity'] * train_data['Price']
    train_data['FutureSpend'] = train_data['FutureSpend'].round(2)

    columns_to_cap = ['Quantity', 'R_Score', 'F_Score', 'M_Score', 'Price']
    # Capping outliers in train data
    for column in columns_to_cap:
        train_data = cap_outliers(train_data, column)

    # Capping outliers in validation data
    for column in columns_to_cap:
        validation_data = cap_outliers(validation_data, column)

    # Target feature not included here.
    feature_columns = [
        'RFM_Level', 'ClusterGroup', 'R_Score', 'F_Score',
        'M_Score', 'Quantity', 'Price',
        'Month', 'DayOfWeek', 'TimeOfDay'
    ]

    categorical_features = [
        'RFM_Level',
        'ClusterGroup',
    ]

    # Check data distribution
    check_data_distribution(
        train_data, feature_columns, categorical_features)

    # Uncomment below to run cross-validation performance check
    # cross_val_performance(
    #     transformed_training_data, feature_columns, categorical_features)

    # Run models with or without GridSearch
    # Set use_gridsearch to True to use it.
    run_models(
        train_data, feature_columns,
        categorical_features, use_gridsearch=True)

    # Predict future spending
    predictions = make_predictions(validation_data, feature_columns)

    # prepare_final_output(results)
    visualize_predictions(validation_data, predictions)


if __name__ == '__main__':
    main()
