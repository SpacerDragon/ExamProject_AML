# This scripts purpose is to predict customer lifetime value of 5 years,
# and creating visualizations of data distrisbution and feature
# importances, and preparing the final outputs:
# customer_insigths.csv
# actual_vs_predicted_clv.png
#
# Author: Per Idar Rød.
# post@peridar.net


import os
import pickle
import math
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import (StandardScaler, OneHotEncoder)
from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV)


# Setting paths
csv_dir = './csv_files/'
model_dir = './models/'
plot_dir = './plots/'
os.makedirs(csv_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)


def check_data_distribution(data, feature_columns, categorical_features):
    """
    Check the distribution of the data by plotting histograms
    and box plots for numerical features.

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
        col for col in feature_columns if col not in
        categorical_features and not col.startswith('Rule_')]

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
    """
    Plot and save a bar chart of feature importances.

    Args:
        importances (dict): A dictionary where keys are
            feature names and values are their importances.
        title (str): The title for the plot, which will also
            be used as the filename for saving the plot.

    Returns:
        None
    """
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
            ('scaler', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ], remainder='passthrough'
    )

    return preprocessor


def run_models(
        data,
        feature_columns,
        categorical_features,
        use_randomsearch=True,
        best_model=None):
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
        best_model
    """

    # Split the data
    X = data[feature_columns]
    y = data['Actual_CLV'].values.reshape(-1, 1)

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
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            random_state=1,
        )}

    if best_model:
        models = {best_model: models[best_model]}

    # Hyperparameter grids
    param_grids = {
        'Random Forest': {
            'model__n_estimators': [200, 250, 300],
            'model__min_samples_split': [2, 3, 4],
            'model__max_depth': [5, 10, 15],
        },
        'Gradient Boosting': {
            'model__n_estimators': [150, 200, 250, 300, 350],
            'model__min_samples_split': [2, 3, 4, 5],
            'model__max_depth': [4, 5, 6],
            'model__learning_rate': [0.04, 0.05, 0.09, 0.1, 0.11],

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
            preprocessor.fit(X_train)
            X_train_transformed = preprocessor.transform(X_train)

            # Get feature names after preprocessing
            num_feature_names = preprocessor.transformers_[0][2]
            cat_feature_names = preprocessor.transformers_[
                1][1].get_feature_names_out(categorical_features).tolist()
            transformed_feature_names = num_feature_names + cat_feature_names

            # Define the RFE(Recursive Feature Elimination) step
            rfe = RFE(estimator=model, n_features_to_select=10, step=1)
            rfe.fit(X_train_transformed, y_train_scaled.ravel())

            # Print the 10 best features selected by RFE
            selected_features = [feature for feature, support in zip(
                transformed_feature_names, rfe.support_) if support]
            print(
                f"Top 10 features selected by RFE for {name}: {selected_features}")
            selected_features = pd.DataFrame(selected_features)
            selected_features.to_csv(
                f"{csv_dir}{name}_rfe_selected_features.csv", index=False)
            print(f"Saved top 10 features selected by RFE for {name} to csv!")

            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('rfe', rfe),
                ('model', model)
            ])

            if use_randomsearch and name in param_grids:
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
                    f"Best params for {name}: {random_search.best_params_}")

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
        mae = mean_absolute_error(y_test_original, y_pred)
        r2 = r2_score(y_test_original, y_pred)
        result = {'Model': name,
                  'MSE': mse.round(2),
                  'R-squared': r2.round(2),
                  'MAE': mae.round(2)
                  }

        # Feature importances for applicable models
        if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
            feature_importances = pipeline.named_steps['model']\
                .feature_importances_
            feature_names = [col for col in feature_columns
                             if col not in categorical_features] + \
                list(pipeline.named_steps['preprocessor'].named_transformers_[
                    'cat'].get_feature_names_out())
            result['Feature Importances'] = dict(
                zip(feature_names, feature_importances.round(5)))

        results.append(result)

    training_results_df = pd.DataFrame(results)
    columns_to_print = ['Model', 'MSE', 'R-squared', 'MAE']
    print("\nTest results DF:\n", training_results_df[columns_to_print])
    training_results_df.to_csv(
        f"{csv_dir}model_evaluation_metrics.csv", index=False)
    print(
        f"\nModel eval metrics saved to {csv_dir}model_evaluation_metrics.csv")

    # Identify the best model based on Test MSE
    best_model_row = training_results_df.loc[training_results_df['MSE'].idxmin(
    )]
    best_model = best_model_row['Model']
    print(f"Best model based on Test MSE: {best_model}")

    for index, row in training_results_df.iterrows():
        if row['Model'] in ['Random Forest', 'Gradient Boosting']:
            plot_feature_importances(
                row['Feature Importances'],
                f"{row['Model']} Feature Importances")

    return training_results_df, best_model


def make_predictions(data, feature_columns, best_model):
    """
    Make predictions using the pre-trained best model and inverse
    transform the scaled predictions.

    Args:
        data (pd.DataFrame): The input data.
        feature_columns (list): List of feature columns to be used
        for making predictions.
        best_model_name (str): The name of the best model to be
        used for predictions.

    Returns:
        pd.DataFrame: DataFrame with predictions added.
    """

    model_path = f'{model_dir}{best_model}.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            pipeline, y_scaler = pickle.load(f)
        print(f"Loaded {best_model} and scaler from pickle.\n")

        X = data[feature_columns]
        y_pred_scaled = pipeline.predict(X).reshape(-1, 1)
        y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()

        # Clip negative predictions
        y_pred = [max(0, pred) for pred in y_pred]

        data["CLV_Predictions"] = y_pred

    return data


def prepare_final_output(results):
    """
    Main function to execute the data pipeline, including preprocessing,
    feature engineering and model training.

    Returns:
        None
    """
    final_columns = [
        'Customer ID', 'Country',
        'R_Score', 'F_Score', 'M_Score',
        'RFM_Level', 'Has_Rule', 'Quantity', 'Price',  'Actual_CLV'
    ]
    prediction_column = [
        col for col in results.columns if '_Predictions' in col]
    final_columns.extend(prediction_column)

    # Select the final columns
    final_data = results[final_columns].copy(deep=True)
    final_data[prediction_column] = final_data[prediction_column].round(2)
    final_data['Actual_CLV'] = final_data['Actual_CLV'].round(2)

    final_data.to_csv(f"{csv_dir}customer_insights_CLV.csv", index=False)
    print(
        f"Final customer insights saved to {csv_dir}final_customer_insights.csv")


def visualize_predictions(predictions):
    """
    Plot the top N customers by predicted CLV.

    Args:
        predictions (pd.DataFrame): The DataFrame containing predictions.
        top_n (int): The number of top customers to display.

    Returns:
        None
    """

    # Display top 10 customers.
    top_n = 10
    top_customers = predictions.nlargest(
        top_n, 'CLV_Predictions').sort_values(
        by='CLV_Predictions', ascending=False)
    plt.figure(figsize=(10, 6))

    sns.barplot(x=top_customers['Customer ID'].astype(
        str), y=top_customers['CLV_Predictions'],
        palette='viridis', hue=top_customers['Customer ID'].astype(
        str), legend=False)
    plt.title(f'Top {top_n} Customers by Predicted CLV')
    plt.xlabel('Customer ID')
    plt.ylabel('Predicted CLV')
    plt.savefig(f"{plot_dir}top_customers_by_clv.png")

    # Scatter plot for actual vs. predicted CLV for all customers,
    # validation data.
    plt.figure(figsize=(14, 7))
    sns.scatterplot(
        x=predictions['Actual_CLV'], y=predictions['CLV_Predictions'], label='Predictions')
    plt.plot([predictions['Actual_CLV'].min(), predictions['Actual_CLV'].max()],
             [predictions['Actual_CLV'].min(), predictions['Actual_CLV'].max()],
             'r--', label='Perfect Prediction Line')  # Line for perfect prediction
    plt.title('Actual CLV vs Predicted CLV')
    plt.xlabel('Actual CLV')
    plt.ylabel('Predicted CLV')
    plt.legend()
    plt.savefig(f"{plot_dir}actual_vs_predicted_clv.png")

    # # Residual plot
    # plt.figure(figsize=(14, 7))
    # sns.scatterplot(x=predictions['Actual_CLV'],
    #                 y=predictions['Actual_CLV'] - predictions['CLV_Predictions'])
    # plt.axhline(0, color='red', linestyle='--')
    # plt.title('Residuals: Actual CLV - Predicted CLV')
    # plt.xlabel('Actual CLV')
    # plt.ylabel('Residuals (Actual CLV - Predicted CLV)')
    # plt.savefig(f"{plot_dir}residuals_clv.png")

    plt.show()


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

    # Add actual_clv column (target label)
    train_data['Actual_CLV'] = train_data['TotalSpend'] * 5

    # Perform log-transformations
    train_data['Price_log'] = np.log1p(train_data['Price'])
    train_data['Quantity_log'] = np.log1p(train_data['Quantity'])

    validation_data['Price_log'] = np.log1p(validation_data['Price'])
    validation_data['Quantity_log'] = np.log1p(validation_data['Quantity'])

    # Target feature not included here.
    base_columns = [
        'RFM_Level', 'R_Score', 'F_Score',
        'M_Score', 'Price_log', 'Quantity_log',
        'Month', 'DayOfWeek', 'TimeOfDay',
        'Has_Rule'
    ]

    categorical_features = [
        'RFM_Level',
    ]

    rule_columns = [col for col in train_data.columns if 'Rule_' in col]

    feature_columns = base_columns + rule_columns

    # Reset the index to avoid ambiguity
    train_data.reset_index(drop=True, inplace=True)

    # Check data distribution
    check_data_distribution(
        validation_data, feature_columns, categorical_features)
    # check_data_distribution(
    #     validation_data, feature_columns, categorical_features)

    # Initialize model manually, None selects the best model automatically.
    # Uncomment the model wanted to run only that.
    model = None
    # model = 'Linear Regression'
    # model = 'Random Forest'
    # model = 'Gradient Boosting'

    # Run models with or without RandomSearchCV.
    # RandomSearchCV has an integral Cross-Validation.
    # Set use_randomsearch to True to use it.
    training_results_df, model = run_models(
        train_data, feature_columns, categorical_features,
        use_randomsearch=True, best_model=model)

    # Predict Customer Lifetime Value
    predictions = make_predictions(
        validation_data, feature_columns, model)

    validation_data['Actual_CLV'] = validation_data['TotalSpend'] * 5

    # Displaying the predictions, and save plots
    visualize_predictions(predictions)

    # Preparing final output and save results to csv file.
    prepare_final_output(predictions)


if __name__ == '__main__':
    main()
