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
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (train_test_split,
                                     cross_val_score,
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

# Custom transformer for log transformation


class LogTransformer:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log1p(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def preprocess_data(data, feature_columns, categorical_features):
    """Function for preprocessing of data.
    Returns preprocessor
    """
    numerical_features = [
        col for col in feature_columns if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('log', LogTransformer()),
                ('scaler', StandardScaler())
            ]), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])

    return preprocessor


def check_data_distribution(data, feature_columns, categorical_features):
    print("\nData distribution - Summary Statistics:\n",
          data[feature_columns].describe())

    numerical_features = [
        col for col in feature_columns if col not in categorical_features]

    # Histogram with KDE
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    for i, col in enumerate(numerical_features):
        row, col_idx = divmod(i, 3)
        sns.histplot(data[col], bins=30, kde=True, ax=axs[row, col_idx])
        axs[row, col_idx].set_title(f'Distribution of {col}')
    fig.tight_layout()
    fig.savefig(f"{plot_dir}Histogram_with_KDE.png")
    print("\nSaved Histogram to ./plots")
    plt.close(fig)

    # Box-and-whisker plots
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    for i, col in enumerate(numerical_features):
        row, col_idx = divmod(i, 3)
        sns.boxplot(y=data[col], ax=axs[row, col_idx])
        axs[row, col_idx].set_title(f'Box plot of {col}')
    fig.tight_layout()
    fig.savefig(f"{plot_dir}Box_and_whisker.png")
    print("\nSaved Box-and-whisker plot to ./plots")
    plt.close(fig)


def run_models(
        data,
        feature_columns,
        categorical_features,
        use_gridsearch=True):
    X = data[feature_columns]
    y = data['FutureSpend']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1)

    preprocessor = preprocess_data(data, feature_columns, categorical_features)

    # Model setup
    models = {
        'Linear Regression': LinearRegression(n_jobs=-1),
        'Random Forest': RandomForestRegressor(
            random_state=1,
            verbose=1,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            random_state=1,
            verbose=1,
        )}
    # Setting parameter grid for each model
    param_grids = {
        'Random Forest': {
            'model__n_estimators': [100, 200],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_split': [2, 5]
        },
        'Gradient Boosting': {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_depth': [3, 5, 7]
        }
    }

    results = []
    # Pipeline setup and running models
    for name, model in models.items():
        model_path = f'{model_dir}{name}.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                pipeline = pickle.load(f)
            print(f"Loaded {name} from pickle.\n")
        else:
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            if use_gridsearch and name in param_grids:
                grid_search = GridSearchCV(
                    pipeline, param_grids[name],
                    cv=5, scoring='neg_mean_squared_error',
                    verbose=1)

                grid_search.fit(X_train, y_train)
                pipeline = grid_search.best_estimator_
                print(
                    f"Best parameters for {name}: {grid_search.best_params_}")
            else:
                pipeline.fit(X_train, y_train)

            with open(model_path, 'wb') as f:
                pickle.dump(pipeline, f)
            print(f"Saved {name} model to pickle.")

        # Predictions and evaluations
        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
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
    model_names = ['Linear Regression', 'Random Forest', 'Gradient Boosting']
    predictions = {}

    for name in model_names:
        model_path = f'{model_dir}{name}.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                pipeline = pickle.load(f)
            print(f"Loaded {name} from pickle.\n")

            X = data[feature_columns]
            y_pred = pipeline.predict(X).round(2)
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
            'Cross-Validation MSE Scores': ', '.join(f'{score:.2f}' for score in mse_scores),
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


def prepare_final_output(results):
    final_columns = [
        'InvoiceDate', 'Invoice', 'Customer ID', 'Country',
        'StockCode', 'Description', 'Quantity', 'Price',
        'ClusterGroup', 'R_Score', 'F_Score', 'M_Score',
        'RFM_Level', 'Linear Regression_Predictions',
        'Gradient Boosting_Predictions',
        'Random Forest_Predictions'
    ]

    rule_columns = [col for col in results.columns if col.startswith('Rule_')]
    final_columns.extend(rule_columns)
    final_data = results[final_columns]

    final_data.to_csv(f"{csv_dir}final_customer_insights.csv", index=False)
    print(
        f"Final customer insights saved to {csv_dir}final_customer_insights.csv")


def visualize_predictions(data):
    total_income_past = data['Price'].sum()
    total_predicted_future_spend = data['Random Forest_Predictions'].sum()
    # Print the sums for confirmation
    print(f"Total Income for Past Two Years: {total_income_past}")
    print(f"Total Predicted Future Spend: {total_predicted_future_spend}")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Creating a DataFrame for plotting
    comparison_data = pd.DataFrame({
        'Period': ['Past Two Years', 'Predicted Next Year'],
        'Total Spend': [total_income_past, total_predicted_future_spend]
    })

    # Check the basic statistics of the predictions
    print("Random Forest:\n", data['Random Forest_Predictions'].describe())

    # # Set the cap value to the 99th percentile
    # cap_value = data['Random Forest_Predictions'].quantile(0.99)
    # data['Random Forest_Predictions_Capped'] = np.clip(
    #     data['Random Forest_Predictions'], 0, cap_value)
    #
    # # Check the new summary statistics after capping
    # print("Capped:\n", data['Random Forest_Predictions_Capped'].describe())
    #
    # # Identify predictions that are significantly higher than the rest
    # outliers = data[data['Random Forest_Predictions'] >
    #                 data['Random Forest_Predictions'].quantile(0.99)]
    # print("Outliers:\n", outliers[['Random Forest_Predictions']].describe())

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


def transform_features(data):
    # Log transformation for Quantity and Price to handle skewness
    data['Log_Quantity'] = np.log1p(data['Quantity'])
    data['Log_Price'] = np.log1p(data['Price'])

    # Binning the TimeOfDay into categories (e.g., morning, afternoon, evening)
    bins = [0, 6, 12, 18, 24]
    labels = ['Night', 'Morning', 'Afternoon', 'Evening']
    data['TimeOfDay_Binned'] = pd.cut(
        data['TimeOfDay'], bins=bins, labels=labels, right=False)

    return data


def main():
    training_data = pd.read_csv(f"{csv_dir}training_data.csv")
    complete_data = pd.read_csv(f"{csv_dir}data.csv")

    # Apply transformations
    transformed_training_data = transform_features(training_data)
    transformed_complete_data = transform_features(complete_data)

    feature_columns = [
        'RFM_Level', 'ClusterGroup', 'R_Score', 'F_Score',
        'M_Score', 'Log_Quantity', 'Log_Price', 'Month', 'DayOfWeek', 'TimeOfDay_Binned'
    ]

    categorical_features = [
        'RFM_Level',
        'ClusterGroup',
        'TimeOfDay_Binned'
    ]

    # Uncomment below to run cross-validation performance check
    # cross_val_performance(
    #     transformed_training_data, feature_columns, categorical_features)

    # Run models with or without GridSearch
    # Set use_gridsearch to True to use it.
    run_models(
        transformed_training_data, feature_columns,
        categorical_features, use_gridsearch=False)

    # Check data distribution
    check_data_distribution(
        transformed_complete_data, feature_columns, categorical_features)

    # Predict future spending
    results = make_predictions(transformed_complete_data, feature_columns)

    prepare_final_output(results)
    visualize_predictions(results)


if __name__ == '__main__':
    main()
