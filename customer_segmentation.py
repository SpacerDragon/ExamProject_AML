# This script applies unsuperviced learning(clustering) to the dataset,
# and compares the results from the rfm-analysis from the data_analysis.py
# script. The goal of this script is to segment customers, with the end goal to
# more accurately apply correct marketing/cross-selling strategies to the
# different customer segments.
# Author: Per Idar Rød.

# Importing the neccesary libraries.
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

csv_dir = './csv_files'
os.makedirs(csv_dir, exist_ok=True)


def kMeans_clustering(data, clv=False):
    data = data.copy(deep=True)
    # Converting from string to datetime
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    # Adding some time elements for the clustering algo.
    data['Month'] = data['InvoiceDate'].dt.month
    data['DayOfWeek'] = data['InvoiceDate'].dt.dayofweek
    data['TimeOfDay'] = data['InvoiceDate'].dt.hour
    print("Starting clustering pipeline...")
    try:
        # Defining columns for different preprocessing.
        categorical_cols = ['Country']
        numeric_cols = ['Quantity', 'Price']
        time_cols = ['Month', 'DayOfWeek', 'TimeOfDay']

        # Preproccessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_cols),
                ('cat', OneHotEncoder(), categorical_cols),
            ], verbose=True
        )

        data_processed = preprocessor.fit_transform(
            data[numeric_cols + categorical_cols + time_cols])

        # Clustering
        kmeans = KMeans(n_clusters=3, random_state=1, verbose=1)
        kmeans.fit(data_processed)
        cluster_labels = kmeans.labels_
        data['ClusterGroup'] = cluster_labels

        # Evaluating clusters
        print("Starting Evaluation using silhouette.")
        silhouette = silhouette_score(
            data_processed,
            cluster_labels,
            sample_size=70000,
            random_state=1,
            n_jobs=-1
        )
        print("Silhouette Score:", silhouette)

        # Analyzing results
        print(data[['Customer ID', 'ClusterGroup']].head(10))

        cluster_summary = data.groupby('ClusterGroup').agg({
            'Quantity': ['mean', 'median'],
            'Price': ['mean', 'median'],
            'R_Score': 'mean',
            'F_Score': 'mean',
            'M_Score': 'mean'
        }).reset_index()

        print("\nCluster Summary:\n", cluster_summary)

        # Comparing the findings
        cross_tab = pd.crosstab(
            data['RFM_Level'], data['ClusterGroup'], margins=True)
        print("\nCross Tabulation:\n", cross_tab)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cross_tab.iloc[:-1, :-1],
                    annot=True, fmt="d", cmap="Blues")
        ax.set_title('Heatmap of RFM Level and ML Cluster Cross-Tabulation')
        ax.set_xlabel('ClusterGroup')
        ax.set_ylabel('RFM Level')
        fig.tight_layout()

        if not clv:
            fig.savefig(f'{csv_dir}/heatmap_rfm_clustergroups.png', dpi=300,
                        format='png', bbox_inches='tight')

        base_columns = ['InvoiceDate',
                        'Month',
                        'DayOfWeek',
                        'TimeOfDay',
                        'Invoice',
                        'Customer ID',
                        'Country',
                        'StockCode',
                        'StockCodeVariation',
                        'Description',
                        'Quantity',
                        'Price',
                        'R_Score',
                        'F_Score',
                        'M_Score',
                        'RFM_Level',
                        'ClusterGroup',
                        ]

        rule_columns = [
            col for col in data.columns if 'Rule_' in col and 'lift' in col]

        column_order = base_columns + sorted(rule_columns)
        if clv:
            clv_columns = column_order + ['FutureSpend']
            data = data[clv_columns]
            data.to_csv(f"{csv_dir}/clv_featured_clustered.csv", index=False)
        else:
            data = data[column_order]
            data.to_csv(f"{csv_dir}/data_featured_clustered.csv",
                        index=False)

        return data, silhouette
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None


def main():
    # Loading the datasets processed in 'data_analysis.py'
    data = pd.read_csv(f"{csv_dir}/data_featured.csv")
    feature_data = pd.read_csv(f"{csv_dir}/clv_data_featured.csv")

    # Output for visual confirmation.
    results, silhoutte = kMeans_clustering(data)
    if results is not None:
        print("\nClustering executed successfully.\n")
        print("Results:\n", results.head(10))
        print("Silhoutte:\n", silhoutte)
    else:
        print("Clustering failed.")

    clv_results, clv_silhoutte = kMeans_clustering(feature_data, clv=True)
    if clv_results is not None:
        print("\nClustering executed successfully.\n")
        print("Results:\n", clv_results.head(10))
        print("Silhoutte:\n", clv_silhoutte)
    else:
        print("Clustering failed.")

    plt.close('all')


if __name__ == "__main__":
    main()
