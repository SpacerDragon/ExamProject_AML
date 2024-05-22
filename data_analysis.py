# Online Retail Market Basket Analysis and Customer Segmentation, 2009-2011,UK
#
# Script to load, clean and combine data from 2009-2010 and 2010-2011,
# as well as removing missing entries, generating rules
# using the apriori method, grouping customers using the rfm(recency,
# frequency, monetary) method. Applying unsupervised clustering to the data.
# Creating visualizations showing the impact of the rules created.
# Finally exporting the enriched DataFrame as a new csv-file, ready
# for machine learning algorithms.

# Author: Per Idar Rød.
# post@peridar.net

# Importing necessary libraries
import os
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split


# Setting up paths
csv_dir = './csv_files/'
plot_dir = './plots/'
os.makedirs(csv_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)


def preprocess_data():
    """
    Load, clean, and preprocess transaction data from two CSV files.

    The function performs the following steps:
    1. Load data from two CSV files.
    2. Combine the data into a single DataFrame.
    3. Remove duplicates.
    4. Split letter codes from numeric codes in the 'StockCode' column.
    5. Remove specific entries and filter out negative quantities and prices.
    6. Handle missing values in 'Customer ID', 'StockCode', and 'Description'
    columns.
    7. Return the cleaned and preprocessed DataFrame.

    Returns:
        pd.DataFrame: The cleaned and preprocessed transaction data.
    """

    # Load data from CSV files
    print("Loading data.")
    try:
        data1 = pd.read_csv("2009-2010.csv", encoding='ISO-8859-1')
        data2 = pd.read_csv("2010-2011.csv", encoding='ISO-8859-1')
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        return None

    # Display shapes of the dataframes
    print("Data1 (2009-2010) shape:", data1.shape)
    print("Data2 (2010-2011) shape:", data2.shape)

    # Combine the dataframes
    data = pd.concat([data1, data2])
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    print("Combined data shape:", data.shape)

    # Remove duplicates
    data.drop_duplicates(keep='first', inplace=True)
    print(
        "Dropped duplicates. Combined data (no duplicates) shape:", data.shape)

    # Converting from string to datetime
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    # Create time features
    data['Month'] = data['InvoiceDate'].dt.month
    data['DayOfWeek'] = data['InvoiceDate'].dt.dayofweek
    data['TimeOfDay'] = data['InvoiceDate'].dt.hour

    # Add total spend column
    data['TotalSpend'] = data['Quantity'] * data['Price']

    # Display unique customers
    unique_customers = data['Customer ID'].nunique()
    print("Unique customers:", unique_customers)

    # Split letter codes from numbers in 'StockCode' column
    data['StockCodeVariation'] = data['StockCode'].str.extract(
        '([A-Za-z]+)', expand=False)
    data['StockCode'] = data['StockCode'].str.extract(r'(\d+)', expand=False)

    # Extract letter from 'Invoice' column and drop it
    data['CancelledOrders'] = data['Invoice'].str.extract(
        '([A-Za-z]+)', expand=False)
    data.drop('CancelledOrders', axis=1, inplace=True)
    data['Invoice'] = data['Invoice'].str.extract(r'(\d+)', expand=False)

    # Remove specific entries and filter out negative quantities and prices
    data = data[~data['Description'].isin(['Manual', 'POSTAGE'])]
    data = data[~data['StockCodeVariation'].isin(
        ['ADJUST', 'D', 'DOT', 'CRUK'])]
    data = data[data['Quantity'] > 0]
    data = data[data['Price'] > 0]

    # Handle missing 'Customer ID' values
    data.dropna(subset=['Customer ID'], inplace=True)
    data['Customer ID'] = data['Customer ID'].astype(int)

    # Handle missing 'StockCode' values
    data['StockCode'] = data['StockCode'].fillna(0).astype(int)

    # Handle missing 'Description' values
    description_dict = data.dropna(
        subset=['Description']).set_index('StockCode')[
        'Description'].to_dict()
    data['Description'] = data.apply(
        lambda row: description_dict[row['StockCode']] if pd.isnull(
            row['Description']) and row['StockCode'] in
        description_dict else row['Description'],
        axis=1
    )
    data.dropna(subset=['Description'], inplace=True)

    # Fill missing 'StockCodeVariation' with 'None'
    data['StockCodeVariation'] = data['StockCodeVariation'].fillna('None')

    # Display the shape of the cleaned data
    print("Shape of cleaned data:", data.shape)

    # Display missing entries in the combined dataset
    missing_entries = data.isnull().sum()
    print("Missing Entries in combined dataset:\n", missing_entries)

    # Set the column order for the DataFrame
    column_order = [
        'Customer ID', 'Country',
        'InvoiceDate', 'Invoice',
        'StockCode', 'StockCodeVariation', 'Description',
        'Month', 'DayOfWeek', 'TimeOfDay',
        'Quantity', 'Price', 'TotalSpend'
    ]

    data = data[column_order]

    # Display sample of cleaned data
    print("Cleaned data sample:\n", data.head().to_string(index=False))

    return data


def rfm_analysis(data):
    """
    Perform RFM analysis on the data and return the data with RFM features.

    Args:
        data (pd.DataFrame): The input data for RFM analysis.

    Returns:
        pd.DataFrame: The data with additional RFM features.
    """

    # Adding one day to the end of the 'InvoiceDate' to avoid getting 0s.
    current_date = data['InvoiceDate'].max() + pd.Timedelta(days=1)

    # Grouping data and aggregating new columns
    rfm = data.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (current_date - x.max()).days,
        'Invoice': 'count',
        'Price': lambda x: x.sum()
    }).rename(columns={'InvoiceDate': 'R_Score',
                       'Invoice': 'F_Score',
                       'Price': 'M_Score'
                       })

    # Define levels based on raw scores
    def rfm_level(df):
        if df['R_Score'] <= rfm['R_Score'].quantile(0.33) and \
                df['F_Score'] > rfm['F_Score'].quantile(0.66) and \
                df['M_Score'] > rfm['M_Score'].quantile(0.66):
            return 'High Value'
        elif df['R_Score'] > rfm['R_Score'].quantile(0.33) and \
                df['R_Score'] <= rfm['R_Score'].quantile(0.66) and \
                df['F_Score'] > rfm['F_Score'].quantile(0.33) and \
                df['M_Score'] > rfm['M_Score'].quantile(0.33):
            return 'Medium Value'
        else:
            return 'Low Value'

    # Setting labels on every entry
    rfm['RFM_Level'] = rfm.apply(rfm_level, axis=1)

    # Merging the data frames
    data = data.merge(rfm[['R_Score', 'F_Score', 'M_Score',
                      'RFM_Level']], on='Customer ID', how='left')

    print("\nCustomer Segmentation results:\n", data.shape)
    print(data.head())

    return data


def market_basket_analysis(data):
    """
    Perform market basket analysis on the given data and
    generate association rules.

    Args:
        data (pd.DataFrame): The input transactional data.

    Returns:
        pd.DataFrame: The data with additional features
        based on the top association rules.
    """

    # Converting grouped items to a list of lists,
    # where each list contains items from a single invoice.
    transactions = data.groupby('Invoice')['Description'].apply(list).tolist()

    # Transforming DataFrame using the TransactionEncoder, a one-hot encoder.
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    # Converting back to pandas DataFrame
    df_transactions = pd.DataFrame(te_array, columns=te.columns_)

    # Running the apriori algorithm.
    print("\nApplying Apriori algorithm!")
    frequent_items = apriori(df_transactions, min_support=0.015,
                             use_colnames=True, verbose=True)

    # Generating the rules on the dataset.
    print("\nGenerating rules...\n")
    rules = association_rules(
        frequent_items, metric='confidence', min_threshold=0.6)

    # Cleaning up the rules DataFrame
    rules['antecedents'] = rules['antecedents'].apply(
        lambda x: ', '.join(x))
    rules['consequents'] = rules['consequents'].apply(
        lambda x: ', '.join(x))
    rules['support'] = rules['support'].apply(lambda x: round(x, 5))
    rules['confidence'] = rules['confidence'].apply(lambda x: round(x, 5))
    rules['lift'] = rules['lift'].apply(lambda x: round(x, 5))

    # Selecting columns for output.
    focused_rules = rules[['antecedents',
                           'consequents',
                           'support',
                           'confidence',
                           'lift'
                           ]]

    # Sorting Dataframes for displaying/saving to csv.
    confidence_sorted = focused_rules.sort_values(
        by='confidence', ascending=False)

    try:
        print("Saving rules to file!")
        confidence_sorted.to_csv(f'{csv_dir}confidence_rules.csv')
        print("Files Saved!")
    except Exception as e:
        print(f"Failed to save the file: {e}")
        exit(1)

    # Filter the top rules based on lift from top confidence rules.
    top_confidence_rules = confidence_sorted.head(20)
    top_rules = top_confidence_rules[
        top_confidence_rules['lift'] > 1.0].head(10)

    # Aggregating descriptions by invoice into a list of items
    data['Items'] = data.groupby(
        'Invoice')['Description'].transform(lambda x: ', '.join(set(x)))

    # Function to set binary value based on if the customer is part of
    # any of the top 10 rules created by the apriori analysis.
    def apply_rule(items, antecedents, consequents):
        items_set = set(items.split(", "))
        antecedents_set = set(antecedents.split(", "))
        consequents_set = set(consequents.split(", "))
        return 1 if antecedents_set.issubset(items_set) \
            and consequents_set.issubset(items_set) else 0

    # Applying top rules based on confidence
    for index, rule in top_rules.iterrows():
        antecedents = rule['antecedents']
        consequents = rule['consequents']
        column_name = f'Rule_{index}_lift'
        # Apply rule
        data[column_name] = data['Items'].apply(
            lambda items: apply_rule(items, antecedents, consequents)
        )

    for col in data.columns:
        if col.startswith('Rule_'):
            print(f'Found matches:{col}: {data[col].sum()}')

    print("\nDataFrame in market basket analysis:\n", data.head())
    print("Shape of data:\n", data.shape)

    # Dropping the 'Items' column as it clutters up the dataframe.
    data.drop(columns='Items', inplace=True)

    # Making a new column containing both antecedent and consequents,
    # primarily for the visualizations.
    confidence_sorted['itemsets'] = confidence_sorted['antecedents'] + \
        ' +\n ' + confidence_sorted['consequents']

    # Plotting scatterplot
    plt.figure(figsize=(10, 8))
    plt.scatter(top_rules['support'], top_rules['confidence'],
                alpha=0.5, c=top_rules['lift'], cmap='viridis', s=100)
    plt.colorbar(label='Lift')
    plt.title('Association Rules')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.grid(True)
    plt.savefig(f'{plot_dir}association_rules.png')

    # Setting the lower xlim to 0.6 because the threshold in the
    # association rules was set to this.
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    sns.barplot(x='confidence', y='itemsets',
                data=confidence_sorted.head(10), color='darkblue')
    ax1.set_xlim(0.6, 1.0)
    ax1.set_title('Top 10 Rules for Confidence')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Itemsets (Antecedent + Consequent)')
    ax1.xaxis.set_major_locator(MaxNLocator(10))
    fig1.tight_layout()
    fig1.savefig(f'{plot_dir}top_itemsets.png', dpi=300,
                 format='png', bbox_inches='tight')

    plt.close('all')

    return data


def aggregate_data(data):
    """
    Aggregate the data by 'Customer ID', compute relevant features,
    and set the columns.

    Args:
        data (pd.DataFrame): The input data to be aggregated.

    Returns:
        pd.DataFrame: The aggregated data with relevant columns.
    """

    # Identify all rule columns
    rule_columns = [col for col in data.columns if col.startswith('Rule_')]

    # Define aggregation methods
    aggregation_methods = {
        'Country': 'first',
        'Quantity': 'sum',
        'TotalSpend': 'sum',
        'R_Score': 'mean',
        'F_Score': 'mean',
        'M_Score': 'mean',
        'RFM_Level': 'first',
        'Month': 'first',
        'DayOfWeek': 'first',
        'TimeOfDay': 'first'
    }

    # Add rule columns to the aggregation methods
    # Using 'max' to see if any transactions matches
    # the rule.
    for rule in rule_columns:
        aggregation_methods[rule] = 'max'

    # Include any other columns that should be retained
    other_columns = [
        col for col in data.columns if col not in aggregation_methods and col not in rule_columns and col != 'Customer ID']
    for col in other_columns:
        aggregation_methods[col] = 'first'

    # Aggregating data
    aggregated_data = data.groupby(
        'Customer ID').agg(aggregation_methods).reset_index()

    # Create a new column to indicate if any rules are connected to customer
    # aggregated_data['Has_Rule'] = aggregated_data[rule_columns].max(axis=1)

    print("Aggregated data shape:\n", aggregated_data.shape)

    print("Agg data:\n", aggregated_data.head())
    print("Agg data shape:\n", aggregated_data.shape)
    # aggregated_data.to_csv("agg_data.csv", index=False)
    return aggregated_data


def kMeans_clustering(data):
    """
    Perform KMeans clustering on the aggregated data.

    Args:
        data (pd.DataFrame): The aggregated data with relevant columns.

    Returns:
        pd.DataFrame: The data with cluster labels.
    """

    data = data.copy(deep=True)

    # Setting order in rfm_level to get the heatmap correct
    rfm_order = ['High Value', 'Medium Value', 'Low Value']
    data['RFM_Level'] = pd.Categorical(
        data['RFM_Level'], categories=rfm_order, ordered=True)

    print("Starting clustering pipeline...")
    try:
        # Defining columns for different preprocessing.
        categorical_cols = ['Country']
        numeric_cols = ['Quantity', 'Price', 'Month', 'DayOfWeek',
                        'TimeOfDay', 'R_Score', 'F_Score', 'M_Score',
                        'TotalSpend']
        # rule_cols = [col for col in data.columns if 'Rule_' in col]

        # Preproccessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_cols),
                ('cat', OneHotEncoder(), categorical_cols),
            ], verbose=True
        )

        data_processed = preprocessor.fit_transform(
            data[numeric_cols + categorical_cols])

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

        # Including all numeric columns in aggregation
        numeric_agg_cols = numeric_cols
        cluster_summary = data.groupby('ClusterGroup')[numeric_agg_cols].agg([
            'mean', 'median']).reset_index()

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

        fig.savefig(f'{plot_dir}heatmap_rfm_clustergroups.png', dpi=300,
                    format='png', bbox_inches='tight')

        plt.close('all')

        column_order = [
            'InvoiceDate',
            'Customer ID',
            'Country',
            'Month',
            'DayOfWeek',
            'TimeOfDay',
            'Quantity',
            'Price',
            'TotalSpend',
            'R_Score',
            'F_Score',
            'M_Score',
            'RFM_Level',
            'ClusterGroup',
        ]

        # rule_columns = [
        # col for col in data.columns if 'Rule_' in col]

        # column_order = base_columns + sorted(rule_columns)
        data = data[column_order]

        # Round price column to 2 decimals
        data['Price'] = data['Price'].round(2)
        data['M_Score'] = data['M_Score'].round(2)

        print("Results:\n", data.head())
        print("Silhoutte:\n", silhouette)
        print("\nClustering executed successfully.\n")

        data.to_csv(f"{csv_dir}clustered_data.csv", index=False)

        return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def split_data(data):
    """
    Split the data into training and validation sets.

    Args:
        data (pd.DataFrame): The input data.

    Returns:
        None
    """
    # train_data, validation_data = train_test_split(
    #     data, test_size=0.2, random_state=1)

    cutoff_date = pd.to_datetime('2010-12-01')
    train_data = data[data['InvoiceDate'] < cutoff_date]
    validation_data = data[data['InvoiceDate'] >= cutoff_date]

    # train_data.to_csv(f"{csv_dir}train_data.csv", index=False)
    # validation_data.to_csv(f"{csv_dir}validation_data.csv", index=False)

    # print(f"Training data and validation data saved to {csv_dir}")

    return train_data, validation_data


def main():
    """
    Execute the main workflow for data preprocessing,
    RFM analysis, market basket analysis,
    data aggregation, and clustering.

    This function performs the following steps:
    1. Preprocess the data.
    2. Perform RFM analysis on the preprocessed data.
    3. Apply market basket analysis to the RFM data.
    4. Aggregate the data to create customer-level features.
    5. Perform K-Means clustering on the aggregated data.

    Returns:
        None
    """

    data = preprocess_data()

    # Generate Recency, Frequency and Monetary values
    data_rfm = rfm_analysis(data)

    # Generat apriori rules
    data_rfm_and_apriori_rules = market_basket_analysis(
        data_rfm)

    # Clustering
    clustering_results = kMeans_clustering(data_rfm_and_apriori_rules)

    # Split the data
    train_data, validation_data = split_data(data_rfm_and_apriori_rules)

    # Aggregate the data
    train_agg_data = aggregate_data(clustering_results)
    validation_agg_data = aggregate_data(clustering_results)

    numeric_cols_train = train_agg_data.select_dtypes(
        include=['number']).columns
    train_agg_data.loc[:, numeric_cols_train] = train_agg_data[numeric_cols_train].round(
        2)

    numeric_cols_val = validation_agg_data.select_dtypes(
        include=['number']).columns
    validation_agg_data.loc[:, numeric_cols_val] = validation_agg_data[numeric_cols_val].round(
        2)

    # Save aggregated data
    train_agg_data.to_csv(f'{csv_dir}train_data.csv', index=False)
    validation_agg_data.to_csv(f'{csv_dir}validation_data.csv', index=False)


if __name__ == "__main__":
    main()
