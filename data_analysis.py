# Online Retail Market Basket Analysis and Customer Segmentation, 2009-2011,UK
#
# Script to load, clean and combine data from 2009-2010 and 2010-2011,
# as well as removing missing entries, generating rules
# using the apriori method, grouping customers using the rfm(recency,
# frequency, monetary) method. Creating visualizations showing the impact of
# the rules created. Exporting the enriched DataFrame as a new csv-file, ready
# for machine learning algorithms.
# Author: Per Idar Rød.

# Importing necessary libraries
import os
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

csv_dir = './csv_files'
os.makedirs(csv_dir, exist_ok=True)


# Pre-processing
def preprocess_data():

    # Importing from the two different csv files
    # Using ISO-8859-1 for handling pound sign
    print("Loading data.")
    try:
        data1 = pd.read_csv("2009-2010.csv", encoding='ISO-8859-1')
        data2 = pd.read_csv("2010-2011.csv", encoding='ISO-8859-1')
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")

    # Displaying the shapes of the dataframes
    print("Data1: 2009-2010.shape:\n", data1.shape)
    print("Data2: 2010-2011.shape:\n", data2.shape)

    # Concatenating the two original dataframes for easier handling.
    data = pd.concat([data1, data2])
    # Converting to datetime object
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    # Displaying shape of combined dataframe
    print("Combined data.shape:\n", data.shape)

    # Removing duplicates in the new dataframe
    data.drop_duplicates(keep='first', inplace=True)
    print("Dropping duplicates.\n")

    # Confirming drop duplicate operation
    print("Combined data, no duplicates shape:\n", data.shape)

    # Splitting letter codes from numbers in columns
    # Extracting the letters from the StockCode column
    data['StockCodeVariation'] = data['StockCode'].str.extract(
        '([A-Za-z]+)', expand=False)
    # Removing the letters from 'StockCode' column
    data['StockCode'] = data['StockCode'].str.extract(r'(\d+)', expand=False)

    # Extracting the letter from Invoice column and put them in new column
    data['CancelledOrders'] = data['Invoice'].str.extract(
        '([A-Za-z]+)', expand=False)
    # Setting 'None' in all missing fields in 'CancelledOrder' column
    # data['CancelledOrders'] = data['CancelledOrders'].fillna('None')
    # Removing rows containing CancelledOrders
    # cancelled_orders = len(data[data['CancelledOrders'] == 'C'])
    # print(f"Cancelled Orders: {cancelled_orders}")
    # data = data[data['CancelledOrders'] != 'C']
    # Dropping column 'CancelledOrders' as it now contains only 'None'
    data.drop('CancelledOrders', axis=1, inplace=True)

    # Removing the letters from Invoice column
    data['Invoice'] = data['Invoice'].str.extract(r'(\d+)', expand=False)

    # I found out when running the kMeans clustering that one cluster
    # held 'Manual' and 'POSTAGE' in the description column
    # Removing those rows.
    # There also were made manual adjustments in the data,
    # as well as discounts.
    # It was found while running clustering.
    # Removing these as well
    # Other clusters held 'Dotcom postage' and 'Cruk commission'.
    # Next issue were large negative quantitys and price.
    # Removing these as well, as the goal is to identify purchasing
    # behaviour. Returns is not in the scope of my objectives.
    data = data[data['Description'] != 'Manual']
    data = data[data['Description'] != 'POSTAGE']
    data = data[data['StockCodeVariation'] != 'ADJUST']
    data = data[data['StockCodeVariation'] != 'D']
    data = data[data['StockCodeVariation'] != 'DOT']
    data = data[data['StockCodeVariation'] != 'CRUK']
    data = data[data['Quantity'] > 0]
    data = data[data['Price'] > 0]

    # Handling missing 'Customer ID' data.
    # Removing the rows with missing 'Customer ID', we lose quite many
    # rows doing this, but the dataset is still pretty large.
    # Considering not removing if ML-metrics is bad.
    data.dropna(subset=['Customer ID'], inplace=True)
    # Converting 'Customer ID' to integers.
    data['Customer ID'] = data['Customer ID'].astype(int)
    # negative_ones_count = (data['Customer ID'] == -1).sum()
    # print(f"Number of - 1s in Customer ID: {negative_ones_count}")

    # Handling missing 'StockCode' data.
    data['StockCode'] = data['StockCode'].fillna(0).astype(int)
    # Handling missing 'Descriptions'
    # Making a dictionary that maps 'StockCode' to 'Description'
    # from rows where not NaN.
    description_dict = data.dropna(
        subset=['Description']).set_index('StockCode')[
        'Description'].to_dict()
    # Apply a function that check if 'Description' is NaN and if the
    # 'StockCode' exists in dict. If it does, fills 'Descsription' with the
    # corresponding value from dict.
    data['Description'] = data.apply(
        lambda row: description_dict[row['StockCode']] if pd.isnull(
            row['Description']) and row['StockCode']
        in description_dict else row['Description'], axis=1)

    # Removing rows with missing 'Description'
    data.dropna(subset=['Description'], inplace=True)

    # Setting 'None' in all missing fields in 'StockCodeVariation
    data['StockCodeVariation'] = data['StockCodeVariation'].fillna('None')

    # Output shape after removing missing entries.
    print("\nShape of cleaned data:\n", data.shape, "\n")

    # Check for missing entries, output for when testing if removing
    # missing data behaves as expected.
    # Comparing data sets.
    missing_entries1 = data1.isnull().sum()
    print("\nMissing Entries in data1:\n", missing_entries1)
    missing_entries2 = data2.isnull().sum()
    print("\nMissing Entries data2:\n", missing_entries2)
    missing_entries3 = data.isnull().sum()
    print("\nMissing Entries, Combined dataset:\n", missing_entries3)

    # Setting the column order for dataframe 'data'
    column_order = ['InvoiceDate',
                    'Invoice',
                    'Customer ID',
                    'Country',
                    'StockCode',
                    'StockCodeVariation',
                    'Description',
                    'Quantity',
                    'Price']

    data = data[column_order]
    # Displaying sample from the dataframe for visual confirmation.
    print("\nCleaned data sample:\n", data.head().to_string(index=False))

    # Splitting the data for the regression task in (CLV)
    cut_off_date = pd.to_datetime('2010-12-01')
    feature_data = data[data['InvoiceDate'] < cut_off_date]
    target_data = data[data['InvoiceDate'] >= cut_off_date]

    # Calculate total spend for each cutomer in the target period.
    total_spend = target_data.groupby(
        'Customer ID')['Price'].sum().reset_index()
    total_spend.rename(columns={'Price': 'FutureSpend'}, inplace=True)

    # Round FutureSpend to 2 decimal places
    total_spend['FutureSpend'] = total_spend['FutureSpend'].round(2)

    # Merging back into 'feature_data'.
    feature_data = feature_data.merge(
        total_spend, on='Customer ID', how='left')

    # Filling NaN values for customers with no spending data in target period.
    feature_data['FutureSpend'] = feature_data['FutureSpend'].fillna(0)
    # zero_count = (feature_data['FutureSpend'] == 0).sum()
    # print("Number of 0's in the dataset:\n", zero_count)

    return data, feature_data


# Customer segmentation part
def customer_segmentation(data):
    # data = preprocessed_data.copy()

    # Adding one day to the end of the 'InvoiceDate' to avoid getting 0s.
    # This is done because of the recency part in the rfm analysis.
    # In the script more focused on machine learning I will try to segment the
    # customers using machine learning, but the rfm analysis is a very common
    # place to start.
    current_date = data['InvoiceDate'].max() + pd.Timedelta(days=1)

    # Grouping data and aggregating new columns
    rfm = data.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (current_date - x.max()).days,
        'Invoice': 'count',
        'Price': lambda x: x.sum()
    }).rename(columns={'InvoiceDate': 'Recency',
                       'Invoice': 'Frequency',
                       'Price': 'Monetary'
                       })

    # Using pandas QuantileCut function for equal distribution.
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 3, ['3', '2', '1'])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'], 3, ['1', '2', '3'])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 3, ['1', '2', '3'])

    # Combining the scores
    rfm['RFM_Segment'] = rfm['R_Score'].astype(
        str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

    # Define levels. Setting values on customers, based on rfm scores.
    def rfm_level(df):
        if df['RFM_Segment'] == '333':
            return 'High Value'
        elif df['RFM_Segment'] in ['322', '332', '312', '323']:
            return 'Medium Value'
        else:
            return 'Low Value'

    # Setting labels on every entry
    rfm['RFM_Level'] = rfm.apply(rfm_level, axis=1)

    # Merging the data frames
    data = data.merge(rfm[['RFM_Level']], on='Customer ID', how='left')
    data = data.merge(rfm[['R_Score']], on='Customer ID', how='left')
    data = data.merge(rfm[['F_Score']], on='Customer ID', how='left')
    data = data.merge(rfm[['M_Score']], on='Customer ID', how='left')

    print("\nCustomer Segmentation results:\n", data.shape)

    return data


def market_basket_analysis(data, clv=False):
    # Converting grouped items to a list of lists,
    # where each list contains items from a single invoice.
    transactions = data.groupby('Invoice')['Description'].apply(list).tolist()
    # Transforming DataFrame using the TransactionEncoder, a one-hot encoder.
    te = TransactionEncoder()
    # Array containing a matrix
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
        frequent_items, metric='lift', min_threshold=2.0)

    # Making some changes to the data, so it looks better
    # Removing 'Frozenset' in front of every entry in antecedent and consequent
    # and limiting the decimals to 5 digits
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

    # Sorting Dataframes for displaying.
    support_sorted = focused_rules.sort_values(by='support', ascending=False)
    confidence_sorted = focused_rules.sort_values(
        by='confidence', ascending=False)
    lift_sorted = focused_rules.sort_values(by='lift', ascending=False)

    if not clv:
        try:
            print("Saving files!")
            support_sorted.to_csv(f'{csv_dir}/support_rules.csv')
            confidence_sorted.to_csv(f'{csv_dir}/confidence_rules.csv')
            lift_sorted.to_csv(f'{csv_dir}/lift_rules.csv')
            print("Files Saved!")
        except Exception as e:
            print(f"Failed to save the file: {e}")
            exit(1)

    # Creating new features from the rules, and combining them with
    # the 'data' dataframe.
    top_rules_lift = lift_sorted.head(10)

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

    # Top 10 rules taken from the 'confidence_sorted' dataframe.
    for index, rule in top_rules_lift.iterrows():
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

    print("\nDf:\n", data.head())
    print("Shape of data:\n", data.shape)

    # Dropping the 'Items' column as it clutters up the dataframe.
    # only needed to create the binary features for the rules.
    data.drop(columns='Items', inplace=True)

    # Making a new column containing both antecedent and consequents,
    # primarily for the visualizations.
    support_sorted['itemsets'] = support_sorted['antecedents'] + \
        ' +\n ' + support_sorted['consequents']
    confidence_sorted['itemsets'] = confidence_sorted['antecedents'] + \
        ' +\n ' + confidence_sorted['consequents']
    lift_sorted['itemsets'] = lift_sorted['antecedents'] + \
        ' +\n ' + lift_sorted['consequents']

    if not clv:
        plot_path = './plots'
        # Visualization of the most common itemsets(antecedent + consequents).
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(x='support', y='itemsets',
                    data=support_sorted.head(10), color='darkblue')
        ax.set_title('Top 10 Most Common Itemsets by Support')
        ax.set_xlabel('Support')
        ax.set_ylabel('Itemsets (Antecedent + Consequent)')
        ax.xaxis.set_major_locator(MaxNLocator(10))
        fig.tight_layout()
        os.makedirs(plot_path, exist_ok=True)
        fig.savefig(f'{plot_path}/top_itemsets_by_support.png', dpi=300,
                    format='png', bbox_inches='tight')

        # Setting the lower xlim to 0.5 because the threshold in the
        # association rules was set to 0.5. The upper limit is set to 1.0
        # which is the maximum for confidence, showing the full possible range.
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        sns.barplot(x='confidence', y='itemsets',
                    data=confidence_sorted.head(10), color='darkblue')
        ax1.set_xlim(0.5, 1.0)
        ax1.set_title('Top 10 Rules for Confidence')
        ax1.set_xlabel('Confidence')
        ax1.set_ylabel('Itemsets (Antecedent + Consequent)')
        ax1.xaxis.set_major_locator(MaxNLocator(10))
        fig1.tight_layout()
        fig1.savefig(f'{plot_path}/top_itemsets_by_confidence.png', dpi=300,
                     format='png', bbox_inches='tight')

        fig2, ax2 = plt.subplots(figsize=(10, 8))
        sns.barplot(x='lift', y='itemsets',
                    data=lift_sorted.head(10), color='darkblue')
        ax2.set_title('Top 10 Rules for Lift')
        ax2.set_xlabel('Lift')
        ax2.set_ylabel('Itemsets (Antecedent + Consequent)')
        ax2.xaxis.set_major_locator(MaxNLocator(10))
        fig2.tight_layout()
        fig2.savefig(f'{plot_path}/top_itemsets_by_lift.png', dpi=300,
                     format='png', bbox_inches='tight')

        plt.close('all')

    return data


def main():
    # Running the functions from here.
    preprocessed_data, clv_data = preprocess_data()
    segmentation_results = customer_segmentation(preprocessed_data)
    mba_analysed_data = market_basket_analysis(segmentation_results)
    mba_analysed_data.to_csv(f"{csv_dir}/data_featured.csv", index=False)

    # Featured dataset
    clv_cs_results = customer_segmentation(clv_data)
    clv_mba_results = market_basket_analysis(clv_cs_results, clv=True)
    # print("clv cols: \n", clv_mba_results.columns)
    clv_mba_results.to_csv(f'{csv_dir}/clv_data_featured.csv', index=False)


if __name__ == "__main__":
    main()
