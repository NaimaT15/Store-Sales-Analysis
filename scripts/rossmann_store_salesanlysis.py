import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def check_promotion_distribution(train_df, test_df, promotion_column):
    """
    Check the distribution of promotions between training and test sets.
    
    """
    
    # Check for missing values in the promotion column
    if train_df[promotion_column].isnull().sum() > 0 or test_df[promotion_column].isnull().sum() > 0:
        print("Warning: Missing values found in the promotion column. Consider handling them before analysis.")

    # Plot the distribution of promotions in both datasets
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Training set promotion distribution
    sns.histplot(train_df[promotion_column], kde=False, ax=axes[0], color="blue")
    axes[0].set_title('Promotion Distribution in Training Set')
    axes[0].set_xlabel('Promotion')
    axes[0].set_ylabel('Frequency')

    # Test set promotion distribution
    sns.histplot(test_df[promotion_column], kde=False, ax=axes[1], color="green")
    axes[1].set_title('Promotion Distribution in Test Set')
    axes[1].set_xlabel('Promotion')
    axes[1].set_ylabel('Frequency')

    # Display the plots
    plt.tight_layout()
    plt.show()

    # Statistical comparison (optional)
    train_promo_dist = train_df[promotion_column].value_counts(normalize=True)
    test_promo_dist = test_df[promotion_column].value_counts(normalize=True)

    print("Training Set Promotion Distribution (Normalized):\n", train_promo_dist)
    print("\nTest Set Promotion Distribution (Normalized):\n", test_promo_dist)
    
    # Calculate percentage difference between the two distributions
    promo_diff = abs(train_promo_dist - test_promo_dist).fillna(0)
    print("\nPercentage Difference in Promotion Distribution:\n", promo_diff)


def merge_train_store(train_file, store_file):
    """
    Merge the train and store datasets on the 'Store' column.
    """


    # Perform the merge operation on the 'Store' column
    merged_df = pd.merge(train_file, store_file, how='inner', on='Store')

    # Return the merged DataFrame
    return merged_df


def clean_missing_values(merged_df):
    """
    Cleans the missing values in the dataset according to the specified strategies.

    """
    # Fill missing values in 'CompetitionDistance' with the median
    merged_df['CompetitionDistance'].fillna(merged_df['CompetitionDistance'].median(), inplace=True)

    # Fill missing values in 'CompetitionOpenSinceMonth' and 'CompetitionOpenSinceYear' with 0
    merged_df['CompetitionOpenSinceMonth'].fillna(0, inplace=True)
    merged_df['CompetitionOpenSinceYear'].fillna(0, inplace=True)

    # Fill missing values in 'Promo2SinceWeek' and 'Promo2SinceYear' with 0
    merged_df['Promo2SinceWeek'].fillna(0, inplace=True)
    merged_df['Promo2SinceYear'].fillna(0, inplace=True)

    # Fill missing values in 'PromoInterval' with 'None'
    merged_df['PromoInterval'].fillna('None', inplace=True)

    # Return the cleaned DataFrame
    return merged_df

def sales_behavior_holidays_auto(merged_df, sales_column, date_column, state_holiday_column, school_holiday_column, days_before=7, days_after=7):
    """
    Check and compare sales behavior before, during, and after holidays using 'StateHoliday' or 'SchoolHoliday'.
    """
    
    # Ensure date_column is in datetime format
    merged_df[date_column] = pd.to_datetime(merged_df[date_column])

    # Create a new column to indicate the holiday period: State or School holidays
    merged_df['HolidayPeriod'] = 'Normal'

    # Define the mask for state or school holidays
    holiday_mask = (merged_df[state_holiday_column] != '0') | (merged_df[school_holiday_column] == 1)

    # Identify holiday dates
    holiday_dates = merged_df.loc[holiday_mask, date_column].unique()

    # For each holiday, mark the periods before, during, and after
    for holiday in holiday_dates:
        holiday_date = pd.to_datetime(holiday)

        # Mark periods before, during, and after each holiday
        before_mask = (merged_df[date_column] >= holiday_date - pd.Timedelta(days=days_before)) & (merged_df[date_column] < holiday_date)
        during_mask = (merged_df[date_column] == holiday_date)
        after_mask = (merged_df[date_column] > holiday_date) & (merged_df[date_column] <= holiday_date + pd.Timedelta(days=days_after))

        merged_df.loc[before_mask, 'HolidayPeriod'] = 'Before Holiday'
        merged_df.loc[during_mask, 'HolidayPeriod'] = 'During Holiday'
        merged_df.loc[after_mask, 'HolidayPeriod'] = 'After Holiday'

    # Plot sales behavior before, during, and after holidays
    plt.figure(figsize=(12, 6))
    sns.barplot(x='HolidayPeriod', y=sales_column, data=merged_df, order=['Before Holiday', 'During Holiday', 'After Holiday', 'Normal'])
    plt.title('Sales Behavior Before, During, and After Holidays')
    plt.xlabel('Holiday Period')
    plt.ylabel('Sales')
    plt.show()

    # Group by the holiday period and summarize the sales behavior
    summary = merged_df.groupby('HolidayPeriod')[sales_column].agg(['mean', 'median', 'std', 'count'])
    print("\nSales Behavior Summary by Holiday Period:\n", summary)


def sales_behavior_holidays_extended(merged_df, sales_column, date_column, state_holiday_column, school_holiday_column, days_before=7, days_after=7):
    """
    Check and compare sales behavior before, during, and after holidays using both StateHoliday and SchoolHoliday columns.
    """
    
    # Ensure date_column is in datetime format
    merged_df[date_column] = pd.to_datetime(merged_df[date_column])

    # Extract holiday dates based on StateHoliday and SchoolHoliday columns
    state_holidays = merged_df[merged_df[state_holiday_column] != '0'][date_column].unique()
    school_holidays = merged_df[merged_df[school_holiday_column] == 1][date_column].unique()

    # Combine both state and school holidays
    all_holidays = pd.Series(list(state_holidays) + list(school_holidays)).unique()

    # Create a new column to indicate the sales time relative to the holiday
    merged_df['HolidayPeriod'] = 'Normal'

    for holiday in all_holidays:
        holiday_date = pd.to_datetime(holiday)

        # Mark periods before, during, and after each holiday
        before_mask = (merged_df[date_column] >= holiday_date - pd.Timedelta(days=days_before)) & (merged_df[date_column] < holiday_date)
        during_mask = (merged_df[date_column] == holiday_date)
        after_mask = (merged_df[date_column] > holiday_date) & (merged_df[date_column] <= holiday_date + pd.Timedelta(days=days_after))

        merged_df.loc[before_mask, 'HolidayPeriod'] = 'Before Holiday'
        merged_df.loc[during_mask, 'HolidayPeriod'] = 'During Holiday'
        merged_df.loc[after_mask, 'HolidayPeriod'] = 'After Holiday'

    # Plot sales behavior before, during, and after holidays
    plt.figure(figsize=(12, 6))
    sns.barplot(x='HolidayPeriod', y=sales_column, data=merged_df, order=['Before Holiday', 'During Holiday', 'After Holiday', 'Normal'])
    plt.title('Sales Behavior Before, During, and After Holidays (State and School Holidays)')
    plt.xlabel('Holiday Period')
    plt.ylabel('Sales')
    plt.show()

    # Group by the holiday period and summarize the sales behavior
    summary = merged_df.groupby('HolidayPeriod')[sales_column].agg(['mean', 'median', 'std', 'count'])
    print("\nSales Behavior Summary by Holiday Period:\n", summary)



def seasonal_purchase_behavior(merged_df, sales_column, date_column, state_holiday_column):
    """
    Analyze seasonal purchase behaviors (e.g., Christmas, Easter) using StateHoliday and other known dates.
    """
    
    # Ensure date_column is in datetime format
    merged_df[date_column] = pd.to_datetime(merged_df[date_column])

    # Create a new column for seasonality
    merged_df['Season'] = 'Regular'

    # Define Christmas and Easter periods
    christmas_mask = merged_df[state_holiday_column] == 'c'
    easter_mask = merged_df[state_holiday_column] == 'b'
    
    # Define other common holiday periods if available
    public_holiday_mask = merged_df[state_holiday_column] == 'a'

    # Update the 'Season' column based on the holiday type
    merged_df.loc[christmas_mask, 'Season'] = 'Christmas'
    merged_df.loc[easter_mask, 'Season'] = 'Easter'
    merged_df.loc[public_holiday_mask, 'Season'] = 'Public Holiday'

    # Plot seasonal behavior
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Season', y=sales_column, data=merged_df, order=['Regular', 'Christmas', 'Easter', 'Public Holiday'])
    plt.title('Sales Behavior During Seasonal Events (Christmas, Easter, Public Holidays)')
    plt.xlabel('Seasonal Event')
    plt.ylabel('Sales')
    plt.show()

    # Group by the season and summarize the sales behavior
    summary = merged_df.groupby('Season')[sales_column].agg(['mean', 'median', 'std', 'count'])
    print("\nSales Behavior Summary by Season:\n", summary)



def analyze_sales_customers_correlation(merged_df, sales_column, customers_column):
    """
    Analyze the correlation between sales and the number of customers.
    """
    
    # Calculate the Pearson correlation coefficient between sales and customers
    correlation = merged_df[[sales_column, customers_column]].corr().iloc[0, 1]
    print(f"Correlation between {sales_column} and {customers_column}: {correlation:.4f}")

    # Scatter plot to visualize the relationship between sales and customers
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=customers_column, y=sales_column, data=merged_df)
    plt.title(f'Sales vs Customers (Correlation: {correlation:.4f})')
    plt.xlabel('Number of Customers')
    plt.ylabel('Sales')
    plt.show()

    # Optional: Joint plot for deeper analysis
    sns.jointplot(x=customers_column, y=sales_column, data=merged_df, kind='reg', height=8)
    plt.show()


def analyze_promo_effects(merged_df, sales_column, customers_column, promo_column):
    """
    Analyze the effect of promotions on sales and customers.
    """
    
    # Separate promo and non-promo days
    promo_sales = merged_df[merged_df[promo_column] == 1]
    no_promo_sales = merged_df[merged_df[promo_column] == 0]
    
    # Calculate Sales per Customer to analyze how much customers spend on promo and non-promo days
    merged_df['SalesPerCustomer'] = merged_df[sales_column] / merged_df[customers_column]

    # 1. Compare Sales on Promo vs Non-Promo Days
    plt.figure(figsize=(10, 6))
    sns.barplot(x=promo_column, y=sales_column, data=merged_df)
    plt.title('Sales on Promo Days (1) vs Non-Promo Days (0)')
    plt.xlabel('Promo')
    plt.ylabel('Sales')
    plt.show()

    # 2. Compare Number of Customers on Promo vs Non-Promo Days
    plt.figure(figsize=(10, 6))
    sns.barplot(x=promo_column, y=customers_column, data=merged_df)
    plt.title('Number of Customers on Promo Days (1) vs Non-Promo Days (0)')
    plt.xlabel('Promo')
    plt.ylabel('Number of Customers')
    plt.show()

    # 3. Compare Sales Per Customer on Promo vs Non-Promo Days
    plt.figure(figsize=(10, 6))
    sns.barplot(x=promo_column, y='SalesPerCustomer', data=merged_df)
    plt.title('Sales Per Customer on Promo Days (1) vs Non-Promo Days (0)')
    plt.xlabel('Promo')
    plt.ylabel('Sales Per Customer')
    plt.show()

    # Summary Statistics for Sales, Customers, and Sales Per Customer
    promo_summary = merged_df.groupby(promo_column)[[sales_column, customers_column, 'SalesPerCustomer']].agg(['mean', 'median', 'std', 'count'])
    print("\nPromo Effect Summary (Sales, Customers, Sales Per Customer):\n", promo_summary)




def analyze_promo_by_top_stores(merged_df, sales_column, customers_column, promo_column, store_column, store_type_column, top_n=10):
    """
    Analyze the effect of promotions on sales and customers by store and store type,
    limiting the analysis to the top N stores based on sales.
    """

    # Group by store and analyze sales and customer changes with/without promotions
    promo_sales_by_store = merged_df.groupby([store_column, promo_column])[[sales_column, customers_column]].mean().unstack()

    # Get top N stores based on average sales (with and without promo)
    top_stores = promo_sales_by_store[sales_column].mean(axis=1).nlargest(top_n).index

    # Filter the dataset for only the top N stores
    filtered_promo_sales_by_store = promo_sales_by_store.loc[top_stores]

    # Visualize promo effectiveness by store for the top N stores
    ax = filtered_promo_sales_by_store.plot(kind='bar', figsize=(12, 6), title=f'Average Sales and Customers for Top {top_n} Stores (Promo vs Non-Promo)')
    plt.xlabel('Store')
    plt.ylabel('Average Sales/Customers')
    plt.legend(['Sales (No Promo)', 'Sales (Promo)', 'Customers (No Promo)', 'Customers (Promo)'])
    
    # Rotate the x-axis labels to avoid overlap
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()

    # Analyze the impact of promotions across store types
    plt.figure(figsize=(10, 6))
    sns.barplot(x=store_type_column, y=sales_column, hue=promo_column, data=merged_df)
    plt.title('Sales by Store Type (Promo vs Non-Promo)')
    plt.xlabel('Store Type')
    plt.ylabel('Sales')
    plt.show()

    # Summary statistics of promo effects by store
    promo_effectiveness_summary = merged_df.groupby([store_column, promo_column])[[sales_column, customers_column]].agg(['mean', 'std', 'count'])
    print(f"\nPromo Effectiveness Summary for Top {top_n} Stores (Sales and Customers):\n", promo_effectiveness_summary)



def analyze_store_opening_closing_behavior(merged_df, sales_column, customers_column, open_column, date_column):
    """
    Analyze trends of customer behavior during store opening and closing times.
    """

    # Convert date_column to datetime format if needed
    merged_df[date_column] = pd.to_datetime(merged_df[date_column])

    # 1. Compare Sales and Customers when Store is Open vs Closed
    plt.figure(figsize=(10, 6))
    sns.barplot(x=open_column, y=sales_column, data=merged_df)
    plt.title('Sales When Store is Open (1) vs Closed (0)')
    plt.xlabel('Store Open (1) vs Closed (0)')
    plt.ylabel('Sales')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=open_column, y=customers_column, data=merged_df)
    plt.title('Number of Customers When Store is Open (1) vs Closed (0)')
    plt.xlabel('Store Open (1) vs Closed (0)')
    plt.ylabel('Number of Customers')
    plt.show()

    # 2. Analyze day-wise trends for customer behavior during store opening and closing
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=merged_df[date_column].dt.day_name(), y=customers_column, hue=open_column, data=merged_df, ci=None)
    plt.title('Customer Behavior by Day of Week (Open vs Closed)')
    plt.xlabel('Day of the Week')
    plt.ylabel('Number of Customers')
    plt.show()

    # 3. Analyze store sales by day-wise trends
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=merged_df[date_column].dt.day_name(), y=sales_column, hue=open_column, data=merged_df, ci=None)
    plt.title('Sales by Day of Week (Open vs Closed)')
    plt.xlabel('Day of the Week')
    plt.ylabel('Sales')
    plt.show()

    # Summary statistics of sales and customer behavior when open vs closed
    open_closed_summary = merged_df.groupby(open_column)[[sales_column, customers_column]].agg(['mean', 'median', 'std', 'count'])
    print("\nStore Opening and Closing Summary (Sales and Customers):\n", open_closed_summary)



def analyze_weekday_open_stores_and_weekend_sales(merged_df, sales_column, date_column, store_column):
    """
    Identify stores open on all weekdays (Monday to Friday) and analyze how that affects their sales on weekends.
    
    """

    # Ensure the date column is in datetime format
    merged_df[date_column] = pd.to_datetime(merged_df[date_column])

    # Add a day of the week column based on the Date
    merged_df['DayOfWeek'] = merged_df[date_column].dt.day_name()

    # Filter for stores open on all weekdays
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    weekday_data = merged_df[merged_df['DayOfWeek'].isin(weekdays) & (merged_df['Open'] == 1)]
    
    # Group by store to count how many weekdays each store is open
    weekday_open_counts = weekday_data.groupby(store_column)['DayOfWeek'].nunique()
    
    # Identify stores that are open all 5 weekdays
    stores_open_all_weekdays = weekday_open_counts[weekday_open_counts == 5].index
    print(f"Number of stores open on all weekdays: {len(stores_open_all_weekdays)}")

    # Now, analyze weekend sales for these stores
    weekend_days = ['Saturday', 'Sunday']
    weekend_data = merged_df[(merged_df[store_column].isin(stores_open_all_weekdays)) & 
                             (merged_df['DayOfWeek'].isin(weekend_days)) & 
                             (merged_df['Open'] == 1)]

    # 1. Visualize weekend sales (Saturday vs Sunday)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='DayOfWeek', y=sales_column, data=weekend_data)
    plt.title('Weekend Sales for Stores Open on All Weekdays')
    plt.xlabel('Weekend Day')
    plt.ylabel('Sales')
    plt.show()

    # 2. Summary statistics for weekend sales
    weekend_sales_summary = weekend_data.groupby('DayOfWeek')[sales_column].agg(['mean', 'median', 'std', 'count'])
    print("\nWeekend Sales Summary for Stores Open on All Weekdays:\n", weekend_sales_summary)

    # Optional: Compare weekend sales for stores that are open vs closed during weekdays
    non_weekday_open_stores = merged_df[~merged_df[store_column].isin(stores_open_all_weekdays)]
    non_weekday_weekend_data = non_weekday_open_stores[non_weekday_open_stores['DayOfWeek'].isin(weekend_days) & (non_weekday_open_stores['Open'] == 1)]

    # 3. Visualize weekend sales for stores not open all weekdays
    plt.figure(figsize=(10, 6))
    sns.barplot(x='DayOfWeek', y=sales_column, data=non_weekday_weekend_data)
    plt.title('Weekend Sales for Stores Not Open on All Weekdays')
    plt.xlabel('Weekend Day')
    plt.ylabel('Sales')
    plt.show()

    # 4. Summary statistics for stores not open all weekdays
    non_weekday_weekend_sales_summary = non_weekday_weekend_data.groupby('DayOfWeek')[sales_column].agg(['mean', 'median', 'std', 'count'])
    print("\nWeekend Sales Summary for Stores Not Open on All Weekdays:\n", non_weekday_weekend_sales_summary)





def analyze_assortment_effect_on_sales(merged_df, sales_column, assortment_column):
    """
    Analyze how the assortment type affects sales.
    """

    # 1. Group by assortment type and calculate summary statistics for sales
    assortment_sales_summary = merged_df.groupby(assortment_column)[sales_column].agg(['mean', 'median', 'std', 'count'])

    # 2. Visualize the sales distribution for each assortment type
    plt.figure(figsize=(10, 6))
    sns.barplot(x=assortment_column, y=sales_column, data=merged_df)
    plt.title('Sales Distribution by Assortment Type')
    plt.xlabel('Assortment Type')
    plt.ylabel('Sales')
    plt.show()

    # 3. Display summary statistics
    print("\nSales Summary by Assortment Type:\n", assortment_sales_summary)


def analyze_competitor_distance_effect_on_sales(merged_df, sales_column, competition_distance_column, store_type_column):
    """
    Analyze how the distance to the nearest competitor affects sales and how this varies in city centers.
    """

    # 1. Correlation between competition distance and sales
    correlation = merged_df[[sales_column, competition_distance_column]].corr().iloc[0, 1]
    print(f"Correlation between {competition_distance_column} and {sales_column}: {correlation:.4f}")

    # 2. Visualize how sales are distributed across different ranges of competition distance
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=competition_distance_column, y=sales_column, data=merged_df)
    plt.title('Sales vs Competition Distance')
    plt.xlabel('Competition Distance (meters)')
    plt.ylabel('Sales')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=store_type_column, y=sales_column, data=merged_df, hue=competition_distance_column)
    sns.stripplot(x=store_type_column, y=sales_column, data=merged_df, hue=competition_distance_column, jitter=True, size=2, alpha=0.5, dodge=True)
    plt.title('Sales by Store Location and Competition Distance')
    plt.xlabel('Store Location')
    plt.ylabel('Sales')
    plt.show()

    # Summary statistics for sales based on competition distance
    distance_sales_summary = merged_df.groupby([competition_distance_column, store_type_column])[sales_column].agg(['mean', 'median', 'std', 'count'])
    print("\nSales Summary by Competition Distance and Store Location:\n", distance_sales_summary)




def analyze_new_competitors_effect_on_sales(merged_df, sales_column, competition_distance_column, date_column, store_column):
    """
    Analyze the effect of the opening of new competitors on store sales by checking stores with initially 'NA' 
    in CompetitionDistance and later getting values.
    """

    # Ensure the date column is in datetime format
    merged_df[date_column] = pd.to_datetime(merged_df[date_column])

    # 1. Identify stores with initially 'NA' in CompetitionDistance and later have a valid value
    na_competitors = merged_df[merged_df[competition_distance_column].isna()]
    new_competitors = merged_df[merged_df[competition_distance_column].notna()]

    # Merge based on Store and Date to track when a competitor entered
    stores_with_new_competitors = new_competitors[store_column].unique()

    # Filter data for stores that had 'NA' for competitors initially
    affected_stores = merged_df[merged_df[store_column].isin(stores_with_new_competitors)]

    # 2. Analyze sales before and after the competitor entered the market
    # Assuming that after the competitor distance is no longer 'NA', a competitor opened
    affected_stores['CompetitorEntered'] = affected_stores[competition_distance_column].notna()

    # Create a 'Before' and 'After' column for competitor entry
    affected_stores['TimePeriod'] = ['Before Competitor' if x is None else 'After Competitor' for x in affected_stores[competition_distance_column]]

    # 3. Visualize sales before and after competitor entry
    plt.figure(figsize=(10, 6))
    sns.barplot(x='TimePeriod', y=sales_column, data=affected_stores)
    plt.title('Sales Before and After Competitor Entry')
    plt.xlabel('Competitor Entry Period')
    plt.ylabel('Sales')
    plt.show()

    # 4. Plot a time series of sales for affected stores
    affected_stores.set_index(date_column, inplace=True)
    affected_stores.groupby('TimePeriod')[sales_column].plot(legend=True)
    plt.title('Sales Trends Before and After Competitor Entry')
    plt.ylabel('Sales')
    plt.xlabel('Date')
    plt.show()

    # 5. Summary statistics
    competitor_effect_summary = affected_stores.groupby('TimePeriod')[sales_column].agg(['mean', 'median', 'std', 'count'])
    print("\nSales Summary Before and After Competitor Entry:\n", competitor_effect_summary)

