import pandas as pd # library to handle dataframes
import matplotlib # plots, visualization 
import matplotlib.pyplot as plt # library to plot graphics
import numpy as np # to handle matrices
import seaborn as sns # to build graphics
from scipy.stats import kurtosis, skew # statistics of numerical values
from scipy import stats # ststistical tests, statistical analysis 
from scipy.stats import linregress # for linear regression 
import re #regex
data = pd.read_json('winemag-data-130k-v2.json')
data.head()
data.describe(include='all',).T # descriptive statistics for the data transposed to make it easier to read

# there are some NaN values, let's see how many 
# how many NaN values are in data 
total_null_values = data.isna().sum().sum() 
print('Total null values:', total_null_values)

complete_cases = len(data.dropna()) # number of rows of 
print('Complete cases:',complete_cases)
# checking missing values 
data.isnull().sum()
# checking data types in the dataframe 
data.dtypes
# replacing missing values in 'price' column with the mean value
mean_price = data['price'].mean()
data['price'] = data['price'].fillna(mean_price)
# reseting index to ensure alignment
data.reset_index(drop=True, inplace=True)

# checking if imputation worked
data.isnull().sum()


best_grape_ratings = data.groupby('variety')['points'].mean().sort_values(ascending=False).head(10)
best_grape_ratings # top 10 grapes used in most highly rated grapes

# scatter plot to visualize the relationship between the 'price'and'points'
plt.figure(figsize=(10, 6))
sns.scatterplot(x='price', y='points', data=data)
plt.title('Relationship between Price and Rating Points')
plt.xlabel('Price')
plt.ylabel('Rating Points')
plt.show()

# linear regression
slope, intercept, r_value, p_value, std_err = linregress(data['price'], data['points'])

print("Slope:", slope)
print("Intercept:", intercept)
print("r_value:",r_value)
print("p_value:",p_value)
print("Standard error:",std_err)

# find Pearson correlation coefficient
pearson_corr, _ = stats.pearsonr(data['price'], data['points'])
pearson_corr

# top countries producing wines with the highest ratings

country_ratings = data.groupby('country')['points'].mean().sort_values(ascending=False)
top_countries_ratings = country_ratings.head(10)  # Get top 10 countries
top_countries_ratings


# how many wines from each country is in dataframe
country_counts = data['country'].value_counts()
country_counts

# top 10 countries that produce more than 80  wines with the highest ratings
top_countries = country_counts[country_counts > 80].index
top_countries


# filter out data that only includes wine from selected top countries
filtered_data = data[data['country'].isin(top_countries)]
filtered_data


# average rating for each country for filtered data (top 10)
average_rating = filtered_data.groupby('country')['points'].mean()
top_10_countries = average_rating.sort_values(ascending=False).head(10)
                                                                

print("Top 10 countries producing more than 80 wines with the highest ratings:")
print(top_10_countries)


# bar plot for top countries producing wines with highest raitings

# top 10 countries that produce more than 80  wines with the highest ratings
top_countries = country_counts[country_counts > 80].index
# filter out data that only includes wine from selected top countries
filtered_data = data[data['country'].isin(top_countries)]

# group data by rating
avg_rating = filtered_data.groupby('country')['points'].mean().reset_index()
# sort countries by ratings 
avg_rating_sorted = avg_rating.sort_values(by='points', ascending=False)
# top 10 countries 
top_countries = avg_rating_sorted.head(10)

plt.figure(figsize=(10, 6))
plt.title('Top Countries Producing Wines with the Highest Ratings')
plt.xlabel('Country')
plt.ylabel('Average Rating Points')
sns.barplot(x='points', y='country', data=top_countries, palette='viridis')
plt.xticks(rotation=360)
plt.show()


# plot a histogram showing the distribution of wine ratings
plt.figure(figsize=(10, 6))
plt.hist(data['points'], bins=30)
plt.title('Distribution of Wine Ratings')
plt.xlabel('Points')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# bar chart for top 10 countries that produce the most wine
top_10_countries = data['country'].value_counts().head(10)

plt.figure(figsize=(10, 6))
# set different color for each country 
sns.barplot(x=top_10_countries.index, y=top_10_countries.values, palette='viridis')
plt.title('Top 10 Countries Producing the Most Wines')
plt.xlabel('Country')
plt.ylabel('Number of Wines')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()


# box plot for distribution for different grape varieties 
plt.figure(figsize=(12, 8))
sns.boxplot(x='variety', y='price', data=data)
plt.title('Distribution of Wine Prices for Different Varieties')
plt.xlabel('Variety')
plt.ylabel('Price')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.grid(True)
plt.show()


# box plot for top 10 grapes
top_10_grape_varieties = best_grape_ratings.index
filtered_data = data[data['variety'].isin(top_10_grape_varieties)]
plt.figure(figsize=(12, 8))
sns.boxplot(x='variety', y='price', data=filtered_data)
plt.title('Distribution of Wine Prices for Top 10 Grape Varieties with Highest Ratings')
plt.xlabel('Variety')
plt.ylabel('Price')
plt.xticks(rotation=30)  # Rotate x-axis labels for better readability
plt.grid(True)
plt.show()



# frequency of each grape variety
grape_variety_counts = data['variety'].value_counts()

# 10 most popular grape varieties
top_10_popular_grapes = grape_variety_counts.head(10).index

# include only wines with the 10 most popular varieties
top_10_popular_grapes_data = data[data['variety'].isin(top_10_popular_grapes)]

# average rating for each variety within top 10 popular grapes
best_rated_within_top_10 = top_10_popular_grapes_data.groupby('variety')['points'].mean().sort_values(ascending=False)

best_rated_within_top_10 


# plot a box plot for top 10 popular grapes 
top_10_popular_grapes_data = data[data['variety'].isin(top_10_popular_grapes)]
best_rated_grape_varieties = best_rated_within_top_10.index
filtered_data = top_10_popular_grapes_data[top_10_popular_grapes_data['variety'].isin(best_rated_grape_varieties)]
filtered_data = filtered_data[filtered_data['price'] < 600]
plt.figure(figsize=(12, 8))
sns.boxplot(x='variety', y='price', data=filtered_data)
plt.title('Wine Prices for Top 10 Grape Varieties with Highest Ratings')
plt.xlabel('Grape Variety')
plt.ylabel('Price')
plt.xticks(rotation=30)  # for better visibility 
plt.grid(True)
plt.show()



# check how many expensive and affordable wines are there in the dataframe
num_expensive_wines = data[data['price'] > 400].shape[0]
num_affordable_wines = data[(data['price'] >= 0) & (data['price'] <= 100)].shape[0]
num_expensive_wines
num_affordable_wines

# find z_scores to identify outliers in 'price'
z_scores = (data['price'] - data['price'].mean()) / data['price'].std()
np.mean(z_scores)
np.std(z_scores)


# define a threshold for identifying outliers
# points that are more than 3 standard deviations away from the mean
# threshold is predetermined
threshold = 3
# filter data to include only rows with outliers 
outliers = data[np.abs(z_scores) > threshold]
outliers.shape


# box plot for wine prices for top 10 grape varieties, without outliers
top_10_popular_grapes_data = data[data['variety'].isin(top_10_popular_grapes)]
best_rated_grape_varieties = best_rated_within_top_10.index
filtered_data = top_10_popular_grapes_data[top_10_popular_grapes_data['variety'].isin(best_rated_grape_varieties)]
filtered_data = filtered_data[~filtered_data.index.isin(outliers.index)]
plt.figure(figsize=(12, 8))
sns.boxplot(x='variety', y='price', data=filtered_data)
plt.title('Wine Prices for Top 10 Grape Varieties with Highest Ratings')
plt.xlabel('Grape Variety')
plt.ylabel('Price')
plt.xticks(rotation=30)  # Rotate x-axis labels for better readability
plt.grid(True)
plt.show()

# use countplot to show top 15 provinces in wine production 
plt.figure(figsize=(10, 6))
sns.countplot(y='province', data=data, order=data['province'].value_counts().index[:15])
plt.title('Top Provinces by Wine Production')
plt.xlabel('Count')
plt.ylabel('Province')
plt.show()


# box plot to show point average for top 10 provinces
plt.figure(figsize=(12, 8))
# use boolean indexing to filter data to get rows with most popular provinces
sns.boxplot(x='points', y='province', data=data[data['province'].isin(data['province'].value_counts().index[:10])])
plt.title('Provinces vs Points')
plt.xlabel('Points')
plt.ylabel('Province')
plt.show()


# plot top 15 wineries that produce most wines
plt.figure(figsize=(12, 8))
sns.countplot(y='winery', data=data, order=data['winery'].value_counts().index[:15])
plt.title('Top 15 Wineries')
plt.xlabel('Count')
plt.ylabel('Winery')
plt.show()


# box plot of prices by winery, where price ranges from 0 to 150 for top 10 wineries
filtered_data = data[(data['price'] >= 0) & (data['price'] <= 150) & (data['winery'].isin(data['winery'].value_counts().index[:10]))]
plt.figure(figsize=(12, 8))
sns.boxplot(x='price', y='winery', data=filtered_data)
plt.title('Price by Winery')
plt.xlabel('Price')
plt.ylabel('Winery')
plt.show()



# box plot of points by top wineries 
plt.figure(figsize=(12, 8))
sns.boxplot(x='points', y='winery', data=data[data['winery'].isin(data['winery'].value_counts().index[:10])])
plt.title('Points by Winery')
plt.xlabel('Points')
plt.ylabel('Winery')
plt.show()


# use subplots to plot side by side bar plots of top wineries and top provinces 
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
# top 10 wineries subplot                                                                                     
sns.countplot(y='winery', data=data, order=data['winery'].value_counts().index[:10], ax=axes[0])
axes[0].set_title('Top Wineries')
axes[0].set_xlabel('Count')
axes[0].set_ylabel('Winery') 
# top 10 provinces                                                                                       
sns.countplot(y='province', data=data, order=data['province'].value_counts().index[:10],ax=axes[1])
axes[1].set_title('Top Provinces')
axes[1].set_xlabel('Count')
axes[1].set_ylabel('Province')
# adjust space between subplots            
plt.tight_layout()
plt.show()

# average price for each country in each group 'country'
# resetting index after calculating mean price in each group
count = data.groupby('country')['price'].mean().reset_index()
plt.figure(figsize=(12,8))
sns.pointplot(x ='price',y ='country', data=count,  color='r')
plt.title('Country wise average wine price')
plt.xlabel('Price')
plt.ylabel('Country')
plt.grid(True)
plt.show()
