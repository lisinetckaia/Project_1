import pandas as pd #Library to handle with dataframes
import matplotlib
import matplotlib.pyplot as plt # Library to plot graphics
import numpy as np # To handle with matrices
import seaborn as sns # to build modern graphics
from scipy.stats import kurtosis, skew # it's to explore some statistics of numerical values
from scipy import stats

data = pd.read_json('winemag-data-130k-v2.json')




#path = '../Downloads/'
#wine_150k = pd.read_csv(path + 'wine_data1.csv')
#wine_130k = pd.read_csv(path + 'wine_data2.csv')
#wine = pd.concat([wine_150k,wine_130k],axis=0)
#wine.shape
#wine.head()




data.head()




data.describe(include='all',).T # descriptive statistics for the data transposed to make it easier to read





wine_variety = data['variety'] # retrieve the column 'variety' from data
wine_descriptions = data['description'] # retrieve the column 'description' from data




wine_variety


wine_descriptions

variety_counts = wine_variety.value_counts() # counts unique values of labels
variety_counts



# how many NaN values are in data 
total_null_values = data.isna().sum().sum() 
print('Total null values:', total_null_values)



complete_cases = len(data.dropna()) # number of rows of 
print('Complete cases:',complete_cases)



#drop colums that we won't use and replace current data dataframe
data.drop(columns=['taster_name', 'taster_twitter_handle', 'region_1', 'winery'], inplace=True) 




data.head()




best_grape_ratings = data.groupby('variety')['points'].mean().sort_values(ascending=False).head(10)
best_grape_ratings # top 10 grapes used in most highly rated grapes



# scatter plot to visualize the relationship between the 'price'and'points'
plt.figure(figsize=(10, 6))
sns.scatterplot(x='price', y='points', data=data)
plt.title('Relationship between Price and Rating Points')
plt.xlabel('Price')
plt.ylabel('Rating Points')
plt.show()





# find Pearson correlation coefficient between price and rating points: 
pearson_corr, _ = stats.pearsonr(data['price'], data['points'])
pearson_corr
    





country_ratings = data.groupby('country')['points'].mean().sort_values(ascending=False)
top_countries_ratings = country_ratings.head(10)  # Get top 10 countries
print("Top countries producing wines with the highest ratings:")
print(top_countries_ratings)
# the result doesn't look right, lets try looking ata data with no outliers




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
plt.figure(figsize=(10, 6))
top_10_countries.plot(kind='bar')
plt.title('Top Countries Producing Wines with the Highest Ratings')
plt.xlabel('Country')
plt.ylabel('Average Rating Points')
sns.barplot(x=top_10_countries.index, y=top_10_countries.values, palette='viridis')
plt.xticks(rotation=45)
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

# plot looks too crowded 




# previous box plot was too crowded, so plot it for top 10 grapes
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

# plot looks inconclusive, so let's consider different top 10, without outliers



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
# Z-scores are very close to zero on average, so the values in the 'price' column are centered around the mean of the distribution.(standard normal distribution)




np.std(z_scores)
# variation of the Z-scores is close to 1 which is expected in standard normal distribution.


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
