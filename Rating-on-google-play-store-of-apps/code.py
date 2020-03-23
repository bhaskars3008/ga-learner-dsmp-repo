# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv(path)

# Plot histogram for rating
data.plot.hist(by = "Rating", bins = 50)
data = data[data['Rating'] <= 5]
data.plot.hist(data['Rating'], bins = 50)

#Code starts here


#Code ends here


# --------------
# code starts here
# Check for null values in all
total_null = data.isnull().sum()
print(total_null)
# percentage of null values
percent_null = (total_null/data.isnull().count())
print(percent_null)
# Concatenate both the values
missing_data = pd.concat([total_null, percent_null],axis = 1,keys =['Total','Percent'])
print(missing_data)
# Drop the null values
data = data.dropna()
# Create new variables
total_null_1 = data.isnull().sum()
print(total_null_1)
percent_null_1 = (total_null_1/data.isnull().count())
print(percent_null_1)
missing_data_1 = pd.concat([total_null_1,percent_null_1],axis=1,keys=['Total',"Percent"])
print(missing_data_1)


# code ends here


# --------------

#Code starts here
x = sns.catplot(x = "Category", y = "Rating", data = data, kind ="box", height = 109)
x.set_xticklabels(rotation=90)
x.set_titles('Rating vs Category')


#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here
# Number of Installs
data[['Installs']].count()
# replace + by space
data['Installs']=data['Installs'].str.replace('+','')
# replace , by space
data['Installs']=data['Installs'].str.replace(',','')
# convert the installs column datatype to int 
data['Installs'] = data['Installs'].astype(int)
#Label Encoder used on installs
le = LabelEncoder()
data['Installs'] = le.fit_transform(data['Installs'])
# Plotting using seaborn
x = sns.regplot(x="Installs", y='Rating', data=data)
plt.title('Rating vs Installs',size = 20)
#Code ends here



# --------------
#Code starts here
price = data['Price'].count()
#remove $ sign
data['Price'] = data['Price'].str.strip('$')
# convert the column values in integer type
data['Price'] = data['Price'].astype(float)
# plot the data
x=sns.regplot(x="Price",y = 'Rating',data = data)
#x.set_titles('Rating vs Price [Regplot]')
#Code ends here


# --------------

#Code starts here
# unique values
print(len(data['Genres'].unique()),'genres')
## Splitting the column to include only the first genre
data['Genres'] = data['Genres'].str.split(';').str[0]
#Grouping genres and rating
gr_mean=data[['Genres','Rating']].groupby(['Genres'], as_index=False).mean()
print(gr_mean.describe())
#Sorting the grouped dataframe by rating
gr_mean=gr_mean.sort_values('Rating')
print(gr_mean.head(1))
print(gr_mean.tail(1))
#Code ends here


# --------------

#Code starts here
#Converting the column into datetime format
data['Last Updated'] = pd.to_datetime(data['Last Updated'])
#Creating new column having Last Updated in days
data['Last Updated Days'] = (data['Last Updated'].max()-data['Last Updated']).dt.days
# Setting the size of the figure
plt.figure(figsize = (10, 10))
#Plotting a regression plot between rating and last updated
sns.regplot(x='Last Updated Days', y='Rating', color = 'lightpink', data=data)
#Setting the title of the plot
plt.title('Rating vs Last Updated [Regplot]', size = 20)
#Code ends here


