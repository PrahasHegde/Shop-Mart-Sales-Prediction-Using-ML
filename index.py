#imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#load the datset
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')

print(train.head())

print(train.shape)
print(test.shape)

#getting null values from train & test dataset
print(train.isnull().sum())

print(test.isnull().sum())

#let’s know the number of unique values present in each of the columns
for i in train.columns:
  print(f"{i} : {train[i].nunique()}")

#data type of each column
for i in train.columns:
  print(f"{i} : {train[i].dtype}")

#creating 2 list for numerical and categorical data
cat_columns = []
num_columns = []

for i in train.columns:
  if train[i].dtype == object:
    cat_columns.append(i)
  else:
    num_columns.append(i)

# print(cat_columns)
# print(num_columns)

"""  fill the null values in the “Item_Weight” with its mean value. 
And I am going to fill the null values in the “Outlet_Size” with its mode value.
 (Because “Item_Weight” is a numerical column whereas “Outlet_Size” is a categorical column) """

#using mean for Item_Weight
train['Item_Weight'].fillna(train['Item_Weight'].mean(), inplace=True)
test['Item_Weight'].fillna(train['Item_Weight'].mean(), inplace=True)
# print(train.isnull().sum())
#print(test.isnull().sum())

#using mode for outlet_size
train['Outlet_Size'].fillna(train['Outlet_Size'].mode()[0], inplace=True)
test['Outlet_Size'].fillna(train['Outlet_Size'].mode()[0], inplace=True)
# print(train.isnull().sum())
#print(test.isnull().sum())


#Visualizing Categorical Columns
print(train[cat_columns].head())

print(train[cat_columns].nunique()) #identify unique categorical data

#Item_Fat_Content columns
print(train['Item_Fat_Content'].value_counts())

#in dataset low fat, Low Fat, LF are same and reg ,Regualr are same
train['Item_Fat_Content'].replace({'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'}, inplace=True)
test['Item_Fat_Content'].replace({'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'}, inplace=True)

sns.barplot(train['Item_Fat_Content'].value_counts()) #barplot for visualization between different fat_content
plt.show()

#visualizing outlet_type
plt.figure(figsize=(10, 5))
sns.barplot(train['Outlet_Type'].value_counts())
plt.show()

#visualizing outlet_location_type
plt.figure(figsize=(10, 5))
sns.barplot(train['Outlet_Location_Type'].value_counts())
plt.show()

#visualizing item_type
plt.figure(figsize=(25, 5))
sns.barplot(train['Item_Type'].value_counts())
plt.show()

#visualizing outlet_indentifier
plt.figure(figsize=(10, 5))
sns.barplot(train['Outlet_Identifier'].value_counts())
plt.show()

#visualizing outlet_size
plt.figure(figsize=(10, 5))
sns.barplot(train['Outlet_Size'].value_counts())
plt.show()

########################################################################################

#Visualizing Numerical Columns

#item_weight
plt.figure(figsize=(5, 5))
sns.histplot(train['Item_Weight'], kde=True)
plt.show()

#item_visiblity
plt.figure(figsize=(5, 5))
sns.histplot(train['Item_Visibility'], kde=True)
plt.show()

#item_MRP
plt.figure(figsize=(5, 5))
sns.histplot(train['Item_MRP'], kde=True)
plt.show()

#establishment_year
plt.figure(figsize=(5, 5))
sns.countplot(x='Outlet_Establishment_Year', data=train)
plt.show()

#outlet_sales
plt.figure(figsize=(5, 5))
sns.histplot(train['Item_Outlet_Sales'], kde=True)
plt.show()


#Handling Categorical Values
print(train[cat_columns].nunique())

#Instead of “FDA15” I am going to change it to just “FD” which means Food. Similarly, Instead of “DRC01" I am going to change it to just ‘DR” which means Drinks
train['Item_Identifier'] = train['Item_Identifier'].str[:2]
test['Item_Identifier'] = test['Item_Identifier'].str[:2]

"""means they have a certain order. For example, if you take grade, we know A is first, B is second, 
and C is third it’s an order those types of columns are also knowns as ordinal columns).
The ordinal columns are ‘Item_Fat_Content’, ‘Outlet_Size’, and ‘Outlet_Location_Type’. 
The rest of the columns are nominal columns because they don’t any ordering in them."""

#convert categorical values to numeric using encoders
#apply the Ordinal Encoder technique to ordinal categorical columns. And One-Hot Encoding for nominal categorical columns
ordinal_cat_columns = ['Item_Fat_Content', 'Outlet_Size', 'Outlet_Location_Type']
nominal_cat_columns = ['Item_Identifier', 'Item_Type', 'Outlet_Identifier', 'Outlet_Type']


# One-Hot Encoding using get_dummies()
train = pd.get_dummies(train, columns=nominal_cat_columns)
test = pd.get_dummies(test, columns=nominal_cat_columns) 

# Initialize the OrdinalEncoder
ordinal_encoder = OrdinalEncoder()

# Define the ordinal columns
ordinal_cols = ['Item_Fat_Content', 'Outlet_Size', 'Outlet_Location_Type']

# Fit and transform the ordinal columns
train[ordinal_cols] = ordinal_encoder.fit_transform(train[ordinal_cols])
test[ordinal_cols] = ordinal_encoder.fit_transform(test[ordinal_cols])

print(train.head())


#splitting features and labels
y = train['Item_Outlet_Sales']
X = train.drop(columns=['Item_Outlet_Sales'])

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=786)

#Building Model

model = LinearRegression()
# Fit the model to the training data
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#Evaluating The Model (MAE,MSE,R2S)

model_r2_score = r2_score(y_test, y_pred=y_pred)
model_mae_score = mean_absolute_error(y_test, y_pred=y_pred)
model_mse_score = mean_squared_error(y_test, y_pred=y_pred)

print(model_r2_score, model_mae_score, model_mse_score)


plt.scatter(y_test, y_pred)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.show()