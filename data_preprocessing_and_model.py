import pandas as pd 
import numpy as np 
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score
from sklearn import metrics

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

train['source']='train'
test['source']='test'
data = pd.concat([train, test],ignore_index=True, sort = True)
print (train.shape, test.shape, data.shape)
print("\n")

print("Checking null values in each Columns...............")
print(data.apply(lambda x: sum(x.isnull())))
print("\n")

print("Analyzing the numerical columns....................")
print(data.describe())
print("\n")

print('Reviewing Nominal Categorical Variables: Checking number of Unique values..............')
print(data.apply(lambda x: len(x.unique())))
print("\n")

print('Filtering categorical variables.............')
categorical_columns = [x for x in data.dtypes.index if data.dtypes[x]=='object']
#Exclude ID cols and source
categorical_columns = [x for x in categorical_columns if x not in ['Item_Identifier','Outlet_Identifier','source']]
#Print frequency of categories
for col in categorical_columns:
    print ('\nFrequency of Categories for varible %s'%col)
    print (data[col].value_counts())
print("\n")

print("Imputing Missing data in Item_Weight Column with average of Item_Weight..................")
#Determine the average weight per item:
data['Item_Weight'].replace('',np.nan)
data['Item_Weight'] = data.groupby('Item_Identifier').transform(lambda x: x.fillna(x.mean()))
#print(data['Item_Weight'])
print("\n")

print("Imputing missing data in Outlet_Size, with the mode size for each Outlet_Type...........................")
data['Outlet_Size'].replace('',np.nan)
data['Outlet_Size'] = data.groupby('Outlet_Type').transform(lambda x: x.fillna(x.mode()))
print("\n")

print('Imputing Item_Visibilty data with the average of Item_Visibilty per Product..................')
map_table = data.groupby('Item_Identifier').mean()
data.loc[:,'Item_Visibility'] = data.loc[:,'Item_Identifier'].map(map_table.to_dict()['Item_Visibility'])

print(data['Item_Visibility'])
"""visibility_avg = data.pivot_table(values = 'Item_Visibility', index = 'Item_Identifier')
data['Item_Visibility'].replace('',np.nan)
#imputing  0 values with mean visibility of products
miss_bool = (data['Item_Visibility'] == 0)
data.loc[miss_bool,'Item_Visibility'] = data.loc[miss_bool,'Item_Identifier'].apply(lambda x: visibility_avg[x])"""
print("\n")


print("Combining and creating broad categories for Item_Type based on their Item_Identifier..........")
#Get the first two characters of ID:
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
#Renaming labels
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD': 'Food','NC':'Non-Consumable','DR': 'Drinks'})
print('Item counts in Item_Type_Combined: ')
print(data['Item_Type_Combined'].value_counts())
print('\n')

print('Calculating ages of stores based on Outlet_Establishment_Year......................')
data['Outlet_Years'] = 2019 - data['Outlet_Establishment_Year']
print('Discribtion of Outlet_Years: ')
print(data['Outlet_Years'])
print('\n')

print('Resolving Typo in Item_Fat_Content.....................')
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF': 'Low Fat', 'reg': 'Regular', 'low fat': 'Low Fat'})
print('Item_Fat_Content value counts after: ')
print(data['Item_Fat_Content'])
print('\n')

print('Marking Item_Fat_Content of Non-Consumable items separately: ')
data.loc[data['Item_Type_Combined'] == 'Non-Consumable','Item_Fat_Content'] = "Non-Edible"
print('Item_Fat_Content value counts after: ')
print(data['Item_Fat_Content'].value_counts())
print('\n')

#=================================================================================================================================
print('Starting One hot encoding process...................................')
encoder = LabelEncoder()
data['Outlet'] = encoder.fit_transform(data['Outlet_Identifier'])
need_to_encode = ['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Type', 'Outlet_Size', 'Item_Type_Combined', 'Outlet']
for i in need_to_encode:
	data[i] = encoder.fit_transform(data[i])
data = pd.get_dummies(data, columns = need_to_encode)
print('Finished One hot encoding..........................')

#print(data.dtypes)
data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)
train = data.loc[data['source']=="train"]
test = data.loc[data['source']=="test"]

#Drop unnecessary columns:
test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
train.drop(['source'],axis=1,inplace=True)

#Export files as modified versions:
train.to_csv("train_modified.csv",index=False)
test.to_csv("test_modified.csv",index=False)

#====================================================================================================================================
target = 'Item_Outlet_Sales'
id_cols = ['Item_Identifier','Outlet_Identifier']

def modelfit(algorithm, d_train, d_test, predictors, target, id_cols, filename):
	algorithm.fit(d_train[predictors], d_train[target])

	#Predict training dataset
	d_train_predictions = algorithm.predict(d_train[predictors])

	#crosss_validation
	score = cross_val_score(algorithm, d_train[predictors], d_train[target], cv = 20, scoring = 'neg_mean_squared_error')
	score = np.sqrt(np.abs(score))

	print('\nPrint Model Report')
	print(("RMSE : %.4g" )% np.sqrt(metrics.mean_squared_error(d_train[target].values, d_train_predictions)))
	print(('crosss_validation_score: Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g') % (np.mean(score),np.std(score),np.min(score),np.max(score)))

	#Predict on testing data:
	d_test[target] = algorithm.predict(d_test[predictors])

	id_cols.append(target)
	submission = pd.DataFrame({ x: d_test[x] for x in id_cols})
	submission.to_csv(filename, index=False)

predictors = [x for x in train.columns if x not in [target]+id_cols]

lin_Reg = LinearRegression(normalize=True)
modelfit(lin_Reg, train, test, predictors, target, id_cols, 'LinearRegression_Predictions.csv')


