#uber fare predection
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

# Understand the type of data.
all_data = pd.read_csv('data/train.csv', nrows=10000).dropna()
columns = all_data.columns
features = ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count']

# Count the null values existing in columns.
null_data_count = all_data.astype(bool).sum(axis=0)

# Remove the null value rows in the target variable.
clean_data = all_data[(all_data[['pickup_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count']] != 0).all(axis=1)]
print(clean_data.describe)
#Features
x = clean_data.loc[:, features].values
# Separating out the target
y = clean_data.loc[:,['fare_amount']].values

# Perform train test split
#train_test_split()
# Identify the output variable.
#fare_amount
# Identify the factors which affect the output variable.
# Standardizing the features
scaler = StandardScaler()

x = StandardScaler().fit_transform(x)
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)
x_pca_df = pd.DataFrame(x_pca,columns = ['principal component 1', 'principal component 2'])
x_pca_df.head()
x_pca_df['fare_amount'] = y

## Plot PCA
fig = plt.figure()
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1') 
ax.set_ylabel('Principal Component 2') 
ax.set_title('2 component PCA') 


indicesToKeep = x_pca_df['fare_amount']
ax.scatter(indicesToKeep,x_pca_df.loc[indicesToKeep, 'principal component 1'] , c = 'r' )
ax.scatter(indicesToKeep,x_pca_df.loc[indicesToKeep, 'principal component 2'] , c = 'g' )

ax.legend(features)
ax.grid()
plt.show()

# Check if there are any biases in your dataset.



# Predict the accuracy using regression models.
# Check and compare the accuracy of the different models.