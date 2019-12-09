import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


from pylab import subplot

raw_movie_data = pd.read_csv("C:\\Users\\C740129\\OneDrive - Standard Bank\\certification\\Ai Engineer\\Data Science with Pandas C-1\\Projects submission for assesment\\Projects for Submission\\Project4_Movielens\\movies.dat",delimiter="::",names = ["MovieID", "Title", "Genres"])
#print(raw_movie_data.head())
raw_user_data = pd.read_csv("C:\\Users\\C740129\\OneDrive - Standard Bank\\certification\\Ai Engineer\\Data Science with Pandas C-1\\Projects submission for assesment\\Projects for Submission\\Project4_Movielens\\users.dat",delimiter="::",names = ["UserID", "Gender", "Age", "Occupation","Zip-code"])
#print(raw_user_data.head())
raw_rating_data = pd.read_csv("C:\\Users\\C740129\\OneDrive - Standard Bank\\certification\\Ai Engineer\\Data Science with Pandas C-1\\Projects submission for assesment\\Projects for Submission\\Project4_Movielens\\ratings.dat",delimiter="::",names = ["UserID", "MovieID", "Rating", "Timestamp"])
#print(raw_rating_data.head())

#Data Wrangling 
movie_rating_data = pd.merge(raw_movie_data,raw_rating_data,on='MovieID', how='outer')
print("--------movie_rating_data----------")
print(movie_rating_data.head())

movie_rating_user_data = pd.merge(movie_rating_data,raw_user_data,on='UserID',how='outer')
print("--------movie_rating_user_data----------")
print(movie_rating_user_data.head())
#Visualize user Age distribution
user_age = raw_user_data['Age']

subplot(2,2,1)
plt.hist(user_age,color = 'blue', edgecolor = 'black',bins=[1,18,25,35,45,50,56])
plt.xticks([1,18,25,35,45,50,56])
plt.xlim([1,56])
plt.xlabel('Age')
plt.ylabel('Counts')
plt.title('Age Distribution')


# Visualize overall rating by users

rating_data = raw_rating_data['Rating']
subplot(2,2,2)
plt.hist(rating_data,color = 'yellow', edgecolor = 'black',bins=5)
plt.xticks([1,2,3,4,5])
plt.xlim([1,5])
plt.xlabel('Rating')
plt.ylabel('Counts')
plt.title('Rating Distribution')

# Find and visualize the user rating of the movie “Toy Story”

toy_story_rating_data = movie_rating_data.query('Title=="Toy Story (1995)"')
print(toy_story_rating_data.head())
subplot(2,2,3)
plt.hist(toy_story_rating_data['Rating'],color = 'red', edgecolor = 'black',bins=5)
plt.xticks([1,2,3,4,5])
plt.xlim([1,5])
plt.xlabel('Rating')
plt.ylabel('Counts')
plt.title('Toy Story (1995) Rating Distribution')

# Find and visualize the viewership of the movie “Toy Story” by age group
toy_st_movie_rating_user_grp_data = movie_rating_user_data.query('Title=="Toy Story (1995)"')
#print("************ toy_st_movie_rating_user_grp_data ************")
#for x in toy_st_movie_rating_user_grp_data:
#    print(x)

subplot(2,2,4)
#toy_st_movie_rating_user_grp_data.columns.droplevel(level=0)
#toy_st_movie_rating_user_grp_data.rename(columns = ['Age','Count'],inplace=True)
#print(toy_st_movie_rating_user_grp_data)
plt.hist(toy_st_movie_rating_user_grp_data['Age'], edgecolor = 'black',bins=[1,18,25,35,45,50,56])
plt.xticks([1,18,25,35,45,50,56])
plt.xlim([1,56])
plt.xlabel('Rating')
plt.ylabel('Counts')
plt.title('Toy Story (1995) Rating Group By Age Distribution')

plt.show()
# Find and visualize the top 25 movies by viewership rating
subplot(2,1,1)
#top_25_movie_data = movie_rating_data.query('Rating==5').groupby(["Title","Rating"]).agg({"Rating": "count"})
top_25_movie_data = movie_rating_data.query('Rating==5').groupby(["Title","Rating"]).size().reset_index(name='count')
print("******* top_25_movie_data *************")
print(top_25_movie_data)
top_25_movie = pd.DataFrame(top_25_movie_data.sort_values(by=['count'], ascending=False).head(25))
print(top_25_movie)
#y=top_25_movie['Title']
#x=top_25_movie['Rating']

plt.barh(top_25_movie['Title'],top_25_movie['count'])
plt.title('The top 25 movies by viewership rating')



# •	Find the rating for a particular user of user id = 2696
subplot(2,1,2)
rating_by_2696 = movie_rating_data.query('UserID==2696')
print(rating_by_2696)

plt.barh(rating_by_2696['Title'],rating_by_2696['Rating'])
plt.title('Visualize  rating for a particular user of user id = 2696')

plt.show()

#	Perform machine learning on first 500 extracted records
ml_movie_data = movie_rating_user_data.head(500)
ml_movie_data = ml_movie_data.drop('Title', axis=1).drop('Genres', axis=1).drop('Timestamp', axis=1).drop('Gender',axis=1).drop('Zip-code',axis=1).drop('UserID',axis=1)
print("****ML movie data 500 *******")
print(ml_movie_data)

X = ml_movie_data.drop('Rating',axis = 1)
Y = ml_movie_data[['Rating']]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=1)

regression_model = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
regression_model.fit(X_train, y_train)

for idx, col_name in enumerate(X_train.columns):
    print("The coefficient for {} is {}".format(col_name, regression_model.coef_[0][idx]))

intercept = regression_model.intercept_[0]
print("The intercept for our model is {}".format(intercept))

ml_score = regression_model.score(X_test, y_test)
print("The score for our model is {}".format(ml_score))

y_predict = regression_model.predict(X_test)



print("***** Actual*****")
print(y_test)
print("**** Predicted *****")
print(y_predict)

print('Mean Absolute Error:', mean_absolute_error(y_test, y_predict))  
print('Mean Squared Error:', mean_squared_error(y_test, y_predict))  
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_predict))) 

ml_movie_data.hist()
plt.show()
