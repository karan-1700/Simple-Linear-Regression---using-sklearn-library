# # SIMPLE LINEAR REGRESSION 
# ### USING SKLEARN LIBRARY
#    USING PYTHON

# Using the dataset -- Years of Experience V/S Salary

# importing Libraries
import numpy as np      # for working with arrays
import pandas as pd     # for dealing with DataFrames
import matplotlib.pyplot as plt     # for Data Visualisation


# loading the Training Dataset into Pandas.DataFrame
df_train = pd.read_csv("G://machine learning//SalaryData_Train.csv")


df_train.head(10) # prints the first 10 Rows from the dataset


# Printing the Shape of Dataset
print(df_train.shape)


# Information about the Training DataFrame
df_train.info()

# Since there are no null values, we can proceed further



# Visualising the data
plt.scatter(df_train["years"],df_train["salary"],color='b')   # creates a Scatter Plot
plt.xlabel("Years of Experience")
plt.ylabel("Salary (Rs)")
plt.legend(["Data Points"])
plt.title("Years of Experience VS Salary Plot")
plt.grid()
plt.show()



years = np.array(df_train["years"])  # independent variable ... called  Feature
salary = np.array(df_train["salary"])  # dependent variable ... called  Label



# sampling the data
from sklearn.model_selection import train_test_split

# This splits the dataset into training (70% of dataset)
# and testing data (30% of dataset).

x_train, x_test, y_train, y_test = train_test_split(years, salary, test_size=0.3, random_state=0, shuffle=True)


print("shape of train dataset ",x_train.shape)
print("shape of test dataset ",x_test.shape)

print("\ndata-type of x_train :",type(x_train))



from sklearn.linear_model import LinearRegression
# Create an object of Linear Regression module
reg = LinearRegression()



# Fitting our Training Dataset into the Linear Regression model

#####  reg.fit(x_train, y_train)  # --->> ERROR

# However, we get error as x_train, y_train are 1-Dimensional arrays.
# But, reg.fit() expects  2-Dimensional arrays.

# Therefore, we reshape x_train, y_train.
x_train = x_train.reshape(-1,1)
y_train = y_train.reshape(-1,1)



# Fitting our Training Dataset into the Linear Regression model
reg.fit(x_train, y_train)


# predicting the y_train for the model
predicted_y = reg.predict(x_train)


# Visualising the Regression Line and train_data
plt.plot(x_train, predicted_y, color='g')
plt.scatter(x_train, y_train, color='b')
plt.xlabel('Years of Experience')
plt.ylabel("Salary (in Rs)")
plt.legend(["Regression Line","train_data"])
plt.title("Years of Experience V/S Salary")
plt.grid()
plt.show()



# We can get SLOPE (m) of the Regression Line using reg.coef_
print("m = ", reg.coef_)

# We can get Y-INTERCEPT of the Regression Line using reg.intercept_
print("c = ",reg.intercept_)



# Checking the Accuracy of the model using reg.score().
# It returns the coefficient of determination R^2 of the prediction.

print("Accuracy of Train dataset = ",reg.score(x_train,y_train))

# The best possible score is 1.0 .
# It can even be negative (for arbitrarily worse model).



# Testing the Linear Regression model on test dataset
df_test = pd.read_csv("G://machine learning//SalaryData_Test.csv")

# Checking the shape of the testing dataset
print("shape of df_test : ",df_test.shape)



# Checking the dataset for any Null values
df_test.info()



# Counting the total number of Null values in each Column
df_test.isnull().sum()


x_test = np.array(df_test["years"])

x_test = x_test.reshape(-1,1)


# Predicting the Salary for x_test i.e. testing dataset
y_pred_test = reg.predict(x_test)

# and copying the predicted values into the "salary" column
df_test["salary"] = y_pred_test



# Print the final DataFrame after predicting the "salary"
print(df_test)


# Visualising the data
plt.plot(x_test, y_pred_test, color='b')
plt.scatter(df_test["years"], df_test["salary"], color='r')
plt.xlabel("Years of Experience")
plt.ylabel("Salary (in Rs)")
plt.legend(["Regression Line", "test_data"])
plt.grid()
plt.show()


# ### Made By : Karansinh Padhiar
