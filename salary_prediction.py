import numpy as numpy
import matplotlib.pyplot as pyplot
import pandas as pd

data_set = pd.read_csv('Salary_Data.csv')
years_data = data_set.iloc[:, :-1].values
salary_data = data_set.iloc[:, 1].values


from sklearn.model_selection import train_test_split
years_train, years_test, salary_train, salary_test = train_test_split(
                                                                years_data, 
                                                                salary_data,
                                                                test_size = 1/3,
                                                                random_state = 0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(years_train, salary_train)

salary_prediction = regressor.predict(years_test)

#Visualising the traning set result
pyplot.scatter(years_train, salary_train, color = 'black')
pyplot.plot(years_train, regressor.predict(years_train), color = 'red')
pyplot.title("Salary vs Experience (Training Set)")
pyplot.xlabel('Years of Experience')
pyplot.ylabel('Salary')
pyplot.show()

#Visualising the test set result
pyplot.scatter(years_test, salary_test, color = 'black')
pyplot.plot(years_train, regressor.predict(years_train), color = 'red')
pyplot.title("Salary vs Experience (Test Set)")
pyplot.xlabel('Years of Experience')
pyplot.ylabel('Salary')
pyplot.show()