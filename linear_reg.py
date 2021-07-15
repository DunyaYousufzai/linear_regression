import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class linear_regression:
    def __init__(self, file):
        self.file = file
    def trainig(self, a, b,c, ts):
       global x , y, data, training_x, training_y, testing_x, testing_y, linear_reg
       data = pd.read_csv(self.file)
        # all rows and column x
       x =data.iloc[:,a:b].values
       y =data.iloc[:,c].values
        #  you need 2d array
       x =x.reshape(-1,1)
       y =y.reshape(-1,1)
        # 0.3 means 30% for testing and the rest is for traninig
       training_x, testing_x,training_y,testing_y = train_test_split(x, y, test_size = ts, random_state = 0)
       linear_reg = LinearRegression()
        # start trainig with fit method
       linear_reg .fit(training_x,training_y)
    
    def predict(self, k):
       global prediction_y
       prediction_y =linear_reg.predict(testing_x)
       print(testing_y[k])
       print(prediction_y[k])
    
    def trainingPlot(self, title, labelx, labely):
        plt.scatter(training_x, training_y, color = "pink")
        plt.plot(training_x,linear_reg.predict(training_x), color = "blue")
        plt.title(title)
        plt.xlabel(labelx)
        plt.ylabel(LinearRegression)
        plt.show()

    def testingPlot(self, title, labelx, labely):
        # For testing
        plt.scatter(testing_x, testing_y, color = "green")
        plt.plot(training_x,linear_reg.predict(training_x), color = "blue")
        plt.title(title)
        plt.xlabel(labelx)
        plt.ylabel(LinearRegression)
        plt.show()

# do not forget r
linear_regression = linear_regression(r"linear_reg.csv")
linear_regression.trainig(0 , 1 ,1, 0.3)
linear_regression.predict(2)
linear_regression.trainingPlot( "Salary and Experience Trainig Plot", "Exp", "Salary" )
linear_regression.testingPlot("Salary and Experience Testing Plot", "Exp", "Salary" )