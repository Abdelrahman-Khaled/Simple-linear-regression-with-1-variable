import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import  scatter_matrix
import csv
def Compute_error(population,theta,actual_ans,size): #cost function

    return (1/2*size)*(np.dot(population,theta)-acatual_ans).T.dot(np.dot(population,theta)-acatual_ans)

def Gradient_descent(population,theta,actual_ans,size,alfa,number_of_iterations):
            prediction=0
            for i in range (number_of_iterations):
                prediction=np.dot(population,theta)
                print(prediction)
                theta = theta - population.T.dot(prediction - acatual_ans)*alfa*(1/size)

            return theta,prediction;
# reading data
with open ('data.csv') as csvfile :
    data=list(csv.DictReader(csvfile))
theta=np.zeros((2,1))
m=len(data)
population=np.ones((len(data),2))
population[:,1]=np.array([d['population'] for d in data ])
acatual_ans=np.zeros((m,1))
acatual_ans[:,0]=np.array([d['profit'] for d in data])

plt.scatter(population[:,1],acatual_ans[:,0])
theta,prediction=Gradient_descent(population,theta,acatual_ans,m,.01,3000)
print(theta)

plt.plot(population[:,1],prediction[:,0])
plt.show()