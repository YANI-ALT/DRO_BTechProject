import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def noise():
    return np.random.normal(scale=0.2)

def regression_ground_truth(x):
    m=0.5
    c=1
    return m*x+c


def generate_data(N):
    dataset=[]

    x_data=[]
    y_data=[]
    for i in range(N):
        x=np.random.uniform(0,2)
        y=regression_ground_truth(x)+noise()
        x_data.append(x)
        y_data.append(y)
        dataset.append((x,y))
    df=pd.DataFrame({'x':x_data,'y':y_data})
    df.to_csv('Dummy_data.csv')
    return dataset

def plot_data(dataset):
    x=[pt[0] for pt in dataset]
    y=[pt[1] for pt in dataset]
    plt.scatter(x,y)
    plt.savefig("fig.png")


dataset=generate_data(40)
plot_data(dataset)
