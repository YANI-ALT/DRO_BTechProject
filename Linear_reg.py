import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor, LinearRegression
import numpy as np
# norm selection for the computation
NORM="two" # 'one','two','inf'

#wasserstein radi
e_wradi=0

# delta for huber loss function
delta=1

# dataset
df=pd.read_csv('Dummy_data.csv')
x=list(df['x'])[3:10]
y=list(df['y'])[3:10]
print('X : ',x)
print('Y : ',y)
# both x and y should be of equal number one for each data point
assert(len(x)==len(y))

#datapoints
N=len(x)

def get_line_pts(w):
    x=[]
    y=[]
    for xi in range(0,5):
        yi=w[1]*xi+w[0]
        y.append(yi)
        x.append(xi)

    return x,y


def HuberRegressor_fit(x,y):
    HuberReg_obj=HuberRegressor()
    X_data=[[1,xi] for xi in x]
    fit_obj=HuberReg_obj.fit(X_data,y)

    return fit_obj

def plot_data(x,y,w,plot_name):
    #plot the data
    plt.scatter(x,y,label='data-points')

    # plot predicted line
    x_line,y_line=get_line_pts(w)
    plt.plot(x_line,y_line,color='r',label='Linear-Regression')

    # plot predicted line
    x_line,y_line=get_line_pts([1,0.5])
    plt.plot(x_line,y_line,color='g',label='Actual')

    #plot the HuberRegressor fitted line
    fit_obj=HuberRegressor_fit(x,y)
    y_predict=[]
    x_predict=[0,1,2,3,4]
    for xi in x_predict:
        y_predict.append(fit_obj.predict([[1,xi]])[0])

    plt.plot(x_predict,y_predict,color='black',label="sklearn-HuberRegressor")

    plt.legend(loc='lower right')
    plt.title(plot_name)
    plt.savefig(plot_name)

def calculate_loss(x,y,w):
    loss=0
    for i in len(x):
        loss=loss+(w[1]*x[i]+w[0]-y[i])**2

    return loss


try:

    # Create a new model
    m = gp.Model("Linear_Regression_no_regularisation")

    #hyperplane w=[w1,w2] each x=[1,x1] data point
    w0=m.addVar(vtype=GRB.CONTINUOUS,name='w0',lb=-GRB.INFINITY)
    w1=m.addVar(vtype=GRB.CONTINUOUS,name='w1',lb=-GRB.INFINITY)

    Obj_value=0
    abs_i={}
    abs_value={}
    for i in range(N):
        # Obj_value=Obj_value+(1/N)*((w1*x[i]+w0-y[i])**2)

        # if we make the loss an absolute value one
        var='abs_'+str(i)
        abs_i[var]=m.addVar(vtype=GRB.CONTINUOUS,name='abs_'+str(i))
        abs_value[i]=m.addVar(vtype=GRB.CONTINUOUS)
        m.addConstr(abs_value[i]==w1*x[i]+w0-y[i])
        m.addGenConstrAbs(abs_i[var],abs_value[i], "absconstr_"+str(i))
        Obj_value=Obj_value+(1/N)*(abs_i[var])

    m.setObjective(Obj_value, GRB.MINIMIZE)

    # Optimize model
    m.optimize()
    print()
    print("All assignments :")
    w_sol=[]
    for i,v in enumerate(m.getVars()):
        print('%s %g' % (v.varName, v.x))
        if v.varName=="w0":
            w_sol.append(v.x)
        if v.varName=="w1":
            w_sol.append(v.x)
        # if(i==3):
        #     break
    plot_name="Plots/Linear_Regression_no_regularisation.png".format(NORM,e_wradi,delta)
    plot_data(x,y,w_sol,plot_name)

    # print(m.display())
    m.write('Linear_Regression_no_regularisation.lp')


    # for v in m.getVars():
    #     print('%s %g' % (v.varName, v.x))


except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ': ' + str(e))

except AttributeError:
    print('Encountered an attribute error')
