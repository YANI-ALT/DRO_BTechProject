import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR
from sklearn import datasets
import numpy as np

# norm selection for the computation
NORM="two" # 'one','two','inf'

#wasserstein radi
e_wradi=1e-1

# cost of switching a label
kappa=1

# decide how many points to use
EXP_SIZE=10

# dataset
def get_iris_dataset(to_plot=True):

    # using the iris dataset
    iris=datasets.load_iris()
    X = iris.data[:, :2] # we only take the first two features.
    X=np.c_[np.ones(X.shape[0]), X]

    y = iris.target

    mask=y!=1  # take the labels 0,2

    X=X[mask]
    X=X[45:55]
    y=y[mask]
    y=y[45:55]

    y[y==0]=-1
    y[y==2]=1

    print(y)
    if(to_plot):
        plt.scatter(X[:, 1], X[:, 2], c=y, cmap=plt.cm.Set1, edgecolor="k")
        plt.xlabel("Sepal length")
        plt.ylabel("Sepal width")
        plt.savefig("Plots/DataUsed_Iris_dataset.png")

    return X,y




def get_conic_info(X,y,col):
    # the conic defined for the data C1x+c2y<=d
    # C1=[[1,0],[0,1]]
    # c2=[1,1]
    # d=[2,2]
    C1=[]
    c2=[]
    d=[]

    d_value=(max(X.flatten())+max(y))

    for i in range(col):
        row_vector=[0]*col
        row_vector[i]=1
        C1.append(row_vector)
        c2.append(1)
        d.append(d_value)
    print("===== CONIC information ======")
    print("C1 : ",C1)
    print("c2 : ",c2)
    print("d : ",d)
    print()

    return C1,c2,d

def get_line_pts(w):
    x=[]
    y=[]
    for xi in range(4,10):
        yi=(-w[0]*xi)/w[1]
        y.append(yi)
        x.append(xi)

    return x,y

def LinearSVR_fit(x,y):
    linSVR=LinearSVR(epsilon=epsilon,loss='epsilon_insensitive')
    X_data=[[1,xi] for xi in x]
    fit_obj=linSVR.fit(X_data,y)

    return fit_obj

def plot_data(x,y,w,plot_name):
    #plot the data
    plt.scatter(x[:, 1], x[:, 2], c=y, cmap=plt.cm.Set1, edgecolor="k",label="data-points")
    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")

    # plot predicted line
    x_line,y_line=get_line_pts(w)
    plt.plot(x_line,y_line,color='r',label='DRO-SVM')
    #
    # # plot predicted line
    # x_line,y_line=get_line_pts([1,0.5])
    # plt.plot(x_line,y_line,color='g',label='Actual')
    #
    # #plot the LinearSVR fitted line
    # fit_obj=LinearSVR_fit(x,y)
    # y_predict=[]
    # x_predict=[0,1,2,3,4]
    # for xi in x_predict:
    #     y_predict.append(fit_obj.predict([[1,xi]])[0])
    #
    # plt.plot(x_predict,y_predict,linestyle='dashed',color='black',label="LinearSVR")

    plt.legend(loc='lower right')
    plt.title(plot_name)
    plt.savefig(plot_name)

# initialize all the data
x,y=get_iris_dataset()
FEATURES=x.shape[1] # we append the 1 constant also to the feature space
print("FEATURES : ",FEATURES)
print("X : ",x)
print("y : ",y)

C1,c2,d=get_conic_info(x,y,FEATURES)
N=x.shape[0]
try:

    # Create a new model
    m = gp.Model("Simple_SVM")

    col=FEATURES

    w={}
    #hyperplane w=[w1,w2,..wn] each x=[1,x1...wn] data point
    for i in range(col-1):
        var_name="w"+str(i)
        w[i]=m.addVar(vtype=GRB.CONTINUOUS,name=var_name,lb=-GRB.INFINITY)

    print("Setting objective")
    # Create variables
    Obj_value=0.5*(w[1]*w[1]+w[0]*w[0])

    m.setObjective(Obj_value, GRB.MINIMIZE)
    print("----Adding constraints----")
    for i in range(N):
        y_wx=gp.LinExpr(x[i][1:],w.values())
        m.addConstr(1-(y[i]*y_wx)<=0)


    #Add contraint with data points

    # Optimize model
    m.optimize()
    # print(m.display())
    m.write('Simple_SVM.lp')
    print()
    print("All assignments :")
    w_sol=[]
    for i,v in enumerate(m.getVars()):
        print('%s %g' % (v.varName, v.x))
        if v.varName=="w0":
            w_sol.append(v.x)
        if v.varName=="w1":
            w_sol.append(v.x)
        if v.varName=="w2":
            w_sol.append(v.x)
        if(i==3):
            print("d: ",d)
            break


    plot_name="Plots/Simple_SVM_plot_NORM_{}_wassradi_{}_kappa_{}.png".format(NORM,e_wradi,kappa)
    plot_data(x,y,w_sol,plot_name)




    # for v in m.getVars():
    #     print('%s %g' % (v.varName, v.x))


except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ': ' + str(e))

except AttributeError:
    print('Encountered an attribute error')
