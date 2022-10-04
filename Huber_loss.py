import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor, LinearRegression
import numpy as np
# norm selection for the computation
NORM="two" # 'one','two','inf'

#wasserstein radi
e_wradi=100

# delta for huber loss function
delta=1.57

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
    plt.plot(x_line,y_line,color='r',label='DRO-HuberLoss')

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



try:

    # Create a new model
    m = gp.Model("Huber_loss")
    # if(NORM=='two'):
    #     m.params.NonConvex = 2

    z_i={}
    abs_i={}
    abs_value={}
    #hyperplane w=[w1,w2] each x=[1,x1] data point
    w0=m.addVar(vtype=GRB.CONTINUOUS,name='w0',lb=-GRB.INFINITY)
    w1=m.addVar(vtype=GRB.CONTINUOUS,name='w1',lb=-GRB.INFINITY)

    if NORM=="inf" or NORM=="one":
        # the dual norm will be one-norm =summation of abs(xi)
        abs_w0=m.addVar(vtype=GRB.CONTINUOUS,name='|w0|')
        abs_w1=m.addVar(vtype=GRB.CONTINUOUS,name='|w1|')


    # create zi and other variables
    for i in range(N):
        var_name='z_'+str(i)
        z_i[var_name]=m.addVar(vtype=GRB.CONTINUOUS,name=var_name)

        # create a variable for the absolute value in the objective function
        abs_i['abs_'+str(i)]=m.addVar(vtype=GRB.CONTINUOUS,name='abs_'+str(i))
        abs_value[i]=m.addVar(vtype=GRB.CONTINUOUS)

    Obj_value=0
    Norm=0
    # Add constraint
    if NORM=="two":
        # Set objective
        Norm=e_wradi*delta*(w0**2+w1**2+1)
        # m.addConstr(==norm_value*norm_value,'norm-constraint')
    elif NORM=="inf":
        #the corresponding dual norm is a one norm
        m.addGenConstrAbs(w0,abs_w0, "absconstr_w0")
        m.addGenConstrAbs(w1,abs_w1, "absconstr_w1")

        Norm=e_wradi*delta*(abs_w1+abs_w0+1)
        # m.addConstr(==norm_value,'norm-constraint')
    elif NORM=="one":
        m.addGenConstrAbs(w0,abs_w0, "absconstr_w0")
        m.addGenConstrAbs(w1,abs_w1, "absconstr_w1")
        norm_value=m.addVar(vtype=GRB.CONTINUOUS,name="norm(w,-1)")
        m.addGenConstrMax(norm_value,[abs_w0,abs_w1],1)

        Norm=e_wradi*delta*(norm_value)

    #absolute value constraints
    for i in range(N):
        #|<w,xi>-yi-zi|
        var='abs_{}'.format(i)
        exp=gp.LinExpr([1,x[i],-1],[w0,w1,z_i['z_'+str(i)]])
        exp.add(-y[i])

        m.addConstr(abs_value[i]==exp)
        m.addGenConstrAbs(abs_i[var],abs_value[i], "absconstr_"+str(i))


    for i in range(N):
        zi="z_"+str(i)
        Obj_value=Obj_value+(1/N)*0.5*(z_i[zi]**2)+gp.LinExpr([(1/N)*delta],[abs_i['abs_'+str(i)]])

    Obj_value=Obj_value+Norm
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
    plot_name="Plots/Huber_loss_fig_NORM_{}_wassradi_{}_delta_{}.png".format(NORM,e_wradi,delta)
    plot_data(x,y,w_sol,plot_name)
    # print(m.display())
    m.write('Huber_loss.lp')


    # for v in m.getVars():
    #     print('%s %g' % (v.varName, v.x))


except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ': ' + str(e))

except AttributeError:
    print('Encountered an attribute error')
