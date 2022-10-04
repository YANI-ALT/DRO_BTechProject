import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
 # norm selection for the computation
NORM="two" # 'one','two','inf'

#wasserstein radi
e_wradi=1

# cost of switching a label
kappa=0.01

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
    X=X[48:52]
    y=y[mask]
    y=y[48:52]

    y[y==0]=-1
    y[y==2]=1

    print(y)
    if(to_plot):
        plt.scatter(X[:, 1], X[:, 2], c=y, cmap=plt.cm.Set1, edgecolor="k")
        plt.xlabel("Sepal length")
        plt.ylabel("Sepal width")
        plt.savefig("Plots/DataUsed_Iris_dataset.png")

    return X,y

x,y=get_iris_dataset()
FEATURES=x.shape[1] # we append the 1 constant also to the feature space
print("FEATURES : ",FEATURES)
print("X : ",x)
print("y : ",y)


N=x.shape[0]

# Create a new model
m = gp.Model("Logistic Regression")


col=FEATURES
s_i={}
w={}
#hyperplane w=[w1,w2,..wn] each x=[1,x1...wn] data point

if NORM=="two":
    m.params.NonConvex = 2

for i in range(col):
    var_name="w_"+str(i)
    w[var_name]=m.addVar(vtype=GRB.CONTINUOUS,name=var_name, lb = -GRB.INFINITY)

#lambda variable
lambda_var=m.addVar(vtype=GRB.CONTINUOUS,name='Lambda')
# Create variables

# create si
for i in range(N):
    var_name='s_'+str(i)
    s_i[var_name]=m.addVar(vtype=GRB.CONTINUOUS,name=var_name,lb=-GRB.INFINITY)

Obj_value=lambda_var*e_wradi

for si in s_i.keys():
    Obj_value=Obj_value+(1.0/N)*s_i[si]

m.setObjective(Obj_value, GRB.MINIMIZE)




print("----Adding constraints----")
for i in range(N):
    var_name = 's_' + str(i)

    y_wx1 = gp.LinExpr(-y[i]*x[i], list(w.values()))

    exp_var_helper = m.addVar(vtype=GRB.CONTINUOUS, name="exp_var_helper_"+str(i))
    m.addConstr(exp_var_helper == y_wx1, "exp_var_helper_constr_" + str(i))

    exp_var = m.addVar(vtype=GRB.CONTINUOUS, name="exp_var_"+str(i))
    m.addGenConstrExp(exp_var_helper,exp_var, "exp_var_constr_"+str(i))

    log_helper = m.addVar(vtype=GRB.CONTINUOUS, name="log_helper_" + str(i))
    m.addConstr(log_helper==exp_var+1,"log__constr_"+str(i))

    log_var = m.addVar(vtype=GRB.CONTINUOUS, name="log_var_" + str(i))
    m.addGenConstrLog(log_helper,log_var, "log_var_constr_" + str(i))

    m.addConstr(log_var <= s_i[var_name],"constr1_"+str(i))
    #

    y_wx2 = gp.LinExpr(y[i]*x[i], list(w.values()))

    exp_var_helper2 = m.addVar(vtype=GRB.CONTINUOUS, name="exp_var_helper2_" + str(i))
    m.addConstr(exp_var_helper2 == y_wx2, "exp_var_helper_constr2_" + str(i))

    exp_var2 = m.addVar(vtype=GRB.CONTINUOUS, name="exp_var2_" + str(i))
    m.addGenConstrExp(exp_var_helper2,exp_var2, "exp_var_constr2_" + str(i))

    log_helper2 = m.addVar(vtype=GRB.CONTINUOUS, name="log_helper2_" + str(i))
    m.addConstr(log_helper2 == exp_var2 + 1, "log__constr2_" + str(i))

    log_var2 = m.addVar(vtype=GRB.CONTINUOUS, name="log_var2_" + str(i))

    m.addGenConstrLog(log_helper2, log_var2, "log_var_constr2" + str(i))
    m.addConstr(log_var2- kappa*lambda_var <= s_i[var_name], "constr2_"+str(i))

    # adding the dual norm constraints
    # we use the fact that 1/p+1/q=1
    # if we take the dual of 2-norm it is the 2-norm itself.



# print("----Adding the norm constraints----")
if NORM == 'inf':
    pass


elif NORM == 'one':
    # pass
    ## here the dual will be inf-norm=max(abs values of components)
    abs_w_value = []
    max_w_component = m.addVar(vtype=GRB.CONTINUOUS)

    # here the dual will be 1-norm=summation of absolute values of components
    for j in range(col):
        abs_w_value.append(m.addVar(vtype=GRB.CONTINUOUS))
        m.addGenConstrAbs(abs_w_value[j], w["w_"+str(j)],"w_abs_value_" + str(j))

    m.addGenConstrMax(max_w_component, abs_w_value)  # ,"Max_norm_constraint_i="+str(i))
    m.addConstr(max_w_component <= lambda_var, "Lambda_norm_constraint")

elif NORM == 'two':

    # here the dual will be 2-norm
    m.addQConstr(gp.quicksum(w["w_"+str(j)] * w["w_"+str(j)]
                             for j in range(col)) <= lambda_var * lambda_var)
    # pass

# m.addConstr(lambda_var>=0)

m.optimize()
m.write('logistic_regression.lp')
print()
print("All assignments :")
w_sol=[]
for i,v in enumerate(m.getVars()):
    print('%s %g' % (v.varName, v.x))
    if v.varName=="w_0":
        w_sol.append(v.x)
    if v.varName=="w_1":
        w_sol.append(v.x)
    if v.varName=="w_2":
        w_sol.append(v.x)

def get_line_pts(w):
    x=[]
    y=[]
    print(w)
    for xi in range(4,10):
        yi=(-w[1]*xi-w[0])/w[2]
        y.append(yi)
        x.append(xi)

    return x,y

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

plot_name="Plots/logistic_regression_plot_NORM_{}_wassradi_{}_kappa_{}.png".format(NORM,e_wradi,kappa)
plot_data(x,y,w_sol,plot_name)

# print(m.display())
m.write('logistic_regression.lp')
