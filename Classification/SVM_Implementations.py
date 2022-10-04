import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR
from sklearn import datasets
import numpy as np
from sklearn import svm
import pprint
# norm selection for the computation
NORM="two" # 'one','two','inf'
e_wradi=0.1#wasserstein radi
kappa=0.1# cost of switching a label
EXP_SIZE=10 # decide how many points to use

# dataset
def get_iris_dataset(to_plot=True):

    # using the iris dataset
    iris=datasets.load_iris()
    X = iris.data[:, :2] # we only take the first two features.
    X=np.c_[np.ones(X.shape[0]), X]

    y = iris.target

    mask=y!=1  # take the labels 0,2

    X=X[mask]
    X=X
    y=y[mask]
    y=y

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
    for xi in range(1,10):
        yi=(-w[1]*xi-w[0])/w[2]
        y.append(yi)
        x.append(xi)

    return x,y

def SVM_fit(x,y):
    # clf = svm.SVC(C=1/e_wradi,kernel='linear')
    clf = svm.LinearSVC(penalty="l2",loss="hinge",C=1/e_wradi)
    fit_obj=clf.fit(list(x[:,1:3]),y)

    return fit_obj

def plot_data(x,y,w_values,plot_name):
    #plot the data
    plt.scatter(x[:, 1], x[:, 2], c=y, cmap=plt.cm.Set1, edgecolor="k",label="data-points")
    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")
    colours=['r','blue','g','c','m']
    for c,key_ in zip(colours[0:len(w_values)],w_values):
        # plot predicted line
        x_line,y_line=get_line_pts(w_values[key_])
        marker='-'
        if c=='blue':
            marker='-.'
        plt.plot(x_line,y_line,color=c,label=key_,marker=marker)

    # #plot the LinearSVR fitted line
    # fit_obj=LinearSVR_fit(x,y)
    # y_predict=[]
    # x_predict=[0,1,2,3,4]
    # for xi in x_predict:
    #     y_predict.append(fit_obj.predict([[1,xi]])[0])
    #
    # plt.plot(x_predict,y_predict,linestyle='dashed',color='black',label="LinearSVR")
    fit_obj=SVM_fit(x,y)
    # create grid to evaluate model
    xx = np.linspace(4, 9, 30)
    yy = np.linspace(1, 6, 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = fit_obj.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    plt.contour(XX, YY, Z, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"])
    plt.xlim(4,7)
    plt.ylim(1,7)
    plt.legend()
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
def DRSVM_withSupport():
    # Create a new model
    m = gp.Model("DRSVM_withSupport")

    if(NORM=='two'):
         m.params.NonConvex = 2

    col=FEATURES
    s_i={}
    p_i_plus={}
    p_i_minus={}
    w={}
    #hyperplane w=[w1,w2,..wn] each x=[1,x1...wn] data point
    for i in range(col):
        var_name="w"+str(i)
        w[i]=m.addVar(vtype=GRB.CONTINUOUS,name=var_name,lb=-GRB.INFINITY)

    #lambda variable
    lambda_var=m.addVar(vtype=GRB.CONTINUOUS,name='Lambda')
    # Create variables

    # create pi+,pi-,si
    for i in range(N):
        var_name='s_'+str(i)
        pi_plus_name='p_'+str(i)+'_plus'
        pi_minus_name='p_'+str(i)+'_minus'

        s_i[var_name]=m.addVar(vtype=GRB.CONTINUOUS,name=var_name)

        for j in range(col):
            p_i_plus[i,j]=m.addVar(vtype=GRB.CONTINUOUS,name=pi_plus_name+f"[{j}]")
            p_i_minus[i,j]=m.addVar(vtype=GRB.CONTINUOUS,name=pi_minus_name+f"[{j}]")

    # print(p_i_plus.keys())
    # Set objective
    Obj_value=lambda_var*e_wradi

    for si in s_i.keys():
        Obj_value=Obj_value+(1.0/N)*s_i[si]

    m.setObjective(Obj_value, GRB.MINIMIZE)

    data_constr_1_value=[]
    data_constr_2_value=[]


    print("----Adding constraints----")
    for i in range(N):
        var_name='s_'+str(i)


        pi_plus_name='p_'+str(i)+'_plus'
        pi_minus_name='p_'+str(i)+'_minus'

        data_constr_1_value.append([])
        data_constr_2_value.append([])

        constraint_name_1='data[{}]_constr1'.format(i)
        constraint_name_2='data[{}]_constr2'.format(i)

        # value of d-C1xi
        conic_value=[]
        for j in range(col):
            conic_value.append(d[j]-sum([C1[j][k]*x[i][k] for k in range(col)]))
            # print(conic_value[j])

        # adding the constraints for si in the objective function

        y_wx=gp.LinExpr(y[i]*x[i],list(w.values()))

        exp1=gp.LinExpr(conic_value,[p_i_plus[i,k] for k in range(col)])
        exp1.add(y_wx,-1)
        exp1.addConstant(1)
        m.addConstr(exp1<=s_i[var_name],constraint_name_1)


        exp2=gp.LinExpr(conic_value,[p_i_minus[i,k] for k in range(col)])
        exp2.add(y_wx,1)
        exp2.add(lambda_var,-kappa)
        exp2.addConstant(1)
        m.addConstr(exp2<=s_i[var_name],constraint_name_2)


        # adding the dual norm constraints
        # we use the fact that 1/p+1/q=1
        # if we take the dual of 2-norm it is the 2-norm itself.

        norm_vector_p_plus=[]
        norm_vector_p_minus=[]
        for j in range(col):

            # || C.T *p_i_plus+y_i*w || <= lambda
            norm_vector_p_plus.append(gp.LinExpr(C1[j],[p_i_plus[i,k] for k in range(col)])+gp.LinExpr(y[i],w[j]))
            norm_vector_p_minus.append(gp.LinExpr(C1[j],[p_i_minus[i,k] for k in range(col)])-gp.LinExpr(y[i],w[j]))
            # these norm vectors should have the same length as w

        # print("----Adding the norm constraints----")
        if NORM=='inf':
            pass

        elif NORM=='one':
            # pass
            ## here the dual will be inf-norm=max(abs values of components)
            abs_norm_value1=[]
            abs_norm_value1_helper={}
            max_norm_component_p_i_plus=m.addVar(vtype=GRB.CONTINUOUS)

            abs_norm_value2=[]
            abs_norm_value2_helper={}
            max_norm_component_p_i_minus=m.addVar(vtype=GRB.CONTINUOUS)
            # here the dual will be 1-norm=summation of absolute values of components
            for j in range(col):
                abs_norm_value1.append(m.addVar(vtype=GRB.CONTINUOUS))
                abs_norm_value1_helper[j]=m.addVar(vtype=GRB.CONTINUOUS)

                abs_norm_value2.append(m.addVar(vtype=GRB.CONTINUOUS))
                abs_norm_value2_helper[j]=m.addVar(vtype=GRB.CONTINUOUS)

                m.addConstr(abs_norm_value1_helper[j]==norm_vector_p_plus[j],"1Norm_abs_norm_value_"+str(j)+"p_"+str(i)+"_plus")
                m.addGenConstrAbs(abs_norm_value1[j],abs_norm_value1_helper[j],"1Norm_abs_value_"+str(j)+"p_"+str(i)+"_plus")

                m.addConstr(abs_norm_value2_helper[j]==norm_vector_p_minus[j],"1Norm_abs_norm_value_"+str(j)+"p_"+str(i)+"_minus")
                m.addGenConstrAbs(abs_norm_value2[j],abs_norm_value2_helper[j],"1Norm_abs_value_"+str(j)+"p_"+str(i)+"_minus")

            m.addGenConstrMax(max_norm_component_p_i_plus,abs_norm_value1)#,"Max_norm_constraint_i="+str(i))
            m.addConstr(max_norm_component_p_i_plus<=lambda_var,"Lambda_norm_constraint_"+"p_"+str(i)+"_plus")

            m.addGenConstrMax(max_norm_component_p_i_minus,abs_norm_value2)#,"Max_norm_constraint_i="+str(i))
            m.addConstr(max_norm_component_p_i_minus<=lambda_var,"Lambda_norm_constraint_"+"p_"+str(i)+"_minus")

        elif NORM=='two':

            # here the dual will be 2-norm
            m.addQConstr(gp.quicksum(norm_vector_p_plus[j] * norm_vector_p_plus[j]
                         for j in range(col)) <= lambda_var * lambda_var)
            m.addQConstr(gp.quicksum(norm_vector_p_minus[j] * norm_vector_p_minus[j]
                          for j in range(col)) <= lambda_var * lambda_var)



    #positive contraints
    print("----Adding positive constraints----")
    for i in range(N):
        var_name='s_'+str(i)

        m.addConstr(s_i[var_name]>=0,var_name+">=0")

        for j in range(col):
            pi_plus_constr_name=f'p_[{i},{j}]_plus>=0' # example : p_2_plus[0]>=0, p_2_plus[1]>=0
            pi_minus_constr_name=f'p_[{i},{j}]_minus>=0'

            m.addConstr(p_i_plus[i,j]>=0,pi_plus_constr_name) # these belong to positive orthant
            m.addConstr(p_i_minus[i,j]>=0,pi_minus_constr_name)


    #Add contraint with data points

    # Optimize model
    m.optimize()

    print("\nAll assignments :")
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

    m.write('SVM_Models/DRSVM_withSupport.lp')

    return w_sol

def DRSVM_withoutSupport():
    # Create a new model
    m = gp.Model("DRSVM_withoutSupport")

    if(NORM=='two'):
         m.params.NonConvex = 2

    col=FEATURES
    s_i={}
    p_i_plus={}
    p_i_minus={}
    w={}
    #hyperplane w=[w1,w2,..wn] each x=[1,x1...wn] data point
    for i in range(col):
        var_name="w"+str(i)
        w[i]=m.addVar(vtype=GRB.CONTINUOUS,name=var_name,lb=-GRB.INFINITY)

    #lambda variable
    lambda_var=m.addVar(vtype=GRB.CONTINUOUS,name='Lambda')
    # Create variables

    # create pi+,pi-,si
    for i in range(N):
        var_name='s_'+str(i)
        s_i[var_name]=m.addVar(vtype=GRB.CONTINUOUS,name=var_name)

    # Set objective
    Obj_value=lambda_var*e_wradi

    for si in s_i.keys():
        Obj_value=Obj_value+(1.0/N)*s_i[si]

    m.setObjective(Obj_value, GRB.MINIMIZE)

    data_constr_1_value=[]
    data_constr_2_value=[]


    print("----Adding constraints----")
    for i in range(N):
        var_name='s_'+str(i)

        constraint_name_1='data[{}]_constr1'.format(i)
        constraint_name_2='data[{}]_constr2'.format(i)

        # adding the constraints for si in the objective function
        y_wx=gp.LinExpr(y[i]*x[i],list(w.values()))
        exp1=gp.LinExpr()
        exp1.add(y_wx,-1)
        exp1.addConstant(1)
        m.addConstr(exp1<=s_i[var_name],constraint_name_1)


        exp2=gp.LinExpr()
        exp2.add(y_wx,1)
        exp2.add(lambda_var,-kappa)
        exp2.addConstant(1)
        m.addConstr(exp2<=s_i[var_name],constraint_name_2)

        # adding the dual norm constraints
        # we use the fact that 1/p+1/q=1
        # if we take the dual of 2-norm it is the 2-norm itself.

    # print("----Adding the norm constraints----")
    if NORM=='inf':
        pass

    elif NORM=='one':
        # pass
        ## here the dual will be inf-norm=max(abs values of components)
        abs_norm_value1=[]
        abs_norm_value1_helper={}
        max_norm_component_w=m.addVar(vtype=GRB.CONTINUOUS)

        # here the dual will be 1-norm=summation of absolute values of components
        for j in range(col):
            abs_norm_value1.append(m.addVar(vtype=GRB.CONTINUOUS))
            abs_norm_value1_helper[j]=m.addVar(vtype=GRB.CONTINUOUS)

            m.addConstr(abs_norm_value1_helper[j]==w[j],"1Norm_abs_norm_value_"+str(j)+"w["+str(i)+"]")
            m.addGenConstrAbs(abs_norm_value1[j],abs_norm_value1_helper[j],"1Norm_abs_value_"+str(j)+"w["+str(i)+"]")

        m.addGenConstrMax(max_norm_component_w,abs_norm_value1)#,"Max_norm_constraint_i="+str(i))
        m.addConstr(max_norm_component_w<=lambda_var,"Lambda_norm_constraint_w")

    elif NORM=='two':

        # here the dual will be 2-norm
        m.addQConstr(gp.quicksum(w[j] * w[j]
                     for j in range(col)) <= lambda_var * lambda_var)

    #positive contraints
    print("----Adding positive constraints----")
    for i in range(N):
        var_name='s_'+str(i)
        m.addConstr(s_i[var_name]>=0,var_name+">=0")

    #Add contraint with data points

    # Optimize model
    m.optimize()

    print("\nAll assignments :")
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

    m.write('SVM_Models/DRSVM_withoutSupport.lp')

    return w_sol

def RegularisedSVM_withoutSupport():
    # Create a new model
    m = gp.Model("RegularisedSVM_withoutSupport")

    if(NORM=='two'):
         m.params.NonConvex = 2

    col=FEATURES
    s_i={}
    p_i_plus={}
    p_i_minus={}
    w={}
    #hyperplane w=[w1,w2,..wn] each x=[1,x1...wn] data point
    for i in range(col):
        var_name="w"+str(i)
        w[i]=m.addVar(vtype=GRB.CONTINUOUS,name=var_name,lb=-GRB.INFINITY)

    #lambda variable
    lambda_var=m.addVar(vtype=GRB.CONTINUOUS,name='Lambda')
    # Create variables

    # create pi+,pi-,si
    for i in range(N):
        var_name='s_'+str(i)
        s_i[var_name]=m.addVar(vtype=GRB.CONTINUOUS,name=var_name)

    # Set objective
    Obj_value=lambda_var*e_wradi

    for si in s_i.keys():
        Obj_value=Obj_value+(1.0/N)*s_i[si]

    m.setObjective(Obj_value, GRB.MINIMIZE)

    data_constr_1_value=[]
    data_constr_2_value=[]


    print("----Adding constraints----")
    for i in range(N):
        var_name='s_'+str(i)

        constraint_name_1='data[{}]_constr1'.format(i)
        # constraint_name_2='data[{}]_constr2'.format(i)

        # adding the constraints for si in the objective function
        y_wx=gp.LinExpr(y[i]*x[i],list(w.values()))
        exp1=gp.LinExpr()
        exp1.add(y_wx,-1)
        exp1.addConstant(1)
        m.addConstr(exp1<=s_i[var_name],constraint_name_1)

        # adding the dual norm constraints
        # we use the fact that 1/p+1/q=1
        # if we take the dual of 2-norm it is the 2-norm itself.

    # print("----Adding the norm constraints----")
    if NORM=='inf':
        pass

    elif NORM=='one':
        # pass
        ## here the dual will be inf-norm=max(abs values of components)
        abs_norm_value1=[]
        abs_norm_value1_helper={}
        max_norm_component_w=m.addVar(vtype=GRB.CONTINUOUS)

        # here the dual will be 1-norm=summation of absolute values of components
        for j in range(col):
            abs_norm_value1.append(m.addVar(vtype=GRB.CONTINUOUS))
            abs_norm_value1_helper[j]=m.addVar(vtype=GRB.CONTINUOUS)

            m.addConstr(abs_norm_value1_helper[j]==w[j],"1Norm_abs_norm_value_"+str(j)+"w["+str(i)+"]")
            m.addGenConstrAbs(abs_norm_value1[j],abs_norm_value1_helper[j],"1Norm_abs_value_"+str(j)+"w["+str(i)+"]")

        m.addGenConstrMax(max_norm_component_w,abs_norm_value1)#,"Max_norm_constraint_i="+str(i))
        m.addConstr(max_norm_component_w<=lambda_var,"Lambda_norm_constraint_w")

    elif NORM=='two':

        # here the dual will be 2-norm
        m.addQConstr(gp.quicksum(w[j] * w[j]
                     for j in range(col)) <= lambda_var * lambda_var)

    #positive contraints
    print("----Adding positive constraints----")
    for i in range(N):
        var_name='s_'+str(i)
        m.addConstr(s_i[var_name]>=0,var_name+">=0")

    #Add contraint with data points

    # Optimize model
    m.optimize()

    print("\nAll assignments :")
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

    m.write('SVM_Models/RegularisedSVM_withoutSupport.lp')

    return w_sol

def RegularisedSVM_withSupport():
    # Create a new model
    m = gp.Model("RegularisedSVM_withSupport")

    if(NORM=='two'):
         m.params.NonConvex = 2

    col=FEATURES
    s_i={}
    p_i_plus={}
    p_i_minus={}
    w={}
    #hyperplane w=[w1,w2,..wn] each x=[1,x1...wn] data point
    for i in range(col):
        var_name="w"+str(i)
        w[i]=m.addVar(vtype=GRB.CONTINUOUS,name=var_name,lb=-GRB.INFINITY)

    #lambda variable
    lambda_var=m.addVar(vtype=GRB.CONTINUOUS,name='Lambda')
    # Create variables

    # create pi+,pi-,si
    for i in range(N):
        var_name='s_'+str(i)
        pi_plus_name='p_'+str(i)+'_plus'
        s_i[var_name]=m.addVar(vtype=GRB.CONTINUOUS,name=var_name)

        for j in range(col):
            p_i_plus[i,j]=m.addVar(vtype=GRB.CONTINUOUS,name=pi_plus_name+f"[{j}]")

    # print(p_i_plus.keys())
    # Set objective
    Obj_value=lambda_var*e_wradi

    for si in s_i.keys():
        Obj_value=Obj_value+(1.0/N)*s_i[si]

    m.setObjective(Obj_value, GRB.MINIMIZE)

    data_constr_1_value=[]
    data_constr_2_value=[]


    print("----Adding constraints----")
    for i in range(N):
        var_name='s_'+str(i)

        data_constr_1_value.append([])

        constraint_name_1='data[{}]_constr1'.format(i)

        # adding the constraints for si in the objective function

        y_wx=gp.LinExpr(y[i]*x[i],list(w.values()))
        exp1=gp.LinExpr()
        exp1.add(y_wx,-1)
        exp1.addConstant(1)
        m.addConstr(exp1<=s_i[var_name],constraint_name_1)

        # adding the dual norm constraints
        # we use the fact that 1/p+1/q=1
        # if we take the dual of 2-norm it is the 2-norm itself.

        norm_vector_p_plus=[]
        for j in range(col):

            # || C.T *p_i_plus+y_i*w || <= lambda
            norm_vector_p_plus.append(gp.LinExpr(C1[j],[p_i_plus[i,k] for k in range(col)])+gp.LinExpr(y[i],w[j]))
            # these norm vectors should have the same length as w

        # print("----Adding the norm constraints----")
        if NORM=='inf':
            pass

        elif NORM=='one':
            # pass
            ## here the dual will be inf-norm=max(abs values of components)
            abs_norm_value1=[]
            abs_norm_value1_helper={}
            max_norm_component_p_i_plus=m.addVar(vtype=GRB.CONTINUOUS)
            # here the dual will be 1-norm=summation of absolute values of components
            for j in range(col):
                abs_norm_value1.append(m.addVar(vtype=GRB.CONTINUOUS))
                abs_norm_value1_helper[j]=m.addVar(vtype=GRB.CONTINUOUS)

                m.addConstr(abs_norm_value1_helper[j]==norm_vector_p_plus[j],"1Norm_abs_norm_value_"+str(j)+"p_"+str(i)+"_plus")
                m.addGenConstrAbs(abs_norm_value1[j],abs_norm_value1_helper[j],"1Norm_abs_value_"+str(j)+"p_"+str(i)+"_plus")


            m.addGenConstrMax(max_norm_component_p_i_plus,abs_norm_value1)#,"Max_norm_constraint_i="+str(i))
            m.addConstr(max_norm_component_p_i_plus<=lambda_var,"Lambda_norm_constraint_"+"p_"+str(i)+"_plus")

        elif NORM=='two':

            # here the dual will be 2-norm
            m.addQConstr(gp.quicksum(norm_vector_p_plus[j] * norm_vector_p_plus[j]
                         for j in range(col)) <= lambda_var * lambda_var)

    #positive contraints
    print("----Adding positive constraints----")
    for i in range(N):
        var_name='s_'+str(i)

        m.addConstr(s_i[var_name]>=0,var_name+">=0")

        for j in range(col):
            pi_plus_constr_name=f'p_[{i},{j}]_plus>=0' # example : p_2_plus[0]>=0, p_2_plus[1]>=0

            m.addConstr(p_i_plus[i,j]>=0,pi_plus_constr_name) # these belong to positive orthant

    #Add contraint with data points

    # Optimize model
    m.optimize()

    print("\nAll assignments :")
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

    m.write('SVM_Models/RegularisedSVM_withSupport.lp')

    return w_sol
def Classical_SVM():
    # Create a new model
    m = gp.Model("Classical_SVM")

    col=FEATURES

    w={}
    #hyperplane w=[w1,w2,..wn] each x=[1,x1...wn] data point
    for i in range(col):
        var_name="w"+str(i)
        w[i]=m.addVar(vtype=GRB.CONTINUOUS,name=var_name,lb=-GRB.INFINITY)

    print("Setting objective")
    # Create variables
    Obj_value=0.5*(w[2]*w[2]+w[1]*w[1]+w[0]*w[0])

    m.setObjective(Obj_value, GRB.MINIMIZE)
    print("----Adding constraints----")
    for i in range(N):
        y_wx=gp.LinExpr(x[i],w.values())
        m.addConstr(1-(y[i]*y_wx)<=0)


    #Add contraint with data points

    # Optimize model
    m.optimize()
    # print(m.display())
    m.write('SVM_Models/Classical_SVM.lp')
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

    return w_sol

# for the same parameters plot the different SVM configurations
w_values={}
w_values["DRSVM_withoutSupport"]=DRSVM_withoutSupport()
w_values["DRSVM_withSupport"]=DRSVM_withSupport()
w_values["RegularisedSVM_withoutSupport"]=RegularisedSVM_withoutSupport()
w_values["RegularisedSVM_withSupport"]=RegularisedSVM_withSupport()
w_values["Classical_SVM"]=Classical_SVM()

if len(w_values.keys())==5:
    plot_name="Plots/SVM_Implementations_NORM_{}_wassradi_{}_kappa_{}.png".format(NORM,e_wradi,kappa)
    print("\nSaved plot as ",plot_name)
    plot_data(x,y,w_values,plot_name)
elif len(w_values.keys())==1:
    plot_name="Plots/{}_NORM_{}_wassradi_{}_kappa_{}.png".format(list(w_values.keys())[0],NORM,e_wradi,kappa)
    print("\nSaved plot as ",plot_name)
    plot_data(x,y,w_values,plot_name)
elif "RegularisedSVM_withSupport" in list(w_values.keys()) and "RegularisedSVM_withoutSupport" in list(w_values.keys()):
    plot_name="Plots/RegularisedSVM_NORM_{}_wassradi_{}_kappa_{}.png".format(NORM,e_wradi,kappa)
    print("\nSaved plot as ",plot_name)
    plot_data(x,y,w_values,plot_name)

print()
pprint.pprint(w_values)
