import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR
from sklearn import datasets
from sklearn.model_selection import KFold,StratifiedKFold
import numpy as np
from sklearn import svm
import pprint
from sklearn.metrics import roc_auc_score
import pickle

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

    # print(y)
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
    # print("===== CONIC information ======")
    # print("C1 : ",C1)
    # print("c2 : ",c2)
    # print("d : ",d)
    # print()

    return C1,c2,d

def get_line_pts(w):
    x=[]
    y=[]
    for xi in range(1,10):
        yi=(-w[1]*xi-w[0])/w[2]
        y.append(yi)
        x.append(xi)

    return x,y

def SVM_fit(x,y,params):
    # clf = svm.SVC(C=1/e_wradi,kernel='linear')
    e_wradi=params["wasserstein_radi"]
    clf = svm.LinearSVC(penalty="l2",loss="hinge",C=1/e_wradi)
    fit_obj=clf.fit(list(x[:,1:3]),y)

    return fit_obj

def plot_data_diff_Kappa(x,y,w_values_diff_k,plot_name,e_wradi):
    plot_count=len(w_values_diff_k)
    if plot_count%3!=0:
        rows=plot_count//3+1
    else:
        rows=plot_count//3
    columns=3
    fig, ax = plt.subplots(rows, columns,figsize=(15,15),sharex=True,sharey=True)
    params["wasserstein_radi"]=e_wradi

    fit_obj=SVM_fit(x,y,params)
    # create grid to evaluate model
    xx = np.linspace(4, 9, 30)
    yy = np.linspace(1, 6, 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = fit_obj.decision_function(xy).reshape(XX.shape)
    count=0
    kappa_values=list(w_values_diff_k.keys())
    colours=['r','blue','g','c','m']
    for row in range(rows):
            for col in range(columns):
            #plot the data
                if(count==plot_count):
                    continue

                w_values=w_values_diff_k[kappa_values[count]]
                kappa_value=kappa_values[count]
                count=count+1

                ax[row,col].scatter(x[:, 1], x[:, 2], c=y, cmap=plt.cm.Set1, edgecolor="k",label="data-points")
                # ax[row,col].xlabel("Sepal length")
                # ax[row,col].ylabel("Sepal width")
                ax[row,col].set_title('Kappa={}'.format(kappa_value))

                for c,key_ in zip(colours[0:len(w_values)],w_values):
                    # plot predicted line

                    x_line,y_line=get_line_pts(w_values[key_])

                    if key_=="RegularisedSVM_withoutSupport":
                        ax[row,col].plot(x_line,y_line,'--',color=c,label=key_)
                    else:
                        ax[row,col].plot(x_line,y_line,color=c,label=key_)


                # plot decision boundary and margins
                ax[row,col].contour(XX, YY, Z, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"])
                ax[row,col].set_xlim(4,7)
                ax[row,col].set_ylim(1,7)
                ax[row,col].legend()

    fig.suptitle(plot_name)
    plt.savefig(plot_name)

def plot_data_sep(x,y,w_values_diff_k,plot_name,e_wradi):
    if len(w_values_diff_k)==0:
        print("NOTHING TO PLOT")
        return
    kappa_values=list(w_values_diff_k.keys())
    plot_titles=list(w_values_diff_k[kappa_values[0]].keys())
    plot_count=len(plot_titles)

    columns=2
    if plot_count%columns!=0:
        rows=plot_count//columns+1
    else:
        rows=plot_count//columns

    fig, ax = plt.subplots(rows, columns,figsize=(15,15),sharex=True,sharey=True)
    params["wasserstein_radi"]=e_wradi

    fit_obj=SVM_fit(x,y,params)
    # create grid to evaluate model
    xx = np.linspace(4, 9, 30)
    yy = np.linspace(1, 6, 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = fit_obj.decision_function(xy).reshape(XX.shape)
    count=0

    colours=['r','blue','g','c','m','y']
    for row in range(rows):
        if(count==len(plot_titles)):
            break
        for col in range(columns):
        #plot the data

            ax[row,col].scatter(x[:, 1], x[:, 2], c=y, cmap=plt.cm.Set1, edgecolor="k",label="data-points")
            # ax[row,col].xlabel("Sepal length")
            # ax[row,col].ylabel("Sepal width")

            ax[row,col].set_title(plot_titles[count])

            # plot decision boundary and margins
            ax[row,col].contour(XX, YY, Z, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"])

            for c,key_ in zip(colours[0:len(kappa_values)],kappa_values):
                # plot predicted line
                # each line is for a different kappa value for the same algo
                x_line,y_line=get_line_pts(w_values_diff_k[key_][plot_titles[count]])
                ax[row,col].plot(x_line,y_line,color=c,label="kappa={}".format(key_))

            ax[row,col].set_xlim(4,7)
            ax[row,col].set_ylim(1,7)
            ax[row,col].legend()

            count=count+1 # each subplot is for a different algo
            if(count==len(plot_titles)):
                break


    fig.suptitle(plot_name)
    plt.savefig(plot_name)

def plot_5CV_data(split_data,data,params,plot_name):
    x=data["X"]
    y=data['y']

    plot_titles=list(split_data.keys())
    plot_count=len(plot_titles)

    columns=2
    if plot_count%columns!=0:
        rows=plot_count//columns+1
    else:
        rows=plot_count//columns

    fig, ax = plt.subplots(rows, columns,figsize=(15,15),sharex=True,sharey=True)

    count=0

    colours=['r','blue','g','c','m','y']
    for row in range(rows):
        if(count==len(plot_titles)):
            break
        for col in range(columns):
        #plot the data
            split_data_instance=split_data[plot_titles[count]]
            x_train=split_data_instance["X_train"]
            x_test=split_data_instance["X_test"]
            y_train=split_data_instance["y_train"]
            y_test=split_data_instance["y_test"]

            ax[row,col].scatter(x_train[:, 1], x_train[:, 2], c=y_train, cmap=plt.cm.Set1, edgecolor="k",label="data-points",alpha=0.1)
            ax[row,col].scatter(x_test[:, 1], x_test[:, 2],marker="x", c=y_test, cmap=plt.cm.Set1,label="data-points")
            # ax[row,col].xlabel("Sepal length")
            # ax[row,col].ylabel("Sepal width")

            ax[row,col].set_title(plot_titles[count])

            # plot decision boundary and margins

            fit_obj=SVM_fit(x_train,y_train,params)
            # create grid to evaluate model
            xx = np.linspace(4, 9, 30)
            yy = np.linspace(1, 6, 30)
            YY, XX = np.meshgrid(yy, xx)
            xy = np.vstack([XX.ravel(), YY.ravel()]).T
            Z = fit_obj.decision_function(xy).reshape(XX.shape)
            ax[row,col].contour(XX, YY, Z, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"])
            w_values=split_data_instance["w_values"]
            for c,key_ in zip(colours[0:len(w_values)],w_values):
                # plot predicted line
                # each line is for a different kappa value for the same algo
                x_line,y_line=get_line_pts(w_values[key_])
                ax[row,col].plot(x_line,y_line,color=c,label=key_)

            ax[row,col].set_xlim(4,7)
            ax[row,col].set_ylim(1,7)
            ax[row,col].legend()

            count=count+1 # each subplot is for a different algo
            if(count==len(plot_titles)):
                break


    fig.suptitle(plot_name)
    plt.savefig(plot_name)


def DRSVM_withSupport(data,params):
    x=data['X']
    y=data['y']
    N=x.shape[0]
    # conic parametes
    C1=params['conic-info']['C1']
    c2=params['conic-info']['c2']
    d=params['conic-info']['d']

    NORM=params["NORM"] # 'one','two','inf'
    e_wradi=params["wasserstein_radi"]#wasserstein radi
    kappa=params["kappa"]#
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


    # print("----Adding constraints----")
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
    # print("----Adding positive constraints----")
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

    w_sol=[]
    for i in range(col):
        varName="w"+str(i)
        v=m.getVarByName(varName)
        w_sol.append(v.x)



    m.write('SVM_Models/DRSVM_withSupport.lp')

    return w_sol

def DRSVM_withoutSupport(data,params):
    x=data['X']
    y=data['y']
    N=x.shape[0]
    NORM=params["NORM"] # 'one','two','inf'
    e_wradi=params["wasserstein_radi"]#wasserstein radi
    kappa=params["kappa"]#
    # Create a new model
    m = gp.Model("DRSVM_withoutSupport")

    if(NORM=='two'):
         m.params.NonConvex = 2

    col=x.shape[1]
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
    w_sol=[]
    for i in range(col):
        varName="w"+str(i)
        v=m.getVarByName(varName)
        w_sol.append(v.x)



    m.write('SVM_Models/DRSVM_withoutSupport.lp')

    return w_sol

def RegularisedSVM_withoutSupport(data,params):
    x=data['X']
    y=data['y']
    N=x.shape[0]
    NORM=params["NORM"] # 'one','two','inf'
    e_wradi=params["wasserstein_radi"]#wasserstein radi
    kappa=params["kappa"]#
    # Create a new model
    m = gp.Model("RegularisedSVM_withoutSupport")

    if(NORM=='two'):
         m.params.NonConvex = 2

    col=x.shape[1]
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
    # print("----Adding positive constraints----")
    for i in range(N):
        var_name='s_'+str(i)
        m.addConstr(s_i[var_name]>=0,var_name+">=0")

    #Add contraint with data points

    # Optimize model
    m.optimize()
    w_sol=[]
    for i in range(col):
        varName="w"+str(i)
        v=m.getVarByName(varName)
        w_sol.append(v.x)


    m.write('SVM_Models/RegularisedSVM_withoutSupport.lp')

    return w_sol

def RegularisedSVM_withSupport(data,params):
    x=data['X']
    y=data['y']
    N=x.shape[0]
    NORM=params["NORM"] # 'one','two','inf'
    e_wradi=params["wasserstein_radi"]#wasserstein radi
    kappa=params["kappa"]#

    # conic parametes
    C1=params['conic-info']['C1']
    c2=params['conic-info']['c2']
    d=params['conic-info']['d']

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
    w_sol=[]
    for i in range(col):
        varName="w"+str(i)
        v=m.getVarByName(varName)
        w_sol.append(v.x)


    m.write('SVM_Models/RegularisedSVM_withSupport.lp')

    return w_sol

def Classical_SVM(data,params):
    x=data['X']
    y=data['y']
    N=x.shape[0]
    NORM=params["NORM"] # 'one','two','inf'
    e_wradi=params["wasserstein_radi"]#wasserstein radi
    kappa=params["kappa"]#
    # Create a new model
    m = gp.Model("Classical_SVM")

    col=x.shape[1]

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
    w_sol=[]
    for i in range(col):
        varName="w"+str(i)
        v=m.getVarByName(varName)
        w_sol.append(v.x)



    return w_sol

def make_all_plots(data,params):
    x,y=data["X"],data['y']
    N=x.shape[0]
    NORM=params["NORM"] # 'one','two','inf'
    e_wradi=params["wasserstein_radi"]#wasserstein radi
    kappa=params["kappa"]#
    # for the same parameters plot the different SVM configurations
    w_values={}
    w_values["DRSVM_withoutSupport"]=DRSVM_withoutSupport(data,params)
    w_values["DRSVM_withSupport"]=DRSVM_withSupport(data,params)
    w_values["RegularisedSVM_withoutSupport"]=RegularisedSVM_withoutSupport(data,params)
    w_values["RegularisedSVM_withSupport"]=RegularisedSVM_withSupport(data,params)
    w_values["Classical_SVM"]=Classical_SVM(data,params)

    if len(w_values.keys())==5:
        plot_name="Plots_SVM_Implementation_V2/SVM_Implementations_NORM_{}_wassradi_{}_kappa_{}.png".format(NORM,e_wradi,kappa)
        print("\nSaved plot as ",plot_name)
        plot_data(x,y,w_values,plot_name)
    elif len(w_values.keys())==1:
        plot_name="Plots_SVM_Implementation_V2/{}_NORM_{}_wassradi_{}_kappa_{}.png".format(list(w_values.keys())[0],NORM,e_wradi,kappa)
        print("\nSaved plot as ",plot_name)
        plot_data(x,y,w_values,plot_name)
    elif "RegularisedSVM_withSupport" in list(w_values.keys()) and "RegularisedSVM_withoutSupport" in list(w_values.keys()):
        plot_name="Plots_SVM_Implementation_V2/RegularisedSVM_NORM_{}_wassradi_{}_kappa_{}.png".format(NORM,e_wradi,kappa)
        print("\nSaved plot as ",plot_name)
        plot_data(x,y,w_values,plot_name)


def vary_kappa(data,params,kappa_values=[0.1, 0.25, 0.5, 0.75,1,1000]):
    x,y=data['X'],data['y']
    w_values_kappa={}
    for k in kappa_values:
        params["kappa"]=k
        w_values={}
        w_values["DRSVM_withoutSupport"]=DRSVM_withoutSupport(data,params)
        w_values["RegularisedSVM_withoutSupport"]=RegularisedSVM_withoutSupport(data,params)
        w_values["Classical_SVM"]=Classical_SVM(data,params)
        w_values_kappa[k]=w_values

    plot_name="Plots_SVM_Implementation_V2/SepratedMultiPlot_NORM_{}_wassradi_{}_kappa_{}.png".format(NORM,e_wradi,kappa_values)
    plot_data_sep(x,y,w_values_kappa,plot_name,params["wasserstein_radi"])
    print("\nSaved plot as ",plot_name)

    plot_name="Plots_SVM_Implementation_V2/MultiPlot_NORM_{}_wassradi_{}_kappa_{}.png".format(NORM,e_wradi,kappa_values)
    plot_data_diff_Kappa(x,y,w_values_kappa,plot_name,params["wasserstein_radi"])
    print("\nSaved plot as ",plot_name)


################ k-fold CV implementation########################
def Train(data,params):
    w_values={}
    w_values["DRSVM_withoutSupport"]=DRSVM_withoutSupport(data,params)
    w_values["RegularisedSVM_withoutSupport"]=RegularisedSVM_withoutSupport(data,params)
    w_values["Classical_SVM"]=Classical_SVM(data,params)

    return w_values

def Predict(w_value,data):
    X=data['X']
    y_actual=data['y']
    y_pred=[]
    for x in X:
        dot_prod=sum([w_value[j]*x[j] for j in range(len(w_value))])
        if dot_prod>=0:
            y_pred.append(-1)
        else:
            y_pred.append(1)
    # print("y_actual : {} y_pred : {}".format(y_actual,y_pred))
    return y_pred

def Cross_Validation_5fold(data,params):
    X=data['X']
    y=data['y']
    SPLITS=5
    kf=StratifiedKFold(n_splits=SPLITS,shuffle=True)
    split_count=0

    split_scores={}
    split_data={}
    for train_index, test_index in kf.split(X,y):
        print("--------Split-{}--------".format(split_count))
        print("Train Index : ",train_index)
        print("Test Index : ",test_index)
        split_count=split_count+1

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print("X_train : ",X_train.shape)
        print("X_test : ",X_test.shape)

        w_values=Train({"X":X_train,"y":y_train},params)

        y_pred={}
        if "Average" not in split_scores:
            split_scores["Average"]={}
        split_scores["Split_{}".format(split_count)]={}
        for key_ in w_values:
            if key_ not in split_scores["Average"]:
                split_scores["Average"][key_]=0

            y_pred[key_]=Predict(w_values[key_],{"X":X_test,"y":y_test})
            score=roc_auc_score(y_test,y_pred[key_])
            print("{} : W = {} ROC_AUC_SCORE= {}%".format(key_,w_values[key_],score*100))
            split_scores["Split_{}".format(split_count)][key_]=score
            split_scores["Average"][key_]=split_scores["Average"][key_]+score/SPLITS
        split_data["Split_{}".format(split_count)]={"X_train":X_train,"X_test":X_test,"y_train":y_train,"y_test":y_test,"w_values":w_values}


    return split_scores,split_data


#########################################

## ONLY FOR DIFFERING KAPPA
# # initialize all the data
# x,y=get_iris_dataset()
# FEATURES=x.shape[1] # we append the 1 constant also to the feature space
# # print("FEATURES : ",FEATURES)
# # print("X : ",x)
# # print("y : ",y)
#
# C1,c2,d=get_conic_info(x,y,FEATURES)
# N=x.shape[0]
# NORM="two" # 'one','two','inf'
# e_wradi=1#wasserstein radi
# kappa=0.1# cost of switching a label
# EXP_SIZE=10 # decide how many points to use
#
# #arguments of data and parameters
# data={'X':x,'y':y}
# conic_info={'C1':C1,'c2':c2,'d':d}
# params={"NORM":NORM,"wasserstein_radi":e_wradi,"kappa":kappa,"conic-info":conic_info}


#########################################

# initialize all the data
# x,y=get_iris_dataset()
# FEATURES=x.shape[1] # we append the 1 constant also to the feature space
# # print("FEATURES : ",FEATURES)
# # print("X : ",x)
# # print("y : ",y)
#
# C1,c2,d=get_conic_info(x,y,FEATURES)
# N=x.shape[0]
# NORM="two" # 'one','two','inf'
# e_wradi=1#wasserstein radi
# kappa=0.1# cost of switching a label
# EXP_SIZE=10 # decide how many points to use
#
# #arguments of data and parameters
# data={'X':x,'y':y}
# conic_info={'C1':C1,'c2':c2,'d':d}
# params={"NORM":NORM,"wasserstein_radi":e_wradi,"kappa":kappa,"conic-info":conic_info}

# results,split_data=Cross_Validation_5fold(data,params)
# # df=pd.DataFrame(results)
# # print(df)
# pickle_name="Pickle_data/5CV_NORM_{}_wassradi_{}_kappa_{}.pkl".format(NORM,e_wradi,kappa)
# plot_name="Plots/5CV_NORM_{}_wassradi_{}_kappa_{}.png".format(NORM,e_wradi,kappa)
# with open(pickle_name, 'wb') as fh:
#    pickle.dump(split_data, fh)
# plot_5CV_data(split_data,data,params,plot_name)
# # print(results)
#########################################

## Running for different values of kappa and wasserstein
def run_cv(Norms,wasserstein_radi,kappa):


    # initialize all the data
    x,y=get_iris_dataset()
    FEATURES=x.shape[1] # we append the 1 constant also to the feature space
    # print("FEATURES : ",FEATURES)
    # print("X : ",x)
    # print("y : ",y)

    C1,c2,d=get_conic_info(x,y,FEATURES)
    N=x.shape[0]
    EXP_SIZE=10 # decide how many points to use

    #arguments of data and parameters
    data={'X':x,'y':y}
    conic_info={'C1':C1,'c2':c2,'d':d}
    df_main=pd.DataFrame()
    for norm in Norms:
        for w_radi in wasserstein_radi:
            for k in kappa:
                params={"NORM":norm,"wasserstein_radi":w_radi,"kappa":k,"conic-info":conic_info}


                results,split_data=Cross_Validation_5fold(data,params)

                df=pd.DataFrame(results)
                df["wasserstein_radi"]=w_radi
                df["NORM"]=norm
                df["kappa"]=k

                df_main=pd.concat([df_main,df])

                
            # # print(df)
            # pickle_name="Pickle_data/5CV_NORM_{}_wassradi_{}_kappa_{}.pkl".format(NORM,e_wradi,kappa)
            # plot_name="Plots/5CV_NORM_{}_wassradi_{}_kappa_{}.png".format(NORM,e_wradi,kappa)
            # with open(pickle_name, 'wb') as fh:
            #    pickle.dump(split_data, fh)
            # plot_5CV_data(split_data,data,params,plot_name)
            # print(results)
    print(df_main)
    df_main.to_csv("Pickle_data/5CV_NORM_{}_wassradi_{}_kappa_{}.csv".format(Norms,wasserstein_radi,kappa))

#########################################


run_cv(["two"],[0.01,0.1,0.5,1],[0.1,0.25,0.5,0.75,1])
