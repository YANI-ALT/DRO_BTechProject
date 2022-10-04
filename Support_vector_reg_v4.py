import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR
# norm selection for the computation
NORM="two" # 'one','two','inf'

#datapoints
N=10
#wasserstein radi
e_wradi=1000

# epsilon-insensitive loss function
epsilon=0.5# epsilon value

# decide how many points to use
EXP_SIZE=30
# dataset
df=pd.read_csv('Dummy_data.csv')
X_data_points=list(df['x'])[0:EXP_SIZE]
x=[[1,xi] for xi in X_data_points]
y=list(df['y'])[0:EXP_SIZE]

print(X_data_points)
print(y)
# both x and y should be of equal number one for each data point
assert(len(x)==len(y))
N=len(x)

#row,col=x.shape
row=N
col=2 # dimension of x,w,d,c1


# the conic defined for the data C1x+c2y<=d
# C1=[[1,0],[0,1]]
# c2=[1,1]
# d=[2,2]
C1=[]
c2=[]
d=[]
d_value=max(list(df['x']))+max(y)
for i in range(col):
    row_vector=[0]*col
    row_vector[i]=1
    C1.append(row_vector)
    c2.append(1)
    d.append(d_value)

print("C1 : ",C1)
print("c2 : ",c2)
print("d : ",d)

def get_line_pts(w):
    x=[]
    y=[]
    for xi in range(0,5):
        yi=w[1]*xi+w[0]
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
    plt.scatter(x,y,label='data-points')

    # plot predicted line
    x_line,y_line=get_line_pts(w)
    plt.plot(x_line,y_line,color='r',label='DRO-Support_vector_reg')

    # plot predicted line
    x_line,y_line=get_line_pts([1,0.5])
    plt.plot(x_line,y_line,color='g',label='Actual')

    #plot the LinearSVR fitted line
    fit_obj=LinearSVR_fit(x,y)
    y_predict=[]
    x_predict=[0,1,2,3,4]
    for xi in x_predict:
        y_predict.append(fit_obj.predict([[1,xi]])[0])

    plt.plot(x_predict,y_predict,linestyle='dashed',color='black',label="LinearSVR")

    plt.legend(loc='lower right')
    plt.title(plot_name)
    plt.savefig(plot_name)


try:

    # Create a new model
    m = gp.Model("Support_vector_reg")
    if(NORM=='two'):
         m.params.NonConvex = 2

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
        Obj_value=Obj_value+(1/N)*s_i[si]

    m.setObjective(Obj_value, GRB.MINIMIZE)
    data_constr_1_value=[]
    data_constr_2_value=[]
    for i in range(N):
        var_name='s_'+str(i)


        pi_plus_name='p_'+str(i)+'_plus'
        pi_minus_name='p_'+str(i)+'_minus'

        data_constr_1_value.append([])
        data_constr_2_value.append([])

        constraint_name_1='data[{}]_constr1'.format(i)
        constraint_name_2='data[{}]_constr2'.format(i)

        # value of d-C1xi-c2yi
        conic_value=[]
        for j in range(col):
            conic_value.append(d[j]-c2[j]*y[i]-sum([C1[j][k]*x[i][k] for k in range(col)]))
            # print(conic_value[j])

        # adding the constraints for si in the objective function
        m.addConstr(y[i]-gp.LinExpr(x[i],w.values())-epsilon
                    + gp.LinExpr(conic_value,[p_i_plus[i,k] for k in range(col)])<=s_i[var_name],constraint_name_1)

        m.addConstr(-y[i]+gp.LinExpr(x[i],w.values())-epsilon
                    + gp.LinExpr(conic_value,[p_i_minus[i,k] for k in range(col)])<=s_i[var_name],constraint_name_2)


        # adding the dual norm constraints
        # we use the fact that 1/p+1/q=1
        # if we take the dual of 2-norm it is the 2-norm itself.

        norm_vector_p_plus=[]
        norm_vector_p_minus=[]
        for j in range(col):
            norm_vector_p_plus.append(gp.LinExpr(C1[j],[p_i_plus[i,k] for k in range(col)])+w[j])
            norm_vector_p_minus.append(gp.LinExpr(C1[j],[p_i_minus[i,k] for k in range(col)])-w[j])

        norm_vector_p_plus.append(gp.LinExpr(c2,[p_i_plus[i,k] for k in range(col)])-1)
        norm_vector_p_minus.append(gp.LinExpr(c2,[p_i_minus[i,k] for k in range(col)])+1)


        if NORM=='inf':

            # pass
            abs_norm_value1={}
            abs_norm_value2={}

            abs_norm_value1_helper={}
            abs_norm_value2_helper={}

            # here the dual will be 1-norm=summation of absolute values of components
            for j in range(col+1):

                # helper variables to store the values for which absolute constraint has to be added
                abs_norm_value1[j]=m.addVar(vtype=GRB.CONTINUOUS)
                abs_norm_value2[j]=m.addVar(vtype=GRB.CONTINUOUS)

                abs_norm_value1_helper[j]=m.addVar(vtype=GRB.CONTINUOUS)
                abs_norm_value2_helper[j]=m.addVar(vtype=GRB.CONTINUOUS)

                m.addConstr(abs_norm_value1_helper[j]==norm_vector_p_plus[j],"infNorm_abs_norm_value_p_plus"+str(j)+"_i="+str(i))
                m.addConstr(abs_norm_value2_helper[j]==norm_vector_p_minus[j],"infNorm_abs_norm_value_p_minus"+str(j)+"_i="+str(i))

                # abs_norm_value2[j]=abs(abs_norm_value1[j])
                m.addGenConstrAbs(abs_norm_value1[j],abs_norm_value1_helper[j],"infNorm_abs_value_p_plus"+str(j)+"_i="+str(i))
                m.addGenConstrAbs(abs_norm_value2[j],abs_norm_value2_helper[j],"infNorm_abs_value_p_minus"+str(j)+"_i="+str(i))

            m.addConstr(gp.quicksum(abs_norm_value1[j] for j in range(col+1))<=lambda_var,"Lambda_p_plus_norm_constraint"+"_i="+str(i))
            m.addConstr(gp.quicksum(abs_norm_value2[j] for j in range(col+1))<=lambda_var,"Lambda_p_minus_norm_constraint"+"_i="+str(i))

        elif NORM=='one':
            # pass
            ## here the dual will be inf-norm=max(abs values of components)
            abs_norm_value1={}
            abs_norm_value1_helper=[]
            abs_norm_value2={}
            abs_norm_value2_helper=[]

            max_norm_component_p_plus=m.addVar(vtype=GRB.CONTINUOUS)
            max_norm_component_p_minus=m.addVar(vtype=GRB.CONTINUOUS)
            # here the dual will be 1-norm=summation of absolute values of components
            for j in range(col+1):
                abs_norm_value1[j]=m.addVar(vtype=GRB.CONTINUOUS)
                abs_norm_value1_helper.append(m.addVar(vtype=GRB.CONTINUOUS))
                m.addConstr(abs_norm_value1[j]==norm_vector_p_plus[j],"1Norm_abs_norm_value_p_plus"+str(j)+"_i="+str(i))
                # abs_norm_value1_helper[j]= abs(abs_norm_value1[j])
                m.addGenConstrAbs(abs_norm_value1_helper[j],abs_norm_value1[j],"1Norm_abs_value_p_plus"+str(j)+"_i="+str(i))

                abs_norm_value2[j]=m.addVar(vtype=GRB.CONTINUOUS)
                abs_norm_value2_helper.append(m.addVar(vtype=GRB.CONTINUOUS))
                m.addConstr(abs_norm_value2[j]==norm_vector_p_minus[j],"1Norm_abs_norm_value_p_minus"+str(j)+"_i="+str(i))
                # abs_norm_value2_helper[j]= abs(abs_norm_value2[j])
                m.addGenConstrAbs(abs_norm_value2_helper[j],abs_norm_value2[j],"1Norm_abs_value_p_minus"+str(j)+"_i="+str(i))

            # max_norm_component_p_plus=max(abs_norm_value1_helper)
            m.addGenConstrMax(max_norm_component_p_plus,abs_norm_value1_helper)
            m.addConstr(max_norm_component_p_plus<=lambda_var,"Lambda_norm_constraint_p_plus_i="+str(i))

            # max_norm_component_p_minus=max(abs_norm_value2_helper)
            m.addGenConstrMax(max_norm_component_p_minus,abs_norm_value2_helper)
            m.addConstr(max_norm_component_p_minus<=lambda_var,"Lambda_norm_constraint_p_minus_i="+str(i))

        elif NORM=='two':
            # here the dual will be 2-norm
            # norm_value1=(p_i_plus[pi_plus_name][0]+w_0)**2+(p_i_plus[pi_plus_name][1]+w_1)**2+(p_i_plus[pi_plus_name][0]+p_i_plus[pi_plus_name][1]-1)**2
            qexp1=gp.QuadExpr()
            qexp2=gp.QuadExpr()
            for k in range(col+1):
                qexp1.add(norm_vector_p_plus[k]**2)
                qexp2.add(norm_vector_p_minus[k]**2)

            # helper variables to store the square root while calculating the norm
            norm_constr1_lhs=m.addVar(vtype=GRB.CONTINUOUS)
            norm_constr2_lhs=m.addVar(vtype=GRB.CONTINUOUS)

            # helper variables to store the norm^2 value
            norm_constr1_lhs_sqr_value=m.addVar(vtype=GRB.CONTINUOUS)
            norm_constr2_lhs_sqr_value=m.addVar(vtype=GRB.CONTINUOUS)

            ## using gp.quicksum
            # m.addConstr(norm_constr1_lhs_sqr_value==gp.quicksum(norm_vector_p_plus[k]**2 for k in range(col)))
            # m.addConstr(norm_constr2_lhs_sqr_value==gp.quicksum(norm_vector_p_minus[k]**2 for k in range(col)))

            m.addConstr(norm_constr1_lhs_sqr_value==qexp1)
            m.addConstr(norm_constr2_lhs_sqr_value==qexp2)

            # take the squre root
            # norm_constr1_lhs= sqrt(norm_constr1_lhs_sqr_value)
            m.addGenConstrPow(norm_constr1_lhs_sqr_value,norm_constr1_lhs,0.5)
            m.addGenConstrPow(norm_constr2_lhs_sqr_value,norm_constr2_lhs,0.5)

            # m.addConstr(norm_constr1_lhs==norm(norm_vector_p_plus,2))
            # m.addConstr(norm_constr2_lhs==norm(norm_vector_p_minus,2))

            m.addConstr(norm_constr1_lhs<=lambda_var,'2Norm-constr_1_data[{}]'.format(i))
            m.addConstr(norm_constr2_lhs<=lambda_var,'2Norm-constr_2_data[{}]'.format(i))
    # edge_constr_s=m.addConstr(variable_dict_PD2['X_s3']+variable_dict_PD1['X_s2']==1,'edge_constr-s')

    #positive contraints

    for i in range(N):
        var_name='s_'+str(i)

        m.addConstr(s_i[var_name]>=0,var_name+">=0")

        for j in range(col):
            pi_plus_constr_name=f'p_[{i}{j}]_plus>=0' # example : p_2_plus[0]>=0, p_2_plus[1]>=0
            pi_minus_constr_name=f'p_[{i}{j}]_minus>=0'

            m.addConstr(p_i_plus[i,j]>=0,pi_plus_constr_name) # these belong to positive orthant
            m.addConstr(p_i_minus[i,j]>=0,pi_minus_constr_name)


    #Add contraint with data points

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
        if(i==3):
            print("d: ",d)
            break


    plot_name="Plots/Support_vector_reg_v4_plot_NORM_{}_wassradi_{}_epsil_{}.png".format(NORM,e_wradi,epsilon)
    plot_data(X_data_points,y,w_sol,plot_name)
    print("\nSAVING PLOT ",plot_name)
    # print(m.display())
    m.write('Support_vector_reg.lp')


    # for v in m.getVars():
    #     print('%s %g' % (v.varName, v.x))


except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ': ' + str(e))

except AttributeError:
    print('Encountered an attribute error')
