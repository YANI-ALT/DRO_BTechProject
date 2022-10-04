import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR
# norm selection for the computation
NORM="one" # 'one','two','inf'

#datapoints
N=10
#wasserstein radi
e_wradi=0

# epsilon-insensitive loss function
epsilon=0.1# epsilon value

# dataset
df=pd.read_csv('Dummy_data.csv')
x=[[1,xi] for xi in list(df['x'])]
y=list(df['y'])

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

    plt.plot(x_predict,y_predict,color='black',label="LinearSVR")

    plt.legend(loc='lower right')
    plt.title(plot_name)
    plt.savefig(plot_name)


try:

    # Create a new model
    m = gp.Model("Support_vector_reg")
    # if(NORM=='two'):
    #     m.params.NonConvex = 2

    s_i={}
    p_i_plus={}
    p_i_minus={}
    w={}
    #hyperplane w=[w1,w2,..wn] each x=[1,x1...wn] data point
    for i in range(col):
        var_name="w"+str(i)
        w[i]=m.addVar(vtype=GRB.CONTINUOUS,name=var_name)

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
        # Add constraint
        # for j in range(col):
        #     data_constr_1_value[i].append(p_i_plus[pi_plus_name][j]*(d[j]-(C1[0][0]+C1[0][1]*x[i])-c2[0]*y[i]))
        #     data_constr_2_value[i].append(p_i_minus[pi_minus_name][j]*(d[j]-(C1[0][0]+C1[0][1]*x[i])-c2[0]*y[i]))

        constraint_name_1='data[{}]_constr1'.format(i)
        constraint_name_2='data[{}]_constr2'.format(i)

        # value of d-C1xi-c2yi
        # conic_value=[]
        intermed_value_c2y=[c2[k]*y[i] for k in range(col)]
        # for j in range(col):
        #     intermed_value_C1x=
        #     conic_value.append(d[j]-intermed_value_C1x-intermed_value_c2y)

        m.addConstr(
                    y[i]-gp.quicksum(w[j]*x[i][j] for j in range(col))
                    - epsilon
                    + gp.quicksum(p_i_plus[i,j]*(d[j]
                            - gp.quicksum(C1[j][k]*x[i][k] for k in range(col))
                            -intermed_value_c2y[j]) for j in range(col))
                    <= s_i[var_name],constraint_name_1)

        m.addConstr(
                    -y[i]+gp.quicksum(w[j]*x[i][j] for j in range(col))
                    - epsilon
                    + gp.quicksum(p_i_minus[i,j]*(d[j]
                            - gp.quicksum(C1[j][k]*x[i][k] for k in range(col))
                            -intermed_value_c2y[j]) for j in range(col))
                    <= s_i[var_name],constraint_name_2)

        # adding the dual norm constraints
        # we use the fact that 1/p+1/q=1
        # if we take the dual of 2-norm it is the 2-norm itself.

        norm_vector_p_plus=[]
        norm_vector_p_minus=[]
        for j in range(col):
            norm_vector_p_plus.append(gp.quicksum(C1[j][k]*p_i_plus[i,k] for k in range(col))+w[j])
            norm_vector_p_minus.append(gp.quicksum(C1[j][k]*p_i_minus[i,k] for k in range(col))+w[j])
        norm_vector_p_plus.append(gp.quicksum(c2[k]*p_i_plus[i,k] for k in range(col))-1)
        norm_vector_p_minus.append(gp.quicksum(c2[k]*p_i_minus[i,k] for k in range(col))+1)


        if NORM=='inf':
            abs_norm_value1={}
            abs_norm_value2={}

            # here the dual will be 1-norm=summation of absolute values of components
            for j in range(col+1):
                abs_norm_value1[j]=m.addVar(vtype=GRB.CONTINUOUS)
                abs_norm_value2[j]=m.addVar(vtype=GRB.CONTINUOUS)
                m.addConstr(abs_norm_value1[j]==norm_vector_p_plus[j])
                m.addGenConstrAbs(abs_norm_value2[j],abs_norm_value1[j])

            m.addConstr(gp.quicksum(abs_norm_value2[j] for j in range(col+1))<=lambda_var)

        elif NORM=='one':
            # here the dual will be inf-norm
            abs_norm_value1={}
            abs_norm_value2=[]
            max_norm_component=m.addVar(vtype=GRB.CONTINUOUS)
            # here the dual will be 1-norm=summation of absolute values of components
            for j in range(col+1):
                abs_norm_value1[j]=m.addVar(vtype=GRB.CONTINUOUS)
                abs_norm_value2.append(m.addVar(vtype=GRB.CONTINUOUS))
                m.addConstr(abs_norm_value1[j]==norm_vector_p_plus[j])
                m.addGenConstrAbs(abs_norm_value2[j],abs_norm_value1[j])

            m.addGenConstrMax(max_norm_component,abs_norm_value2)
            m.addConstr(max_norm_component<=lambda_var)
        elif NORM=='two':
            # here the dual will be 2-norm
            # norm_value1=(p_i_plus[pi_plus_name][0]+w_0)**2+(p_i_plus[pi_plus_name][1]+w_1)**2+(p_i_plus[pi_plus_name][0]+p_i_plus[pi_plus_name][1]-1)**2

            m.addConstr(gp.quicksum(norm_vector_p_plus[k]*norm_vector_p_plus[k] for k in range(col+1))<=lambda_var*lambda_var,'norm-constr_1_data[{}]'.format(i))

            m.addConstr(gp.quicksum(norm_vector_p_minus[k]*norm_vector_p_minus[k] for k in range(col+1))<=lambda_var*lambda_var,'norm-constr_2_data[{}]'.format(i))
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


    plot_name="Plots/Support_vector_reg_v2_plot_NORM_{}_wassradi_{}_epsil_{}.png".format(NORM,e_wradi,epsilon)
    plot_data(list(df['x']),y,w_sol,plot_name)

    # print(m.display())
    m.write('Support_vector_reg.lp')


    # for v in m.getVars():
    #     print('%s %g' % (v.varName, v.x))


except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ': ' + str(e))

except AttributeError:
    print('Encountered an attribute error')
