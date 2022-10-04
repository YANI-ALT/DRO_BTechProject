import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import matplotlib.pyplot as plt
# norm selection for the computation
NORM="two" # 'one','two','inf'

#datapoints
N=10
#wasserstein radi
e_wradi=0.8

# epsilon-insensitive loss function
epsilon=0.1# epsilon value

# the conic defined for the data C1x+c2y<=d
C1=[[1,0],[0,1]]
c2=[1,1]
d=[4,4]

# dataset
df=pd.read_csv('Dummy_data.csv')
x=list(df['x'])
y=list(df['y'])

# both x and y should be of equal number one for each data point
assert(len(x)==len(y))
N=len(x)

def get_line_pts(w):
    x=[]
    y=[]
    for xi in range(0,5):
        yi=w[1]*xi+w[0]
        y.append(yi)
        x.append(xi)

    return x,y

def plot_data(x,y,w,plot_name):
    plt.scatter(x,y)
    x_line,y_line=get_line_pts(w)
    plt.plot(x_line,y_line)

    plt.savefig(plot_name)


try:

    # Create a new model
    m = gp.Model("Support_vector_reg")
    # if(NORM=='two'):
    #     m.params.NonConvex = 2

    s_i={}
    p_i_plus={}
    p_i_minus={}

    #hyperplane w=[w1,w2] each x=[1,x1] data point
    w_0=m.addVar(vtype=GRB.CONTINUOUS,name='w0')
    w_1=m.addVar(vtype=GRB.CONTINUOUS,name='w1')

    #lambda variable
    lambda_var=m.addVar(vtype=GRB.CONTINUOUS,name='Lambda')
    # Create variables

    # create pi+,pi-,si
    for i in range(N):
        var_name='s_'+str(i)
        pi_plus_name='p_'+str(i)+'_plus'
        pi_minus_name='p_'+str(i)+'_minus'

        s_i[var_name]=m.addVar(vtype=GRB.CONTINUOUS,name=var_name)
        p_i_plus[pi_plus_name]=[]
        p_i_minus[pi_minus_name]=[]

        p_i_plus[pi_plus_name].append(m.addVar(vtype=GRB.CONTINUOUS,name=pi_plus_name+"[0]"))
        p_i_plus[pi_plus_name].append(m.addVar(vtype=GRB.CONTINUOUS,name=pi_plus_name+"[1]"))

        p_i_minus[pi_minus_name].append(m.addVar(vtype=GRB.CONTINUOUS,name=pi_minus_name+"[0]"))
        p_i_minus[pi_minus_name].append(m.addVar(vtype=GRB.CONTINUOUS,name=pi_minus_name+"[1]"))

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
        for j in range(2):
            data_constr_1_value[i].append(p_i_plus[pi_plus_name][j]*(d[j]-(C1[0][0]+C1[0][1]*x[i])-c2[0]*y[i]))
            data_constr_2_value[i].append(p_i_minus[pi_minus_name][j]*(d[j]-(C1[0][0]+C1[0][1]*x[i])-c2[0]*y[i]))

        constraint_name_1='data[{}]_constr1'.format(i)
        constraint_name_2='data[{}]_constr2'.format(i)

        m.addConstr(y[i]-w_0-w_1*x[i]-epsilon+data_constr_1_value[i][0]+data_constr_1_value[i][1]<=s_i[var_name],constraint_name_1)
        m.addConstr(-y[i]+w_0+w_1*x[i]-epsilon+data_constr_2_value[i][0]+data_constr_2_value[i][1]<=s_i[var_name],constraint_name_2)


        # adding the dual norm constraints
        # we use the fact that 1/p+1/q=1
        # if we take the dual of 2-norm it is the 2-norm itself.

        if NORM=='inf':
            # here the dual will be 1-norm=summation of absolute values of components

            m.addConstr((p_i_plus[pi_plus_name][0]+w_0) + (p_i_plus[pi_plus_name][1]+w_1) + (p_i_plus[pi_plus_name][0]+p_i_plus[pi_plus_name][1]-1)<=lambda_var)
            m.addConstr((p_i_plus[pi_plus_name][0]+w_0) + (p_i_plus[pi_plus_name][1]+w_1) - (p_i_plus[pi_plus_name][0]+p_i_plus[pi_plus_name][1]-1)<=lambda_var)
            m.addConstr((p_i_plus[pi_plus_name][0]+w_0) - (p_i_plus[pi_plus_name][1]+w_1) + (p_i_plus[pi_plus_name][0]+p_i_plus[pi_plus_name][1]-1)<=lambda_var)
            m.addConstr((p_i_plus[pi_plus_name][0]+w_0) - (p_i_plus[pi_plus_name][1]+w_1) - (p_i_plus[pi_plus_name][0]+p_i_plus[pi_plus_name][1]-1)<=lambda_var)

            m.addConstr(-(p_i_plus[pi_plus_name][0]+w_0) + (p_i_plus[pi_plus_name][1]+w_1) + (p_i_plus[pi_plus_name][0]+p_i_plus[pi_plus_name][1]-1)<=lambda_var)
            m.addConstr(-(p_i_plus[pi_plus_name][0]+w_0) + (p_i_plus[pi_plus_name][1]+w_1) - (p_i_plus[pi_plus_name][0]+p_i_plus[pi_plus_name][1]-1)<=lambda_var)
            m.addConstr(-(p_i_plus[pi_plus_name][0]+w_0) - (p_i_plus[pi_plus_name][1]+w_1) + (p_i_plus[pi_plus_name][0]+p_i_plus[pi_plus_name][1]-1)<=lambda_var)
            m.addConstr(-(p_i_plus[pi_plus_name][0]+w_0) - (p_i_plus[pi_plus_name][1]+w_1) - (p_i_plus[pi_plus_name][0]+p_i_plus[pi_plus_name][1]-1)<=lambda_var)

        elif NORM=='one':
            # here the dual will be inf-norm
            pass
        elif NORM=='two':
            # here the dual will be 2-norm
            norm_value1=(p_i_plus[pi_plus_name][0]+w_0)**2+(p_i_plus[pi_plus_name][1]+w_1)**2+(p_i_plus[pi_plus_name][0]+p_i_plus[pi_plus_name][1]-1)**2
            # m.addConstr(norm_value1<=lambda_var*lambda_var,'norm-constr_1_data[{}]'.format(i))

            norm_value2=(p_i_minus[pi_minus_name][0]-w_0)**2+(p_i_minus[pi_minus_name][1]-w_1)**2+(p_i_minus[pi_minus_name][0]+p_i_minus[pi_minus_name][1]+1)**2
            # m.addConstr(norm_value2<=lambda_var*lambda_var,'norm-constr_2_data[{}]'.format(i))
    # edge_constr_s=m.addConstr(variable_dict_PD2['X_s3']+variable_dict_PD1['X_s2']==1,'edge_constr-s')

    #positive contraints

    for i in range(N):
        var_name='s_'+str(i)

        m.addConstr(s_i[var_name]>=0,var_name+">=0")

        for j in range(2):
            pi_plus_name='p_'+str(i)+'_plus' # example : p_2_plus[0]>=0, p_2_plus[1]>=0
            pi_minus_name='p_'+str(i)+'_minus'

            m.addConstr(p_i_plus[pi_plus_name][j]>=0,pi_plus_name+'['+str(j)+']>=0') # these belong to positive orthant
            m.addConstr(p_i_minus[pi_minus_name][j]>=0,pi_minus_name+'['+str(j)+']>=0')


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
            break


    plot_name="Plots/Support_vector_reg_plot_NORM_{}_wassradi_{}_epsil_{}.png".format(NORM,e_wradi,epsilon)
    plot_data(x,y,w_sol,plot_name)

    # print(m.display())
    m.write('Support_vector_reg.lp')


    # for v in m.getVars():
    #     print('%s %g' % (v.varName, v.x))


except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ': ' + str(e))

except AttributeError:
    print('Encountered an attribute error')
