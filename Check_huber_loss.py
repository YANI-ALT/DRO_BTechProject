NORM=2
x=[1.191777387220301, 0.4627005000636914, 1.5799140823857198, 1.1743457294376425]
y=[1.3234256162318792, 1.370665138887607, 1.9653338675648908, 1.4338327557664814]

e_wradi=0
delta=1

obj_=3.61394638e-02

w0=1.12438
w1=0.532278

z=[0]*4
z[0]=0.435311
# abs_0 4.724e-13
# C5 4.724e-13
z[1]=9.10235e-10
# abs_1 4.09872e-13
# C8 4.09872e-13
z[2]=1.55279e-11
# abs_2 3.93694e-13
# C11 3.93694e-13
z[3]=0.315626
# abs_3 4.70147e-13
# C14 4.70147e-13

#calculate the obj

calc_obj=0.0

assert(len(x)==len(y))
N=len(x)
for i in range(len(x)):
    calc_obj=(0.5*(z[i]**2))+delta*abs(w1*x[i]+w0-y[i]-z[i])

calc_obj=(1/N)*calc_obj
print("CALCULATED OBJ: ",calc_obj)
print("GUROBI OBJ: ",obj_)
