from sklearn.datasets import load_digits
import pandas as pd
import numpy as np
# digits = load_digits(as_frame=True)

def get_data(targets):
    # will return the data frames for the particular targets
    digits = load_digits(as_frame=True)
    target_data=digits.target
    pixel_data=digits.data
    all_data=pd.concat([pixel_data,target_data],axis=1)
    data_to_use=all_data[all_data["target"]==targets[0]]
    for t in targets[1:]:
        data_to_use=pd.concat([data_to_use,all_data[all_data["target"]==t]])
    # print(data_to_use)
    for col in data_to_use.columns[:-1]:
        mu=data_to_use[col].mean()
        sigma=data_to_use[col].std()
        data_to_use[col]=(data_to_use[col]-mu)/sigma
        data_to_use[col]= data_to_use[col].fillna(0)
    # print(data_to_use)
    data_matrix=data_to_use.to_numpy()
    X=data_matrix[:,:-1]
    X=np.c_[np.ones(X.shape[0]), X]
    y=data_matrix[:,-1]
    # print(y)
    print("X : {} y : {}".format(X.shape,y.shape))
    y[y==targets[0]]=-1
    y[y==targets[1]]=1
    
    return X,y

data=get_data([1,7])
