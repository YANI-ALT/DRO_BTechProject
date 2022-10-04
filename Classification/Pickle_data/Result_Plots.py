import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def generate_plot(dataFrame,data_col,name,plot_name):
    plt.figure(figsize=(10, 10))
    sns.barplot(x="param-config",y=data_col,hue="Unnamed: 0",data=dataFrame)
    plt.xticks(rotation = 45)
    plt.title(data_col+" "+name)
    plt.legend(loc="lower right")
    # plt.show()

    plt.savefig(data_col+"_"+name+"_"+plot_name)

def process_dataFrame(dataFrame):

    # name=dataFrame["Unnamed: 0"]
    # norm=dataFrame["NORM"]
    # wasserstein_radi=dataFrame["wasserstein_radi"]
    # kappa=dataFrame["kappa"]
    param_config=[]
    optim_approach=list(dataFrame["Unnamed: 0"].unique())
    extractedData={}
    for op_ap in optim_approach:
        df_temp=dataFrame[dataFrame["Unnamed: 0"]==op_ap]
        test_data=df_temp["Test-Data"].values
        average=df_temp["Average"].values
        if len(param_config)==0:
            #wasserstein_radi NORM  kappa
            param_config=list(zip(df_temp["wasserstein_radi"],df_temp["NORM"],df_temp["kappa"]))

        extractedData[op_ap+"_TestPerformance"]=list(test_data)
        extractedData[op_ap+"_5CV"]=list(average)

    return extractedData,param_config


file="5CV_MNIST_NORM_['two']_wassradi_[0.1]_kappa_[0.1, 0.25, 0.5, 0.75, 1]"
# file="5CV_MNIST_NORM_['one']_wassradi_[0.1, 0.5]_kappa_[0.1, 0.25, 0.5, 0.75, 1].csv"
dataFrame=pd.read_csv(file+".csv")
dataFrame["param-config"]=list(zip(dataFrame["NORM"],dataFrame["wasserstein_radi"],dataFrame["kappa"]))
# extractedData,param_config=process_dataFrame(dataFrame)
plot_name=file+".png"
generate_plot(dataFrame,"Test-Data","[NORM,wasserstein_radi,kappa]",plot_name)
generate_plot(dataFrame,"Average","[NORM,wasserstein_radi,kappa]",plot_name)
# print(pd.DataFrame(extractedData))
