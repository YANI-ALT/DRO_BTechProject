# plot and see the iris dataset for the classification

from sklearn import datasets
import matplotlib.pyplot as plt


iris=datasets.load_iris()
X = iris.data[:, :2] # we only take the first two features.
y = iris.target

mask=y!=1  # take the labels 0,2

X=X[mask]
y=y[mask]

print("======== IRIS DATASET ========")
print("Dataset size : ",len(X))
print("X.shape :",X.shape)
print("y labels : ",set(y))
# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.savefig("Plots/Iris_dataset.png")
