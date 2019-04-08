from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd

lr = linear_model.LinearRegression()
boston = datasets.load_boston()
y = boston.target

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:

# predicted = cross_val_predict(lr, boston.data, y, cv=10)
#
# fig, ax = plt.subplots()
# ax.scatter(y, predicted, edgecolors=(0, 0, 0))
# ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
# ax.set_xlabel('Measured')
# ax.set_ylabel('Predicted')
# plt.show()
#
bost = pd.DataFrame(boston.data)


print(bost.head())
print("**************************")

PRICE = []
CRIM = []
ZN = []
INDUS = []
CHAS = []
NOX = []
RM = []
AGE = []
DIS = []
RAD = []
TAX = []
PTRATIO = []
B = []
LSTAT = []

length = len(boston.target)
for i in range(0, length):
    PRICE.append(boston.target[i])
    CRIM.append(boston.data[i, 0])
    ZN.append(boston.data[i, 1])
    INDUS.append(boston.data[i, 2])
    CHAS.append(boston.data[i, 3])
    NOX.append(boston.data[i, 4])
    RM.append(boston.data[i, 5])
    AGE.append(boston.data[i, 6])
    DIS.append(boston.data[i, 7])
    RAD.append(boston.data[i, 8])
    TAX.append(boston.data[i, 9])
    PTRATIO.append(boston.data[i, 10])
    B.append(boston.data[i, 11])
    LSTAT.append(boston.data[i, 12])
print("################")

plt.scatter(CRIM, PRICE, color='blue', linewidth=3)

# predicted = cross_val_predict(lr, CRIM, PRICE, cv=10)
#
# fig, ax = plt.subplots()
# ax.scatter(PRICE, CRIM, edgecolors=(0, 0, 0))
# ax.set_xlabel('Measured')
# ax.set_ylabel('Predicted')
plt.show()

def my_function():
  print("Hello from a function")

def plotRegession(boston_x):
  bos = load_boston()
  print(bos.data.shape)
  boston_x = bos.data[:, np.newaxis, 2]
  boston_x_train = boston_x[:-20]
  boston_x_test = boston_x[-20:]
  boston_y_train = bos.target[:-20]
  boston_y_test = bos.target[-20:]
  regr = linear_model.LinearRegression()
  regr.fit(boston_x_train, boston_y_train)
  boston_y_pred = regr.predict(boston_x_test)
  plt.scatter(boston_x_test, boston_y_test, color='black')
  plt.plot(boston_x_test, boston_y_pred, color='blue', linewidth=3)
  plt.xticks(())
  plt.yticks(())
  plt.show()
