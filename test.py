# from main import w1, w2, percent, x_test, y_test
from data_process import cleanData
import pandas as pd
import numpy as np


def sig(x, deriv=False):
	if deriv == True:
		return sig(x)*(1-sig(x))

	return 1/(1+np.exp(-x))

w1 = np.array(pd.read_csv('w1.csv').values)
w2 = np.array(pd.read_csv('w2.csv').values)
x = np.array(pd.read_csv('ft.csv').values)
y = np.array(pd.read_csv('lt.csv').values)

predictions = []
scores = 0
for i in range(len(x)):
	l0 = x[i]
	l1 = sig(l0.dot(w1))
	l2 = sig(l1.dot(w2))

	predictions.append([1,0]) if l2[0] > l2[-1] else predictions.append([0,1])

	if predictions[i][0] == y[i][0] and predictions[i][1] == y[i][1]:
		scores += 1 

print(scores/len(y))
	# if round(l2,2) == y[i]:
	# 	print(1)
# percent_test = 1-percent
# 

# for i in range(len(cleanData)//int(percent_test)):
# 	x = np.array([i for i in x[:int(percent*len(x))]])
# 	y = np.array([i for i in y[:int(percent*len(y))]])