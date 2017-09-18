import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

def read_and_trim_training_data_from_csv(filename):
	table = []
	rows = open(filename, 'r').read().split('\r\n')
	for r in rows:
		nextrow = r.split(',')
		if nextrow[1] == '':
			break
		else:
			table.append(nextrow)
	return table[1:] #ignore header

def read_testing_x_data_from_csv(filename):
	x_data = []
	rows = open(filename, 'r').read().split('\r\n')
	for r in rows:
		nextrow = r.split(',')
		if len(nextrow) == 3 and nextrow[1] == '':
			x_data.append(nextrow[2])
	return x_data

def export(predictions, filename):
	outfile = open(filename.replace('.csv', '')+'_predictions.csv', 'w')
	inrows = open(filename, 'r').read().split('\r\n')
	i=0
	outfile.write('Date,Building kWh Usage,Temp Actual and Forecast\n')
	for r in inrows:
		nextrow = r.split(',')
		if len(nextrow) == 3 and nextrow[1] == '':
			i += 1
			nextrow[1] = str(predictions[i-1][0])
			outfile.write(','.join(nextrow)+'\n')
	outfile.close()

filename = 'Data_Test.csv'

# Load and trim the training dataset
data = read_and_trim_training_data_from_csv(filename)

data_X = np.array([d[2] for d in data]).astype(np.float).reshape(-1,1)
data_Y = np.array([d[1] for d in data]).astype(np.float).reshape(-1,1)

regr = linear_model.LinearRegression()

#train our model
regr.fit(data_X, data_Y)

testing_x = np.array(read_testing_x_data_from_csv(filename)).astype(np.float).reshape(-1,1)
data_y_pred = regr.predict(testing_x)
export(data_y_pred, filename)

#plot training data
plt.scatter(data_X, data_Y,  color='black')
#plot line of best fit (modeling our predictions)
plt.plot(testing_x, data_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())
plt.show()