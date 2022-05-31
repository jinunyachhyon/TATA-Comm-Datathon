# univariate lstm
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

import pandas as pd
import matplotlib.pyplot as plt

# specify training data
df = pd.read_csv('/Users/nyach/Downloads/Train Set - Train Set.csv')
#print(df.head())

#renaming index col
df_month= df.rename(columns = {'Region_Name': 'Month'}, inplace = False)
#print(df_month)

#remove index
df_month=df_month.set_index('Month')
#print(df_month)

#transpose data
df_transpose=df_month.transpose()
#print(df_transpose)

pred = pd.DataFrame()

for w in range(0,3915):
	region= ("Region "+ str(w+1))
	#print(region)

	# preparing independent and dependent features
	def prepare_data(timeseries_data, n_features):
		X, y =[],[]
		for i in range(len(timeseries_data)):
			# find the end of this pattern
			end_ix = i + n_features
			# check if we are beyond the sequence
			if end_ix > len(timeseries_data)-1:
				break
			# gather input and output parts of the pattern
			seq_x, seq_y = timeseries_data[i:end_ix], timeseries_data[end_ix]
			X.append(seq_x)
			y.append(seq_y)
		return np.array(X), np.array(y)

	#taking array of the region
	timeseries_data=df_transpose[[region]].to_numpy()
	#print(timeseries_data)
	# choose a number of time steps
	n_steps = 12
	# split into samples
	X, y = prepare_data(timeseries_data, n_steps)

	#print(X),print(y)
	#print(X.shape)

	# reshape from [samples, timesteps] into [samples, timesteps, features]
	n_features = 1
	X = X.reshape((X.shape[0], X.shape[1], n_features))
	print(X.shape)
	##LSTM model requires 3d thus converting data into 3d

	###Building LSTM model
	# define model
	model = Sequential()
	model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
	model.add(LSTM(50, activation='relu'))
	model.add(Dense(1))
	model.compile(optimizer='adam', loss='mse')
	# fit model
	model.fit(X, y, epochs=500, verbose=1)


	#Predicting For the next 15 data
	# demonstrate prediction for next 15 months
	timeseries=df_transpose[[region]].values
	ts=timeseries[-12:]
	#print(ts)
	x_input = ts.flatten()
	print(x_input)
	temp_input = list(x_input)
	lst_output = []
	i = 0
	while (i < 15):

		if (len(temp_input) > 12):
			x_input = np.array(temp_input[1:])
			print("{} month input {}".format(i, x_input))
			#print(x_input)
			x_input = x_input.reshape((1, n_steps, n_features))
			#print(x_input)
			yhat = model.predict(x_input, verbose=0)
			print("{} month output {}".format(i, yhat))
			temp_input.append(yhat[0][0])
			temp_input = temp_input[1:]
			#print(temp_input)
			lst_output.append(yhat[0][0])
			i = i + 1
		else:
			x_input = x_input.reshape((1, n_steps, n_features))
			yhat = model.predict(x_input, verbose=0)
			print(yhat[0])
			temp_input.append(yhat[0][0])
			lst_output.append(yhat[0][0])
			i = i + 1

	print(lst_output)

	z=np.array(lst_output)
	col_name = 'Region ' + str(w+1)
	pred[col_name]=z
	#pred.loc[:, col_name] = z

	print(pred)
	pred.to_csv("Predicted.csv")