from esn import ESN
from lorenz96data import Lorenz96, readBinary
import struct, os, sys
import matplotlib.pyplot as plt
import numpy as np

nexttime = False

input_size = 8
output_size = 8
obs_thinning_step = int(output_size / input_size)
space_thinning_step = 1
time_thinning_step = 1

total_length = 200010
train_length = 100000
test_length = 100000
reservoir_size = 5000
washout = 0
ridge_param = 0.0001

path = './anlmean.dat'

data = readBinary(path, precision=4, nsteps=total_length*time_thinning_step, npoints=output_size*space_thinning_step)
data = data.getData(step=time_thinning_step).T[::space_thinning_step, :]

train_data = data[::obs_thinning_step, :train_length]
target_data = data[:, 1:train_length+1]
test_observed = data[::obs_thinning_step, train_length:train_length+test_length]
test_target = data[:, train_length+1:train_length+test_length+1]

model = ESN(input_size=input_size,
         output_size=output_size,
         reservoir_size=reservoir_size,
         adjacency_density=0.0006,
         spectral_radius=0.1,
         input_scale=0.5)
    
W_out, reservoir = model.train(train_data, target_data=target_data, washout=washout, ridge_param=ridge_param)

predict = model.predict(reservoir, test_observed, ptb_func=None, ptb_scale=1.0, nexttime=nexttime, extended_interval=1000)

predict = Lorenz96(predict.T.flatten(), nsteps=test_length, npoints=output_size)
test_target = Lorenz96(test_target.T.flatten(), nsteps=test_length, npoints=output_size)

predict.saveData(savepath="./predict.csv")
test_target.saveData(savepath="./nature.csv")

diff = predict - test_target

predict.hovmoller(title="Hovmoller Diagram of ESN Predicted Run", savepath="./predict.png", end=1000)
test_target.hovmoller(title="Hovmoller Diagram of Nature Run", savepath="nature.png", end=1000)
diff.hovmoller(title="Hovmoller Diagram of the Difference", savepath="diff.png", end=2000)
plt.show()
rmse = np.mean((diff.getData()[:,::obs_thinning_step] ** 2), axis=1) ** (1/2)
plt.plot(range(1000), rmse[:1000])
plt.title('RMSE of ESN predicted run (Observed points only)')
plt.savefig('./rmse.png')
