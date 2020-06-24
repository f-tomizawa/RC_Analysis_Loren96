#!python3
"""
2019/8/29
This is an numpy implementation of echo state network (reservoir computing).
"""
import numpy as np
import numpy.linalg as LA
import scipy.sparse as sparse

import matplotlib.pyplot as plt
from lorenz96data import Lorenz96
from progressbar import ProgressBar
import os
import sys
import struct

class ESN():
    """
    The Echo State Network.

    args:
     - input_size           : size of the observed model space.
     - output_size          : size of the whole model space. Identical to input_size, if not designated.
     - reservoir_size       : size of the reservoir. Default 1000.
     - adjacency_density    : The density of the adjacency matrix for reservoir evolution. Default 0.5
     - spectral_radius      : Maximum magnitude of eigenvalues of adjacency matrix. Default 1.0
     - input_scale          : scale of the input-reservoir mapping matrix. Default 1.0

    Parameters:
     - A
     - W_in
     - W_out
     - reservoir_size
     - input_size
     - output_size
    """

    def __init__(self, input_size, output_size=None, reservoir_size=1000, adjacency_density=0.5, spectral_radius=1.0, input_scale=1.0):
        print('model init')
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        if output_size == None:
            self.output_size = input_size
        else:
            self.output_size = output_size

        self.A = sparse.rand(reservoir_size, reservoir_size, density=adjacency_density)
        self.A = np.array(self.A.todense())
        self.A = self.A * spectral_radius / (np.abs(LA.eigvals(self.A)).max())

        self.W_in = np.zeros((reservoir_size, input_size))
        q = int(reservoir_size / input_size)
        for i in range(input_size):
            np.random.seed(seed=i)
            self.W_in[i*q:(i+1)*q, i] = input_scale * (np.random.rand(q) * 2 - 1)

    def forward(self, input_vector, previous_state=None):
        """
        Compute the reservoir states of next timestep.

        args:
            - input_vector      : model state at current timestep.
            - previous_state    : reservoir state at current timestep.
        """
        if previous_state is None:
            previous_state = np.zeros((self.reservoir_size, 1))

        # reshape vectors to be column vectors.
        input_vector = input_vector.copy().reshape(-1, 1)
        previous_state = previous_state.copy().reshape(-1, 1)

        # compute the reservoir state at next timestep.
        r = np.tanh(self.A @ previous_state + self.W_in @ input_vector)

        # returns the reservoir state vector as a row vector.
        return r.reshape(-1).copy()

    def train(self, observed_data, target_data=None, washout=0, ridge_param=0.01):
        """
        Train the network with input data.
        args:
        - observed_data     : input matrix (npoints * length). Each columns stands for the state at each time.
        - target_data       : output matrix. Designate if input data (observed data) is imperfect.
        - washout           : timesteps to drop.
        - ridge_param       : parameter for ridge regression
        """
        if type(target_data) == type(None):
            target_data = observed_data[:, 1:]
            observed_data = observed_data[:, :-1]

        # arguments assertion
        if observed_data.shape[0] != self.input_size:
            raise ValueError('Observed data size is set to {}, not {}'.format(self.input_size, observed_data.shape[0]))
        if target_data.shape[0] != self.output_size:
            raise ValueError('target data size is set to {}, not {}'.format(self.output_size, target_data.shape[0]))
        if observed_data.shape[1] != target_data.shape[1]:
            raise ValueError('Observed and target data duration should be the same, got {} and {}'.format(observed_data.shape[1], target_data.shape[1]))

        # calculate reservoir state
        train_length = (observed_data.shape[1])
        p = ProgressBar(0, train_length)

        reservoir = np.zeros((self.reservoir_size, train_length))
        reservoir[:, 0] = self.forward(observed_data[:, 0])
        for i in range(1, train_length):
            p.update(i) # progressbar
            reservoir[:, i] = self.forward(observed_data[:, i], reservoir[:, i-1])
        reservoir_last = reservoir[:, -1].copy()
        print()

        # nonlinear transformation
        reservoir = self.nonlinear_transf(reservoir, inplace=True)

        # washout and target data setting
        reservoir = reservoir[:, washout:]
        target = target_data[:, washout:]
        
        # ridge regression
        U = reservoir @ reservoir.T + ridge_param * np.eye(self.reservoir_size)
        Uinv = LA.inv(U)
        self.W_out = (Uinv @ (reservoir @ target.T)).T
        return self.W_out, reservoir_last

    def predict(self, current_reservoir, test_observed, ptb_func=None, ptb_scale=0.0, nexttime=False, extended_interval=0):
        """
        Generate the predict time series.
        args:
        - current_reservoir     : last reservoir state in training phase
        - test_observed         : whole test data
        - nexttime              : whether to give true state every step
        - extended_interval     : interval to give the true state
        """
        # pertubation
        if ptb_func == 'normal':
            ptb = lambda: np.random.normal(loc=0, scale=ptb_scale, size=self.input_size)
        elif type(ptb_func) != type(None):
            ptb = ptb_func
        else:
            ptb = lambda: np.zeros((self.input_size))

        test_length = test_observed.shape[1]
        predict = np.zeros((self.output_size, test_length))
        p = ProgressBar(0, test_length)

        # initial predict
        current_reservoir = self.forward(test_observed[:, 0] + ptb(), current_reservoir)
        reservoir_transf = self.nonlinear_transf(current_reservoir, inplace=False)
        predict[:, 0] = (self.W_out @ reservoir_transf.reshape(-1,1)).reshape(-1)

        # predict iteration
        for i in range(1, test_length):
            p.update(i) # progressbar
            if nexttime:
                current_reservoir = self.forward(test_observed[:, i] + ptb(), current_reservoir)
            elif extended_interval > 0 and i % extended_interval == 0:
                current_reservoir = self.forward(test_observed[:, i-100])
                for j in range(99, 0, -1):
                    current_reservoir = self.forward(test_observed[:, i-j], current_reservoir)
                current_reservoir = self.forward(test_observed[:, i] + ptb(), current_reservoir)
            else:
                current_reservoir = self.forward(predict[:, i-1], current_reservoir)
            reservoir_transf = self.nonlinear_transf(current_reservoir, inplace=False)
            predict[:,i] = (self.W_out @ reservoir_transf.reshape(-1,1)).reshape(-1)
        return predict

    def nonlinear_transf(self, matrix, inplace=False):
        """
        Apply nonlinear row transformation to input matrix.
        """
        if not inplace:
            matrix = matrix.copy()
        row_pre = matrix[0].copy()
        for i in range(2, matrix.shape[0], 2):
            row_tmp = matrix[i].copy()
            matrix[i] = (matrix[i-1] * row_pre).copy()
            row_pre = row_tmp.copy()
        return matrix

if __name__ == "__main__":
    
    print('initializing')
    nexttime = False

    input_size = 8
    output_size = 8
    obs_thinning_step = int(output_size / input_size)
    time_thinning_step = 1

    total_length = 120000
    train_length = 100000
    test_length = 10000
    reservoir_size = 5000
    washout = 0
    ridge_param = 0.0001

    path = './nature.dat'

    print("read data from " + path)
    binaryFormat = '{}f'.format(int(os.path.getsize(path)/4)) #floatは1データあたり4byte
    binData = open(path, 'rb').read()
    ascData = struct.unpack(binaryFormat, binData)

    nature = list(filter(lambda v: abs(v) > 1e-40, ascData))
    nature = Lorenz96(nature, nsteps=total_length*time_thinning_step, npoints=output_size)
    nature = nature.getData(step=time_thinning_step).T

    train_data = nature[::obs_thinning_step, :train_length]
    target_data = nature[:, 1:train_length+1]
    test_observed = nature[::obs_thinning_step, train_length:train_length+test_length]
    test_target = nature[:, train_length+1:train_length+test_length+1]

    model = ESN(input_size=input_size,
             output_size=output_size,
             reservoir_size=reservoir_size,
             adjacency_density=0.0006,
             spectral_radius=0.1,
             input_scale=0.5)
    
    print('training')
    W_out, reservoir = model.train(train_data, target_data=target_data, washout=washout, ridge_param=ridge_param)

    print('testing')
    predict = model.predict(reservoir, test_observed, nexttime=nexttime)

    predict = Lorenz96(predict.T.flatten(), nsteps=test_length, npoints=8)
    test_target = Lorenz96(test_target.T.flatten(), nsteps=test_length, npoints=8)

    diff = predict - test_target

    predict.hovmoller(title="Hovmoller Diagram of ESN Predicted Run", savepath="./predict.png", end=1000)
    test_target.hovmoller(title="Hovmoller Diagram of Nature Run", savepath="nature.png", end=1000)
    diff.hovmoller(title="Hovmoller Diagram of the Difference", savepath="diff.png", end=1000)
    plt.show()
    rmse = np.mean((diff.getData() ** 2), axis=1) ** (1/2)
    plt.plot(range(1000), rmse[:1000])
    plt.title('RMSE of ESN predicted run')
    plt.savefig('./rmse.png')

    print('done')
