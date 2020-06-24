#!python3
"""
This is the data class for lorenz96 results.
"""

import numpy as np
import os, struct
import matplotlib.pyplot as plt

def readData(filepath, nsteps, oneday=-1, npoints=-1, nens=1):
    if not (os.path.isfile(filepath)):
        raise FileNotFoundError("No such file: '{}'".format(filepath))
    with open(filepath, "r") as f:
        data = f.read().split('\n')[:-1]
    for i in range(len(data)):
        data[i] = list(map(float, data[i].split(',')))
    return Lorenz96(data, nsteps, oneday=oneday, npoints=npoints, nens=nens)

def readBinary(path, precision, nsteps, oneday=-1, npoints=-1, nens=1):
    binaryFormat = '{}f'.format(int(os.path.getsize(path)/precision)) #floatは1データあたり4byte
    binData = open(path, 'rb').read()
    ascData = struct.unpack(binaryFormat, binData)
    data = list(filter(lambda v: abs(v) > 1e-40, ascData))
    return Lorenz96(data, nsteps, oneday=oneday, npoints=npoints, nens=nens)

class Lorenz96():
    """
    The Data Class of Time Series data. \\ 
    Data may include some ensembles, and some grids.


    Parameters:
    - ndays     : Number of days (length of the simulation)
    - oneday    : Number of outputs per day
    - data      : Data sequence (1d vector or 2d array(time * points))
    """
    def __init__(self, data, nsteps, oneday=-1, npoints=-1, nens=1):
        self.data = np.array(data).flatten()
        self.nsteps = int(nsteps)
        self.oneday = int(oneday)
        self.npoints = int(npoints)
        self.nens = int(nens)

        if (self.npoints < 0):
            self.npoints = int(len(self.data) / nsteps / nens)
        if (len(self.data) != self.npoints * self.nsteps * self.nens):
            raise ValueError('data length should be {} * {} * {}, not {}'.format(self.nsteps, self.npoints, self.nens, len(self.data)))


    def __add__(self, other):
        if (len(self.data) != len(other.data)):
            raise ValueError('operands must be the same size, got {} and {}'.format(self.getData().shape, other.getData().shape))
        return Lorenz96(self.data+other.data, nsteps=self.nsteps, oneday=self.oneday, nens=self.nens)

    def __sub__(self, other):
        if (len(self.data) != len(other.data)):
            raise ValueError('operands must be the same size, got {} and {}'.format(self.getData().shape, other.getData().shape))
        return Lorenz96(self.data-other.data, nsteps=self.nsteps, oneday=self.oneday, nens=self.nens)

    def getData(self, start=0, end=-1, step=1, ens=0):
        """
        Method for extracting the data.

        Returns:
            2d Numpy array (timesteps * npoints)

        Arguments:
        - start : Start timestep
        - end  : End timestep
        - step  : Number of step to skip (e.g. step=2 -> t=0,2,4,6,...)
        - ens   : The index of ensemble to extract. -1 for all ensembles.
        """
        if ((type(start) != int) or (type(end) != int) or (type(step) != int) or (type(ens) != int)):
            raise TypeError('indices must be integers')
        if ((ens < 0) or (ens >= self.nens)):
            raise ValueError('ensemble index out of range')

        dataidx = lambda time, ens: time * self.nens * self.npoints + ens * self.npoints
        res = []
        start, end, step = slice(start, end, step).indices(self.nsteps+1)
        for i in range(start, end, step):
            res.append(self.data[dataidx(i,ens):dataidx(i,ens)+self.npoints])
        return np.array(res)

    def saveData(self, savepath="", start=0, end=-1, step=1, ens=0):
        if (os.path.isfile(savepath)):
            raise FileExistsError("File exists: '{}'".format(savepath))
        if not (os.path.isdir(os.path.dirname(savepath))):
            os.makedirs(os.path.dirname(savepath))
        data = self.getData(start=start, end=end, step=step, ens=ens).tolist()
        with open(savepath, "w") as f:
            for state in data:
                f.write(','.join(map(str, state)))
                f.write('\n')
        return None

    def hovmoller(self, start=0, end=-1, step=1, mean=False, ens=0, vmin=-3, vmax=3, figsize=(15,5), savepath=None, title="", xlabel="", ylabel="", interpolation="nearest", facecolor=None):
        """
        Generates the hovmoller diagram.

        Parameters:
        - start     : Start timestep
        - end       : End timestep
        - step      : Number of step to skip (e.g. step=2 -> t=0,2,4,6,...)
        - mean      : If True, generate the diagram of all the ensembles' mean.
        - ens       : Index of ensemble to visualize
        - vmin      : Minimum value of diagram (defalut -3)
        - vmax      : Maximum value of diagram (default 6)
        - figsize   : Figure size (default (15, 5))
        - savepath  : Path to save the figure (If not designated, figures are just shown)
        - title     : Figure Title (displayed on top)
        - xlabel    : X-axis label (displayed on bottom)
        - ylabel    : Y-axis label (displayed on left)
        """
        if mean:
            res = []
            for i in range(self.nens):
                res.append(self.getData(ens=i, start=start, end=end, step=step))
            res = np.mean(np.array(res), axis=0)
        else:
            res = self.getData(start=start, end=end, step=step, ens=ens)

        setRcParams()
        fig=plt.figure(figsize=figsize)
        plt.imshow(res.T, cmap='bwr', aspect='auto', interpolation=interpolation, vmin=vmin, vmax=vmax)
        plt.colorbar(pad=0.03)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.subplots_adjust(left=0.05, right=1.05, bottom=0.1, top=0.95)
        plt.yticks(np.arange(0, 8), range(1, 9), rotation=0)
        plt.xticks(range(0, 1000, 80), range(0, 13*2, 2))
        if savepath != None:
            plt.savefig(savepath, facecolor=facecolor)
        else:
            plt.show()
        fig.clear()
        plt.close()

def setRcParams():
    plt.rcParams['font.size'] = 20
    plt.rcParams['font.family']= 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.labelbottom'] = True
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.width'] = 1.2
    plt.rcParams['grid.linestyle']='--'
    plt.rcParams['grid.linewidth'] = 0.3
    plt.rcParams['lines.linewidth'] = 1.0

    plt.rcParams["legend.markerscale"] = 2
    plt.rcParams["legend.fancybox"] = False
    plt.rcParams["legend.framealpha"] = 1
    plt.rcParams["legend.edgecolor"] = 'black'

if __name__ == '__main__':
    PATH = 'path/to/lorenz96/output/file.dat'
    NSTEPS = 1000 # Number of steps for LETKF expriment
    NPOINTS = 40  # Number of grid points.
    NENS = 20     # NUmber of ensemble members.

    # Keep experiment data as a data class
    data = readBinary(PATH, 4, NSTEPS, NPOINTS)

    # Get the data as NumPy 2d array (timesteps * npoints)
    timeSeriesArray = data.getData()

    # Designate the time range to extract (optional)
    timeSeriesArray = data.getData(start=0, end=500)

    # Display the Hovmoller Diagram
    data.hovmoller()

    # Save the Hovmoller Diagram
    data.hovmoller(savepath="path/to/save/the/figure")
