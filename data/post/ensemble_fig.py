#!python3
# This is the code to visualize the expr1 result

import struct, os
import numpy as np
import matplotlib.pyplot as plt
from lorenz96data import Lorenz96, readData, readBinary

def drawRMSEfig(rmse_series, dirpath):
    # generate the boxplot of every 4 relative timesteps.
    plt.boxplot(rmse_series.tolist()[:500:10], whis=2.0, sym='.')
    #plt.ylim([0.0,7.5])
    plt.xticks(np.arange(1, 51, 4), range(0, 500, 40), rotation=0)
    plt.title('variation of "RMSE"s at each time.')
    plt.xlabel('Timesteps')
    plt.ylabel('RMSE - deviation from nature')
    plt.savefig(dirpath)
    return

def readRC(dirpath, nsteps, npoints, nens, naturepath="./nature8.csv"):
    # naturepath = os.path.join(dirpath, 'nature.csv')
    predictpath = os.path.join(dirpath, 'predict.csv')
    nature = readData(naturepath,nsteps=nsteps, npoints=npoints, nens=nens) 
    predict = readData(predictpath,nsteps=nsteps, npoints=npoints, nens=nens) 
    return predict, nature

def readLETKF(predictdir, naturepath, naturesteps, dur, num, precision=4, npoints=-1, nens=1):
    nature = readBinary(naturepath, precision, nsteps=naturesteps, npoints=npoints, nens=1) 
    predict_list = list()
    for n in range(num):
        path = os.path.join(predictdir, 'simrun{0:03}.dat'.format(n+1))
        predict_list.append(readBinary(path, 4, nsteps=dur, npoints=npoints, nens=nens))
    return predict_list, nature

def setRcParams():
    plt.rcParams['font.size'] = 11
    plt.rcParams['font.family']= 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.width'] = 1.2
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['axes.grid']=True
    plt.rcParams['grid.linestyle']='--'
    plt.rcParams['grid.linewidth'] = 0.3
    plt.rcParams['lines.linewidth'] = 1.0

    plt.rcParams["legend.markerscale"] = 2
    plt.rcParams["legend.fancybox"] = False
    plt.rcParams["legend.framealpha"] = 1

class RmseMeanFig():
    def __init__(self, npoints=1, duration=100, figsize=(6.4,4.4), linewidth=1.0, facecolor="#ffffff", cmap=None, linenum=1):
        setRcParams()
        self.fig = plt.figure(figsize=figsize, facecolor=facecolor)
        self.ax = self.fig.subplots()
        self.npts = npoints
        self.dur = duration
        self.numOfFigs = 0
        self.linewidth = linewidth
        self.facecolor=facecolor
        self.cmap = plt.get_cmap(cmap)
        self.linenum = linenum-1+1
        self.idx = -1+1

    def set_figure(self, title='', xlabel='', ylabel='', ylim=None, legend_loc=None, legend_title='', legend_col=1):
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_ylim(ylim)
        self.ax.set_xlim(0, self.dur)
        # plt.xticks(range(0, 1000, 80), range(0, 13*2, 2))
        self.ax.legend(loc=legend_loc, title=legend_title, ncol=legend_col, fontsize=11)
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.135)

    def show(self):
        plt.show()
    def savefig(self, path):
        plt.savefig(path, facecolor=self.facecolor)
    
    def add_letkf(self, predictdir, naturepath, naturesteps=200010, naturestart=100000, local_dur=0, nens=1, num=0, step=1, label="", color=None, linestyle=None):
        self.idx += 1
        if local_dur == 0:
            local_dur = self.dur
        predict_list, nature = readLETKF(predictdir, naturepath, naturesteps=naturesteps, dur=local_dur, num=num, npoints=self.npts, nens=nens)
        rmse_series = np.zeros((local_dur, num*nens))
        naturestart += 1
        for n in range(num):
            # nature_cut = nature.getData(start=n, end=n+local_dur)
            #ens = np.zeros((local_dur, self.npts))
            #for i in range(nens):
            #    ens = predict_list[n].getData(ens=i)
            #    rmse_series[:,n*nens+i] = np.sqrt(((ens - nature_cut) ** 2).mean(axis=1))
            predict_cut = predict_list[n].getData()
            nature_cut = nature.getData(start=n*local_dur+naturestart, end=(n+1)*local_dur+naturestart)
            rmse_series[:,n] = np.sqrt(((predict_cut - nature_cut) ** 2).mean(axis=1))
        rmse_series_mean = rmse_series.mean(axis=1)

        # self.ax.plot(range(0, self.dur, int(self.dur/local_dur)*step), rmse_series_mean[:int(local_dur/step)], xunits=40, label=label, c=self.cmap(self.idx/self.linenum), linestyle=linestyle)
        self.ax.plot(range(0, self.dur, int(self.dur/local_dur)*step), rmse_series_mean[:int(local_dur/step)], xunits=40, label=label, c=color, ls=linestyle)
        return rmse_series_mean


    def add_rc(self, exprdir, naturepath="./nature8_every10.csv", npoints=0, local_dur=0, num=0, step=1, label="", color=None, linestyle=None):
        self.idx += 1
        if local_dur == 0:
            local_dur = self.dur
        if npoints == 0:
            npoints = self.npts

        predict, nature = readRC(exprdir, naturepath=naturepath, nsteps=local_dur*num, npoints=npoints, nens=1)

        # cut the recursive data
        rmse_series = np.zeros((local_dur, num))
        for n in range(num):
            predict_cut = predict.getData(start=n*local_dur, end=(n+1)*local_dur)
            nature_cut = nature.getData(start=n*local_dur, end=(n+1)*local_dur)
            rmse_series[:,n] = np.sqrt(((predict_cut - nature_cut) ** 2).mean(axis=1))

        rmse_series_mean = rmse_series.mean(axis=1)
        # self.ax.plot(range(0, self.dur, int(self.dur/local_dur)*step), rmse_series_mean[:int(local_dur/step)], label=label, c=self.cmap(self.idx/self.linenum))
        self.ax.plot(range(0, self.dur, int(self.dur/local_dur)*step), rmse_series_mean[:int(local_dur/step)], label=label, c=color, ls=linestyle)
        return

class RmseMeanDiffFig(RmseMeanFig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.letkf = []
        self.rc = []
    def set_figure(self, **kwargs):
        super().set_figure(**kwargs)
    def show(self):
        plt.show()
    def savefig(self, path):
        plt.savefig(path, facecolor=self.facecolor)

    def add_pair(self, letkf_option, rc_option, label="", color=None, linestyle="solid"):
        letkf, letkf_step = self.add_letkf(**letkf_option)
        rc, rc_step = self.add_rc(**rc_option)
        step = letkf_step
        if letkf_step > rc_step:
            rc = rc[::letkf_step/rc_step]
        elif letkf_step < rc_step:
            step = rc_step
            letkf = letkf[::rc_step/letkf_step]
        diff = letkf - rc
        self.ax.plot(range(0, self.dur, step), diff, label=label, c=color)
        return diff

    def add_letkf(self, predictdir, naturepath, naturesteps=200010, naturestart=100000, local_dur=0, nens=1, num=0, step=10):
        if local_dur == 0:
            local_dur = self.dur
        predict_list, nature = readLETKF(predictdir, naturepath, naturesteps=naturesteps, dur=local_dur, num=num, npoints=self.npts, nens=nens)
        rmse_series = np.zeros((local_dur, num*nens))
        naturestart += 1
        for n in range(num):
            # nature_cut = nature.getData(start=n, end=n+local_dur)
            #ens = np.zeros((local_dur, self.npts))
            #for i in range(nens):
            #    ens = predict_list[n].getData(ens=i)
            #    rmse_series[:,n*nens+i] = np.sqrt(((ens - nature_cut) ** 2).mean(axis=1))
            predict_cut = predict_list[n].getData()
            nature_cut = nature.getData(start=n*local_dur+naturestart, end=(n+1)*local_dur+naturestart)
            rmse_series[:,n] = np.sqrt(((predict_cut - nature_cut) ** 2).mean(axis=1))
        rmse_series_mean = rmse_series.mean(axis=1)[:int(local_dur/step)]
        return rmse_series_mean, int(self.dur/local_dur)*step

    def add_rc(self, exprdir, npoints=0, local_dur=0, num=0, step=1):
        if local_dur == 0:
            local_dur = self.dur
        if npoints == 0:
            npoints = self.npts
        predict, nature = readRC(exprdir, nsteps=local_dur*num, npoints=npoints, nens=1)
        # cut the recursive data
        rmse_series = np.zeros((local_dur, num))
        for n in range(num):
            predict_cut = predict.getData(start=n*local_dur, end=(n+1)*local_dur)
            nature_cut = nature.getData(start=n*local_dur, end=(n+1)*local_dur)
            rmse_series[:,n] = np.sqrt(((predict_cut - nature_cut) ** 2).mean(axis=1))
        rmse_series_mean = rmse_series.mean(axis=1)[:int(local_dur/step)]
        return rmse_series_mean, int(self.dur/local_dur)*step

if __name__ == "__main__":
    npoints = 8
    rmseFig = RmseMeanFig(npoints=npoints, duration=100)

    # RC
    basedir = '../ML/extended/8-5000-100000/'
    rmseFig.add_rc(basedir, local_dur=1000, num=100, label='RC')


    # LETKF
    predictdir = "../DA/asis4-8/11-26/extended"
    naturepath = "../DA/nature8.dat"
    rmseFig.add_letkf(predictdir, naturepath, local_dur=100, nens=20, num=100, label="LETKF")

    rmseFig.savefig("letkfvsrc.png")