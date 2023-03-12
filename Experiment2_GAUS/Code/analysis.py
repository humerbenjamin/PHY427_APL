import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import mpl_toolkits.mplot3d
from scipy.signal import savgol_filter
from scipy import stats

def Gauss(x, A, B, C):
    y = A*np.exp(-1*B*(x-C)**2)
    return y

def getZ(a, b, z):
    return z[int(b)][int(a)]

def smoothing(z, dx=0, dy=0):
    for a in range(len(z)-4):
        i = a + 2
        for b in range(len(z[i])-4):
            j = b + 2
            z[i][j] = 0.5*z[i][j] + 0.5*((z[i-1][j]+z[i+1][j]+z[i][j-1]+z[i][j+1]+z[i-2][j]+z[i+2][j]+z[i][j-2]+z[i][j+2])/4)
    return z

def sf_smoothing(z, n=1):
    for i in range(n):
        z = savgol_filter(z, 21, 2)
    return z

def smooth_all(params, n=1):
    for i in range(len(params)):
        params[i][2] = sf_smoothing(params[i][2], n)
    return params


def spot_size(z):
    c = 0
    max = np.max(z)
    hh = max/2
    for i in range(len(z)):
        for j in range(len(z[0])):
            if(z[i][j] > hh):
                c += 1
    ratio =  c / (480*640)
    A_tot = 5.325 * 6.656
    A_circ = A_tot *ratio
    return np.sqrt((A_circ) / (np.pi))

def readfiles(files):
    params = []
    for file in files:
        Z = pd.read_csv(file, sep='\t', header=None)
        z = np.array(Z)
        x = np.linspace(0, 5.325, 480)
        y = np.linspace(0, 6.656, 640)
        x, y = np.meshgrid(x, y)
        X = np.linspace(0, 479, 480)
        Y = np.linspace(0, 639, 640)
        Z = []
        for i in range(len(Y)):
            Z.append([])
            for j in range(len(X)):
                Z[i].append(getZ(Y[i], X[j], z))
        Z = np.array(Z)
        params.append([x, y, Z])
    return params

def plot_means(params, dist):
    y = []
    for i in range(len(params)):
        y.append(np.mean(params[i][2]))
    plot_vals(dist, y, "Mean Intensity as a function of distance", "means")

def plot_maxes(params, dist):
    y = []
    for i in range(len(params)):
        y.append(np.max(params[i][2]))
    for i in range(len(y)):
        y[i] = y[i]*0.8
    plot_vals(dist, y, "Max Intensity as a function of distance", "maxes")

def plot_spots(params, distances):
    y = []
    for i in range(len(params)):
        y.append(spot_size(params[i][2]))
    plot_vals(distances, y, "Spot Size as a function of distance", "spot_size")

def geo_spot(x, A, B):
    return B*(np.abs(x-A))

def plot_vals(x, y, title, type):
    plt.plot(x[1:], y[1:], "o")
    plt.title(title)
    plt.xlabel("Distance (mm)")
    if type == "spot_size":
        plt.ylabel("Equivalent Radius of the Spot (mm)")
        x_data = np.arange(46, 50, 0.001)
        parameters, covariance = sp.optimize.curve_fit(geo_spot, x[1:], y[1:], p0=[48, 0.5])
        print(parameters)
        fit_A = parameters[0]
        fit_B = parameters[1]
        fit_y = geo_spot(x_data, fit_A, fit_B)
        plt.plot(x_data, fit_y, "r")
    if type == "maxes":
        plt.ylabel("Peak Intensity")
        x_data = np.arange(46, 50, 0.001)
        parameters, covariance = sp.optimize.curve_fit(Gauss, x[1:], y[1:], p0=[255, 5, 47])
        print(parameters)
        fit_A = parameters[0]
        fit_B = parameters[1]
        fit_C = parameters[2]
        fit_y = Gauss(x_data, fit_A, fit_B, fit_C)
        plt.plot(x_data, fit_y, "r")
    plt.show()

def plot_sur(x, y, z, name=""):
    ax = plt.axes(projection="3d")
    ax.plot_surface(x, y, z, cmap='magma', edgecolor='none', rstride=10, cstride=10)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(name)
    plt.show()

if __name__ == '__main__':
    filenames = ['Experiment2_GAUS/Data/2023-02-14_60mm-45-5.txt', 'Experiment2_GAUS/Data/2023-02-14_60mm-46.txt', 'Experiment2_GAUS/Data/2023-02-14_60mm-46-5.txt', 'Experiment2_GAUS/Data/2023-02-14_60mm-47.txt', 'Experiment2_GAUS/Data/2023-02-14_60mm-47-5.txt', 'Experiment2_GAUS/Data/2023-02-14_60mm-48.txt', 'Experiment2_GAUS/Data/2023-02-14_60mm-48-5.txt', 'Experiment2_GAUS/Data/2023-02-14_60mm-49.txt', 'Experiment2_GAUS/Data/2023-02-14_60mm-49-5.txt', 'Experiment2_GAUS/Data/2023-02-14_60mm-50.txt']
    distances = [45.5, 46, 46.5, 47, 47.5, 48, 48.5, 49, 49.5, 50]
    params = readfiles(filenames)
    # plot_sur(params[4][0], params[4][1], params[4][2], 'Intensity Distribution at 45.5mm measurement')
    params = smooth_all(params, 200)
    # plot_sur(params[4][0], params[4][1], params[4][2], 'Intensity Distribution at 47mm measurement, smoothed')
    plot_spots(params, distances)
    # print(np.max(params[5][2]))

    