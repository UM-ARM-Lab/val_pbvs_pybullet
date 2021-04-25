#!usr/bin/env python

#
# Author: Haoran Cheng
# Date: 02/06/2020
#

import numpy as np
from numpy.random import randn
from particle_filter import particle_filter
import matplotlib.pyplot as plt
import math
from numpy import genfromtxt
from ekf_batch import measurement_model, h2

Kf1 = genfromtxt('data_csv/Kf_1.csv', delimiter=',')
Kf2 = genfromtxt('data_csv/Kf_2.csv', delimiter=',')
c1 = genfromtxt('data_csv/C_1.csv', delimiter=',').reshape(2,1)
c2 = genfromtxt('data_csv/C_2.csv', delimiter=',').reshape(2,1)
Rot = genfromtxt('data_csv/R.csv', delimiter=',')
trans = genfromtxt('data_csv/t.csv', delimiter=',').reshape(3,1)

class myStruct:
    pass

# process model
def process_model(x, w):
    f = x + w
    return f

# measurement model

# Toy example for tracking a single target using a particle filter and range-bearing measurements

if __name__ == '__main__':
    # process noise covariance
    Q = np.diag(np.power([0.01, 0.01, 0.01], 2))
    # measurement noise covariance
    R = np.diag(np.power([5, 5, 5, 5], 2))
    #initial state covariance 
    Sigma_init = 0.1 * np.eye(3)
    # Cholesky factor of covariance for sampling
    L = np.linalg.cholesky(R)
    z1 = genfromtxt('data_csv/z_1.csv', delimiter=',').T # 2X20
    z2 = genfromtxt('data_csv/z_2.csv', delimiter=',').T # 2X20
    z = np.vstack((z1,z2))  

    # build the system
    sys = myStruct()
    sys.f = process_model
    sys.h = measurement_model
    sys.Q = Q
    sys.R = R

    # initialize the state using the first measurement and triangulation
    # m = [px/pz, py/pz]
    m = np.linalg.inv(Kf1).dot(z[:2,0:1] - c1)
    # z proposal, try some z
    zp = np.arange(1.93,1.97,0.001)
    err = np.zeros(len(zp))
    # for every z, calculate p1, then calculate h2, minimize h2-h2(observed)
    for i, zi in enumerate(zp):
        xi = m[0,0] * zi
        yi = m[1,0] * zi
        pi = np.array([[xi, yi, zi]]).T
        err[i] = np.linalg.norm(h2(pi) - z[2:4, 0:1])
    z_init =  zp[np.argmin(err)]
    x_init = np.array([[m[0, 0]*z_init, m[1,0]*z_init, z_init]]).T

    # initialization
    init = myStruct()
    init.n = 100
    init.x = x_init
    #init.x = np.array([[1,1,1]]).T
    init.Sigma = Sigma_init

    filter = particle_filter(sys, init)
    
    x = np.empty([3, np.shape(z)[1]])  # state
    x[:, 0] = init.x.reshape(-1)
    
    # main loop; iterate over the measurements
    for i in range(1, np.shape(z)[1], 1):
        filter.sample_motion()
        filter.importance_measurement(z[:, i].reshape([4, 1]), measurement_model)
        if filter.Neff < filter.n / 5:
            filter.resampling()
        wtot = np.sum(filter.p.w)
        if wtot > 0:
            a = filter.p.x
            b = filter.p.w
            x[0, i] = np.sum(filter.p.x[:, 0] * filter.p.w.reshape(-1)) / wtot
            x[1, i] = np.sum(filter.p.x[:, 1] * filter.p.w.reshape(-1)) / wtot
            x[2, i] = np.sum(filter.p.x[:, 2] * filter.p.w.reshape(-1)) / wtot
        else:
            print('\033[91mWarning: Total weight is zero or nan!\033[0m')
            x[:, i] = [np.nan, np.nan, np.nan]
        plt.clf()
        
        hp, = plt.plot(filter.p.x[:, 0], filter.p.x[:, 1], '.', color='b', alpha=0.5, markersize=3)
        plt.grid(True)
        plt.axis('equal')
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.pause(0.05)

    print("Final estimated position: x: {}, y: {}, z: {}".format(x[0,-1], x[1,-1],x[2,-1]))
    # plotting
    fsize = 14
    fig = plt.figure()
    line1, = plt.plot(range(np.shape(z)[1]), x[0, : ], color='r', linewidth=2)
    line2, = plt.plot(range(np.shape(z)[1]), x[1, : ], color='g', linewidth=2)
    line3, = plt.plot(range(np.shape(z)[1]), x[2, : ], color='b', linewidth=2)
    plt.legend([line1, line2, line3], [r'x', r'y', r'z'], loc='best',fontsize=fsize)
    plt.xlabel(r'$Step$', fontsize=fsize)
    plt.ylabel(r'$Position$', fontsize=fsize)
    plt.xticks(fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.title("Plot of Object Position vs Step \n (Particle Filter with Batch measurement update)",fontsize=fsize)
    plt.grid(True)
    plt.show()
 



