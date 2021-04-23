#!usr/bin/env python

#
# Author: Haoran Cheng, adapted from Fangtong Liu's work
#

import numpy as np
from numpy.random import randn, rand
from scipy.stats import multivariate_normal
import sophus as sp
from renderer import *
from skimage.feature import hog
from skimage import data, exposure

class myStruct():
    def __init__(self):
        self.x = []
        self.w = []


# This function is used to wrap angles in radians to the interval [-pi, pi]
# pi maps to pi and -pi maps to -pi
def wrapToPI(phase):
    x_wrap = np.remainder(phase, 2 * np.pi)
    while abs(x_wrap) > np.pi:
        x_wrap -= 2 * np.pi * np.sign(x_wrap)
    return x_wrap

# Particle filter class for state estimation of a nonlinear system
# The implementation follows the Sample Importance Resampling (SIR)
# filter a.k.a bootstrap filter
class particle_filter:

    def __init__(self, system, init, sigma = 0.1):
        # Particle filter construct an instance of this class
        #
        # Input:
        #   system: system and noise models
        #   init:   initialization parameters

        self.f = system.f  # process model
        self.Q = system.Q  # process noise covariance
        self.LQ = np.linalg.cholesky(self.Q)  # Cholesky factor of Q
        self.n = init.n  # number of particles
        self.h = system.h
        self.sigma = sigma

        # initialize particles
        self.p = myStruct()  # particles

        wu = 1 / self.n  # uniform weights
        L_init = np.linalg.cholesky(init.Sigma)


        for i in range(self.n):
            self.p.x.append(init.x * sp.SE3.exp(np.dot(L_init, randn(6, 1))))
            # self.p.x.append(np.dot(L_init, 0.5 * np.ones([len(init.x), 1])) + init.x)
            self.p.w.append(wu)
        self.p.w = np.array(self.p.w).reshape(-1, 1)

        # initialize renderer
        self.rd = renderer()


    def sample_motion(self):
        # A simple random walk motion model
        for i in range(self.n):
            # sample noise
            w = np.dot(self.LQ, randn(self.p.x.shape[1], 1))
            # w = np.dot(self.LQ, np.array([[0.5], [0.01]]))
            # propagate the particle
            self.p.x[i, :] = self.f(self.p.x[i, :].reshape(self.p.x.shape[1],1), w).reshape(-1)

    def sample_motion_cv(self):
        # A constant velocity random walk motion model
        for i in range(self.n):
            # sample noise
            w = np.dot(self.LQ, randn(4, 1))
            # propagate the particle
            self.p.x[i, :] = self.f(self.p.x[i, :], w).reshape(-1)

    def importance_measurement(self, img, left_gripper_joint=0, left_gripper2_joint=0):
        # compare important weight for each particle based on the obtain range and bearing measurements
        #
        # Inputs:
        #   z: measurement
        w = np.zeros([self.n, 1])  # importance weights
        hog_observed = hog(img, orientations=8, pixels_per_cell=(16, 16),
                                cells_per_block=(1, 1), multichannel=True)
        for i in range(self.n):
            # render img at the pose of each particle
            img_rd = self.rd.render_img(left_gripper_joint, left_gripper2_joint, self.p.x[i])
            hog_rd = hog(img_rd, orientations=8, pixels_per_cell=(16, 16),
                                cells_per_block=(1, 1), multichannel=True)
            likelihood = np.exp(-1/self.sigma * np.linalg.norm(hog_observed - hog_rd))
            print("likelihood for particle {} is {}".format(i, likelihood))
            self.p.w[i] = self.p.w[i] * likelihood

        # update and normalize weights
        self.p.w = self.p.w / np.sum(self.p.w)
        # compute effective number of particles
        self.Neff = 1 / np.sum(np.power(self.p.w, 2))  # effective number of particles

    def resampling(self):
        # low variance resampling
        W = np.cumsum(self.p.w)
        r = rand(1) / self.n
        # r = 0.5 / self.n
        j = 1
        for i in range(self.n):
            u = r + (i - 1) / self.n
            while u > W[j]:
                j = j + 1
            self.p.x[i, :] = self.p.x[j, :]
            self.p.w[i] = 1 / self.n






