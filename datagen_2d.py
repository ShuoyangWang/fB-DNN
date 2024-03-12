import pdb
from math import cos, pi, sin, sqrt

import numpy as np


def fourier(s, M, j):
    k = j // 2
    if j == 1:
        return np.repeat(1, M)
    elif j % 2 == 0:
        return sqrt(2 / M) * np.cos(2 * pi * k * s)
    elif j % 2 != 0:
        return sqrt(2 / M) * np.sin(2 * pi * k * s)
    assert False, "SHOULD NEVER REACH HERE"

def random_normal(mu2, eigen2, n):
    norm1 = np.random.normal(loc=mu2[0], scale=eigen2[0], size=n)
    norm2 = np.random.normal(loc=mu2[1], scale=eigen2[1], size=n)
    norm3 = np.random.normal(loc=mu2[2], scale=eigen2[2], size=n)
    norm4 = np.random.normal(loc=mu2[3], scale=eigen2[3], size=n)
    norm5 = np.random.normal(loc=mu2[4], scale=eigen2[4], size=n)
    xi2 = np.column_stack((norm1, norm2, norm3, norm4, norm5))
    return xi2

def datagen_2d(p, n0, n1, n2, m1, m2):
    # generate data
    Data = []

    # generate grid points
    S1 = np.linspace(0, 1, m1)
    S2 = np.linspace(0, 1, m2)

    # rate for exponential distribution
    r0 = np.array([0.1, 0.12, 0.14, 0.16, 0.18])

    # df and non-central parameters for t distribution
    df1 = np.array([3, 5, 7, 9, 11])
    ncp1 = np.array([3, 3, 3, 3, 3])

    # sd and mean for normal distribution
    eigen2 = np.array([1, 0.8, 0.6, 0.4, 0.2])
    mu2 = np.array([0, 0, 0, 0, 0])

    for _ in range(1, 6):
        data = []
        #generate projection scores

        #exponential 
        exp1 = np.random.exponential(scale=1/r0[0], size=n0)
        exp2 = np.random.exponential(scale=1/r0[1], size=n0)
        exp3 = np.random.exponential(scale=1/r0[2], size=n0)
        exp4 = np.random.exponential(scale=1/r0[3], size=n0)
        exp5 = np.random.exponential(scale=1/r0[4], size=n0)
        mtx = np.random.normal(loc=0, scale=0.1, size=(n0, 5))
        xi0 = np.column_stack((exp1, exp2, exp3, exp4, exp5)) + mtx

        #student's t 
        t1 = np.random.standard_t(df=df1[0], size=n1) + ncp1[0]
        t2 = np.random.standard_t(df=df1[1], size=n1) + ncp1[1]
        t3 = np.random.standard_t(df=df1[2], size=n1) + ncp1[2]
        t4 = np.random.standard_t(df=df1[3], size=n1) + ncp1[3]
        t5 = np.random.standard_t(df=df1[4], size=n1) + ncp1[4]
        xi1 = np.column_stack((t1, t2, t3, t4, t5)) + mtx

        #normal
        xi2 = random_normal(mu2, eigen2, n2) + mtx

        #generate basis functions
        SS1 = np.repeat(S1, m1); SS2 = np.repeat(S2, m2)
        BB1 = SS1; BB2 = SS2; BB3 = SS1 * SS2; BB4 = (SS1) ** 2; BB5 = (SS2) ** 2
        BB = np.vstack((BB1, BB2, BB3, BB4, BB5))

        #generate discretely observed curves
        data.append(np.dot(xi0, BB))
        data.append(np.dot(xi1, BB))
        data.append(np.dot(xi2, BB))
        Data.append(data)

    for _ in range(6, p + 1):
        data = []

        #generate projection scores
        mtx = np.random.normal(loc=0, scale=0.1, size=(n0, 5))

        xi0 = random_normal(mu2, eigen2, n0) + mtx
        xi1 = random_normal(mu2, eigen2, n1) + mtx
        xi2 = random_normal(mu2, eigen2, n2) + mtx

        #generate basis functions
        SS1 = np.repeat(S1, m1); SS2 = np.repeat(S2, m2)
        BB1 = SS1; BB2 = SS2; BB3 = SS1 * SS2; BB4 = (SS1) ** 2; BB5 = (SS2) ** 2
        BB = np.vstack((BB1, BB2, BB3, BB4, BB5))

        #generate discretely observed curves
        data.append(np.dot(xi0, BB))
        data.append(np.dot(xi1, BB))
        data.append(np.dot(xi2, BB))
        Data.append(data)

    return [Data]
        




        








