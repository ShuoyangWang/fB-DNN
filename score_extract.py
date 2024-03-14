import os
import pdb

import numpy as np

from datagen1d2d import datagen_1d2d, fourier

work_dir = "C:/Users/sukru/lassonet/rtopython"
m1 = m2 = 30
m = 15
p1 = 30
p2 = 100
n0 = n1 = n2 = 100

def cbind(x, y):
    if x.size == 0:
        return np.row_stack(y)
    return np.column_stack((x, y))


for sim in range(1, 51):
    np.random.seed(sim)

    Data = datagen_1d2d(p1, p2, n0, n1, n2, m, m1, m2)
    Data1d = Data[0]
    Data2d = Data[1]

    #1d 
    J = 50
    S = np.linspace(0, 1, num=m)
    phi = np.array([])
    for j in range(1, J + 1):
        phi = cbind(phi, fourier(S, m, j))

    Dat_score_1d = []
    for j in range(p1):
        def func(x):
            return np.dot(x / m, phi)
        score = []
        for i in Data1d[j]:
            score.append(func(i))
        Dat_score_1d.append(score)

    # J = 50
    X0 = X1 = X2 = np.array([])
    J = 50

    for j in range(p1):
        X0 = cbind(X0, Dat_score_1d[j][0][:, :J])
        X1 = cbind(X1, Dat_score_1d[j][1][:, :J])
        X2 = cbind(X2, Dat_score_1d[j][2][:, :J])

    # 2d
    J1 = J2 = 10
    phi1 = np.array([])
    phi2 = np.array([])
    S1 = np.linspace(0, 1, num=m1)
    S2 = np.linspace(0, 1, num=m2)

    for j in range(1, J1 + 1):
        phi1 = cbind(phi1, fourier(S1, m1, j).T)

    for j in range(1, J2 + 1):
        phi2 = cbind(phi2, fourier(S2, m2, j).T)

    phi = np.kron(phi2.T, phi1.T).T
    M = m1 * m2
    Dat_score_2d = []

    for j in range(p2):
        def func(x):
            return np.dot(x / M, phi)
        score = []
        for i in Data2d[j]:
            score.append(func(i))
        Dat_score_2d.append(score)


    ## J = 50
    J = 50
    for j in range(p2):
        X0 = cbind(X0, Dat_score_2d[j][0][:, :J])
        X1 = cbind(X1, Dat_score_2d[j][1][:, :J])
        X2 = cbind(X2, Dat_score_2d[j][2][:, :J])

    X = np.vstack((X0, X1, X2))
    np.savetxt(os.path.join(work_dir, f"J=20/X{sim}.csv"), X, delimiter=",", fmt='%f')
    
    y = np.concatenate((np.repeat(0, n0), np.repeat(1, n1), np.repeat(2, n2)))
    y = y.astype(int)
    np.savetxt(os.path.join(work_dir, f"J=20/y{sim}.csv"), y, fmt='%d', delimiter=",")  


    




