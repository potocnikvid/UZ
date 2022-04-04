import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
from a6_utils import drawEllipse


# a

def draw_vectors(mean, u, s):
    e1 = np.sqrt(s[0]) * u[:, 0]
    e2 = np.sqrt(s[1]) * u[:, 1]
    plt.plot([mean[0], mean[0] + e1[0]], [mean[1], mean[1] + e1[1]], 'r')
    plt.plot([mean[0], mean[0] + e2[0]], [mean[1], mean[1] + e2[1]], 'g')

def direct_pca(x):
    x = x.T
    N = x.shape[1]
    q = np.mean(x, axis=1)
    x_d = np.empty((x.shape))
    for i in range(len(x[0])):
        x_d[0, i] = x[0, i] - q[0]
        x_d[1, i] = x[1, i] - q[1]
    c = (x_d @ x_d.T) / (N - 1)
    u, s, _ = np.linalg.svd(c)
    # to PCA
    y = u.T @ (x.T - q).T
    # from PCA
    x_q = (u @ y).T + q
    return q, c, u, s, y, x_q




x = np.array([[1.0, 0.0], [6.0, 2.0], [5.0, 4.0], [1.0, 3.0], [0.0, 1.0]])
x = x.T

N = x.shape[1]
q = np.mean(x, axis=1)
x_d = np.empty((x.shape))
for i in range(len(x[0])):
    x_d[0, i] = x[0, i] - q[0]
    x_d[1, i] = x[1, i] - q[1]
c = (x_d @ x_d.T) / (N - 1)
u, s, _ = np.linalg.svd(c)


# b

plt.ylim(-4, 7)
plt.xlim(-4, 7)
plt.scatter(x.T[:, 0], x.T[:, 1], color='blue')
drawEllipse(q, c)
draw_vectors(q, u, s)
plt.show()

# c

# make cumulative eigenvalues vector
s_cpy = np.copy(s)
s_cpy[1] = s_cpy[0] + s_cpy[1]
# normalize by largest value
s_norm = s_cpy / (s_cpy[-1])
plt.plot([0, 1, 2], [0, s_norm[0], s_norm[1]])
plt.title(str(round(s_norm[0] * 100)) + ' %')
plt.show()

# d

# to PCA
y = u.T @ (x.T - q).T

u_cpy = np.copy(u)
u[:, -1] = 0
# from PCA
x_q = (u @ y).T + q

mean_y = np.mean(y.T, axis=0)
cov_y = np.cov(y)
u_y, s_y, _ = np.linalg.svd(cov_y)

plt.ylim(-4, 7)
plt.xlim(-4, 7)

plt.scatter(x.T[:, 0], x.T[:, 1], color='blue')
drawEllipse(q, c)
draw_vectors(q, u_cpy, s)

plt.scatter(y.T[:, 0], y.T[:, 1], color='red')
drawEllipse(mean_y, cov_y)
draw_vectors(mean_y, u_y, s_y)

plt.scatter(x_q[:, 0], x_q[:, 1], color='green')
for i in range(len(x_q)):
    plt.annotate(i, (x_q[i, 0], x_q[i, 1]))
plt.show()


# Question: What happens to the reconstructed points? Where is the data projected
# to?
# The data is projected on a line that splits the elipse in half


def euclidean(p1, p2):
    return np.sqrt(np.square(p1[0] - p2[0]) + np.square(p1[1] - p2[1]))

def distances(x, point):
    a = []
    for p in x:
        dist = euclidean(p, point)
        a.append(dist)
    return a

# e

def e_1():
    x = np.array([[1.0, 0.0], [6.0, 2.0], [5.0, 4.0], [1.0, 3.0], [0.0, 1.0]])
    x = x.T

    q_point = np.array([3, 6])
    idx = np.argmin(distances(x.T, q_point))
    min_point = x.T[idx]
    print(idx, min_point)

    # x = x.T
    N = x.shape[1]
    q = np.mean(x, axis=1)
    x_d = np.empty((x.shape))
    for i in range(len(x[0])):
        x_d[0, i] = x[0, i] - q[0]
        x_d[1, i] = x[1, i] - q[1]
    c = (x_d @ x_d.T) / (N - 1)
    u, s, _ = np.linalg.svd(c)


    # to PCA
    y = u.T @ (x.T - q).T

    u[:, -1] = 0

    # from PCA
    x_q = (u @ y).T + q

    q_point_y = u.T @ (q_point - q)
    q_point_q = (u @ q_point_y.T).T + q

    idx = np.argmin(distances(x_q[:-1, :], q_point_q))
    min_point = x_q[idx]
    print(idx, min_point)


    fig = plt.figure(figsize=(30, 10))

    a = fig.add_subplot(131)
    plt.ylim(-4, 7)
    plt.xlim(-4, 7)
    plt.scatter(x.T[:, 0], x.T[:, 1], color='blue')
    for i in range(len(x_q)):
        plt.annotate(i, (x.T[i, 0], x.T[i, 1]))
    plt.scatter(q_point[0], q_point[1], color='orange')
    plt.annotate(5, (q_point[0], q_point[1]))


    b = fig.add_subplot(132)
    plt.ylim(-4, 7)
    plt.xlim(-4, 7)
    plt.scatter(y.T[:, 0], y.T[:, 1], color='red')
    for i in range(len(x_q)):
        plt.annotate(i, (y.T[i, 0], y.T[i, 1]))
    plt.scatter(q_point_y.T[0], q_point_y.T[1], color='orange')
    plt.annotate(5, (q_point_y[0], q_point_y[1]))

    c = fig.add_subplot(133)
    plt.ylim(0, 7)
    plt.xlim(0, 7)
    plt.scatter(x_q[:, 0], x_q[:, 1], color='green')
    plt.scatter(q_point_q[0], q_point_q[1], color='orange')

    for i in range(len(x_q)):
        plt.annotate(i, (x_q[i, 0], x_q[i, 1]))

    plt.annotate(5, (q_point_q[0], q_point_q[1]))

    plt.show()

def e_2():
    x = np.array([[1.0, 0.0], [6.0, 2.0], [5.0, 4.0], [1.0, 3.0], [0.0, 1.0]])

    q_point = np.array([3, 6])
    idx = np.argmin(distances(x, q_point))
    min_point = x[idx]
    print(idx, min_point)

    x = np.vstack((x, q_point))

    x = x.T
    N = x.shape[1]
    q = np.mean(x, axis=1)
    x_d = np.empty((x.shape))
    for i in range(len(x[0])):
        x_d[0, i] = x[0, i] - q[0]
        x_d[1, i] = x[1, i] - q[1]
    c = (x_d @ x_d.T) / (N - 1)
    u, s, _ = np.linalg.svd(c)


    # to PCA
    y = u.T @ (x.T - q).T

    u[:, -1] = 0

    # from PCA
    x_q = (u @ y).T + q



    idx = np.argmin(distances(x_q[:-1, :], x_q[-1, :]))
    min_point = x_q[idx]
    print(idx, min_point)


    fig = plt.figure(figsize=(30, 10))

    a = fig.add_subplot(131)
    plt.ylim(-4, 7)
    plt.xlim(-4, 7)
    plt.scatter(x.T[:, 0], x.T[:, 1], color='blue')
    for i in range(len(x_q)):
        plt.annotate(i, (x.T[i, 0], x.T[i, 1]))

    b = fig.add_subplot(132)
    plt.ylim(-4, 7)
    plt.xlim(-4, 7)
    plt.scatter(y.T[:, 0], y.T[:, 1], color='red')
    for i in range(len(x_q)):
        plt.annotate(i, (y.T[i, 0], y.T[i, 1]))

    c = fig.add_subplot(133)
    plt.ylim(0, 7)
    plt.xlim(0, 7)
    plt.scatter(x_q[:-1, 0], x_q[:-1, 1], color='green')
    plt.scatter(x_q[-1, 0], x_q[-1, 1], color='red')

    for i in range(len(x_q)):
        plt.annotate(i, (x_q[i, 0], x_q[i, 1]))

    plt.annotate(5, (x_q[-1, 0], x_q[-1, 1]))

    plt.show()

e_1()
e_2()