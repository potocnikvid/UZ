import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
from a6_utils import drawEllipse


def draw_vectors(mean, u, s):
    e1 = np.sqrt(s[0]) * u[:, 0]
    e2 = np.sqrt(s[1]) * u[:, 1]
    plt.plot([mean[0], mean[0] + e1[0]], [mean[1], mean[1] + e1[1]], 'r')
    plt.plot([mean[0], mean[0] + e2[0]], [mean[1], mean[1] + e2[1]], 'g')


# a

x = np.array([[1.0, 0.0], [6.0, 2.0], [5.0, 4.0], [1.0, 3.0], [0.0, 1.0]])


# DUAL PCA
x = x.T
N = x.shape[1]
m = x.shape[0]
q = np.mean(x, axis=1)
x_d = np.empty((x.shape))
for i in range(len(x[0])):
    x_d[0, i] = x[0, i] - q[0]
    x_d[1, i] = x[1, i] - q[1]
c = (x_d.T @ x_d) / (m - 1)
u, s, _ = np.linalg.svd(c)
u = (x_d @ u) * np.sqrt(1 / (s * (m - 1)))


# to PCA
y = u.T @ (x.T - q).T
# u[:, -1] = 0
# from PCA
x_q = (u @ y).T + q


fig = plt.figure(figsize=(20, 10))

a = fig.add_subplot(121)
plt.ylim(-4, 7)
plt.xlim(-4, 7)
plt.scatter(x.T[:, 0], x.T[:, 1], color='blue')
for i in range(len(x.T)):
    plt.annotate(i, (x.T[i, 0], x.T[i, 1]))

mean_y = np.mean(y.T, axis=0)
cov_y = np.cov(y)
u_y, s_y, _ = np.linalg.svd(cov_y)
plt.scatter(y.T[:, 0], y.T[:, 1], color='red')
drawEllipse(mean_y, cov_y)
draw_vectors(mean_y, u_y, s_y)

b = fig.add_subplot(122)
plt.ylim(-4, 7)
plt.xlim(-4, 7)
plt.scatter(x_q[:, 0], x_q[:, 1], color='green')
for i in range(len(x_q)):
    plt.annotate(i, (x_q[i, 0], x_q[i, 1]))

plt.show()