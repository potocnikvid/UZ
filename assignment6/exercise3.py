import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import os


def read_data(filename):
    data = []
    with open(filename) as f:
        s = f.read()
        a = np.fromstring(s, sep=" ")
        for i in range(int(len(a)/4)):
            line = []
            for j in range(4):
                line.append(a[4 * i + j])
            data.append(line)
    return np.array(data)


def data_prep(path):
    data = np.empty((0, 84 * 96))
    images = os.listdir(path)
    for name in images:
        img = cv2.imread(path + str(name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.reshape(img, -1)
        data = np.vstack((data, img))
    return data.T


def pca(x):
    m = x.shape[0]
    N = x.shape[1]
    q = np.mean(x, axis=1)
    x_d = np.empty((m, N))
    for i in range(N):
        x_d[:, i] = x[:, i] - q
    c = (x_d.T @ x_d) / (m - 1)
    u, s, _ = np.linalg.svd(c)
    s += np.power(10.0, -15)
    u = (x_d @ u) * np.sqrt(1 / (s * (m - 1)))
    return q, u





# b

def b():
    img_shape = (96, 84)
    data_1 = data_prep("./data/faces/1/")
    q, u = pca(data_1)

    # transform eigenvectors to matrix

    eigvec_1 = np.reshape(u[:, 0], img_shape)
    eigvec_2 = np.reshape(u[:, 1], img_shape)
    eigvec_3 = np.reshape(u[:, 2], img_shape)
    eigvec_4 = np.reshape(u[:, 3], img_shape)
    eigvec_5 = np.reshape(u[:, 4], img_shape)

    plt.subplot(2, 3, 1)
    plt.imshow(eigvec_1, cmap="gray")
    plt.subplot(2, 3, 2)
    plt.imshow(eigvec_2, cmap="gray")
    plt.subplot(2, 3, 3)
    plt.imshow(eigvec_3, cmap="gray")
    plt.subplot(2, 3, 4)
    plt.imshow(eigvec_4, cmap="gray")
    plt.subplot(2, 3, 5)
    plt.imshow(eigvec_5, cmap="gray")
    plt.show()


    img_1_vec = data_1[:, 0]
    img_1 = np.reshape(img_1_vec, img_shape)
    img_1_vec_pixel = np.copy(img_1_vec)
    img_1_vec_pixel[4074] = 0
    img_1_pixel = np.reshape(img_1_vec_pixel, img_shape)

    img_1_y = u.T @ (img_1_vec_pixel.T - q).T
    img_1_q = (u @ img_1_y).T + q

    img_1_q = np.reshape(img_1_q, img_shape)



    img_2_vec = data_1[:, 0]
    img_2 = np.reshape(img_2_vec, img_shape)

    img_2_y = u.T @ (img_2_vec.T - q).T
    img_2_y[3] = 0
    img_2_q = (u @ img_2_y).T + q

    img_2_q = np.reshape(img_2_q, img_shape)

    plt.subplot(2, 4, 1)
    plt.imshow(img_1, cmap='gray')
    plt.title("Pre PCA")
    plt.subplot(2, 4, 2)
    plt.imshow(img_1_pixel, cmap='gray')
    plt.title("Pre PCA pixel")
    plt.subplot(2, 4, 3)
    plt.imshow(img_1_q, cmap='gray')
    plt.title("Post PCA")
    plt.subplot(2, 4, 4)
    plt.imshow(img_1_q - img_1, cmap='gray')
    plt.title("Diff")

    plt.subplot(2, 4, 5)
    plt.imshow(img_2, cmap='gray')
    plt.title("Pre PCA")
    plt.subplot(2, 4, 7)
    plt.imshow(img_2_q, cmap='gray')
    plt.title("Post PCA")
    plt.subplot(2, 4, 8)
    plt.imshow(img_2 - img_2_q, cmap='gray')
    plt.title("Diff")
    plt.show()

# c

def c():
    img_shape = (96, 84)
    data_2 = data_prep("./data/faces/2/")
    q, u = pca(data_2)

    img_vec = data_2[:, 0]
    img = np.reshape(img_vec, img_shape)

    images_y = np.empty((0, 64))
    images_q = np.empty((0, 84*96))
    for i in (32, 16, 8, 4, 2, 1):
        img_y = u.T @ (img_vec.T - q).T
        img_y[i:] = 0
        img_q = (u @ img_y).T + q
        images_y = np.vstack((images_y, img_y))
        images_q = np.vstack((images_q, img_q))


    plt.subplot(3, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original image")
    plt.subplot(3, 3, 4)
    plt.imshow(np.reshape(images_q[0], img_shape), cmap='gray')
    plt.title("32:")
    plt.subplot(3, 3, 5)
    plt.imshow(np.reshape(images_q[1], img_shape), cmap='gray')
    plt.title("16:")
    plt.subplot(3, 3, 6)
    plt.imshow(np.reshape(images_q[2], img_shape), cmap='gray')
    plt.title("8:")
    plt.subplot(3, 3, 7)
    plt.imshow(np.reshape(images_q[3], img_shape), cmap='gray')
    plt.xlabel("4:")
    plt.subplot(3, 3, 8)
    plt.imshow(np.reshape(images_q[4], img_shape), cmap='gray')
    plt.xlabel("2:")
    plt.subplot(3, 3, 9)
    plt.imshow(np.reshape(images_q[5], img_shape), cmap='gray')
    plt.xlabel("1:")
    plt.show()



def d():
    img_shape = (96, 84)
    data_1 = data_prep("./data/faces/2/")
    q, u = pca(data_1)

    q_y = u.T @ (q.T - q).T


    linspace1 = np.sin(np.linspace(-10, 10, 50))
    linspace2 = np.cos(np.linspace(-10, 10, 50))

    for i in range(len(linspace1)):
        q_y[1] = 0
        q_y[2] = 0
        q_y[1] = linspace1[i]
        q_y[2] = linspace2[i]
        q_q = (u @ q_y).T + q

        plt.imshow(np.reshape(q_q, img_shape) - np.reshape(q, img_shape), cmap='gray')
        plt.draw()
        plt.pause(0.1)

    plt.show()





# b()
# c()
d()
