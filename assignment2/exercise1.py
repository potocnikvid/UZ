import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

# a --------------------------------------------------------------------------------------------------------------------

def myhist3(img, bins):
    hist = np.zeros((bins, bins, bins))
    img_cpy = np.copy(img)
    img_arr = np.reshape(img_cpy, (-1, 3))
    img_arr //= int(np.ceil(256 / bins))
    np.add.at(hist, (img_arr[:, 0], img_arr[:, 1], img_arr[:, 2]), 1)
    return (hist/(np.size(img_cpy)/3)).reshape(-1)


img = cv2.imread('dataset/object_01_1.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# b --------------------------------------------------------------------------------------------------------------------

def compare_histograms(hist1, hist2, n, measure):
    if measure == "euclidean":
        iter = (np.float_power(hist1[i] - hist2[i], 2) for i in np.arange(0, n))
        return np.power(np.sum(np.fromiter(iter, float)), 0.5)
    elif measure == "chi":
        iter = (np.power(hist1[i] - hist2[i], 2)/(hist1[i] + hist2[i] + 1e-10) for i in np.arange(0, n))
        return 0.5 * np.sum(np.fromiter(iter, float))
    elif measure == "intersection":
        iter = (np.minimum(hist1[i], hist2[i]) for i in np.arange(0, n))
        return 1.0 - np.sum(np.fromiter(iter, float))
    elif measure == "hellinger":
        iter = (np.power(np.float_power(hist1[i], 0.5) - np.float_power(hist2[i], 0.5), 2) for i in np.arange(0, n))
        return np.power(0.5 * np.sum(np.fromiter(iter, float)), 2)
    else:
        print("wrong distance name")


# c --------------------------------------------------------------------------------------------------------------------

obj_01_1 = cv2.imread('dataset/object_01_1.png')
obj_01_1 = cv2.cvtColor(obj_01_1, cv2.COLOR_BGR2RGB)
obj_02_1 = cv2.imread('dataset/object_02_1.png')
obj_02_1 = cv2.cvtColor(obj_02_1, cv2.COLOR_BGR2RGB)
obj_03_1 = cv2.imread('dataset/object_03_1.png')
obj_03_1 = cv2.cvtColor(obj_03_1, cv2.COLOR_BGR2RGB)


bins = 8
rnge = np.power(bins, 3)
h1 = myhist3(obj_01_1, bins)
plt.subplot(2, 3, 1)
plt.imshow(obj_01_1)
plt.subplot(2, 3, 2)
plt.imshow(obj_02_1)
plt.subplot(2, 3, 3)
plt.imshow(obj_03_1)

measure = "euclidean"
plt.subplot(2, 3, 4)
h1 = myhist3(obj_01_1, bins)
l1 = round(compare_histograms(h1, h1, rnge, measure), 3)
plt.bar(np.arange(0, rnge, 1), h1, width=4)
plt.title('L(h1, h1) = ' + str(l1))

plt.subplot(2, 3, 5)
h2 = myhist3(obj_02_1, bins)
l2 = round(compare_histograms(h1, h2, rnge, measure), 3)
plt.bar(np.arange(0, rnge, 1), h2, width=4)
plt.title('L(h1, h2) = ' + str(l2))

plt.subplot(2, 3, 6)
h3 = myhist3(obj_03_1, bins)
l3 = round(compare_histograms(h1, h3, rnge, measure), 3)
plt.bar(np.arange(0, rnge, 1), h3, width=4)
plt.title('L(h1, h3) = ' + str(l3))

plt.suptitle(str(measure))
plt.show()

# Question: Which image (object_02_1.png or object_03_1.png) is more similar
# to image object_01_1.png considering the L 2 distance? How about the other three
# distances? We can see that all three histograms contain a strongly expressed com-
# ponent (one bin has a much higher value than the others). Which color does this
# bin represent?
# object_03_1.png is more similar to the first one, because the euclidean distance between them is smaller
# Chi-square distance between h1 and h2 is quite similar to the euclidean while the one between h1 and h3 is 30% higher.
# Intersection distance is higher than euclidean by 50% between h1 and h2 and 200% between h1 and h3.
# Hellinger distance is a lot smaller compared to the others. Betweeen h1 and h2 it is about 30% of euclidean and 10% bettween h1 and h2.
# It represents black pixels at bin 0

# d --------------------------------------------------------------------------------------------------------------------


def make_histograms(path, n_bins):
    histograms = np.empty((0, (n_bins ** 3)), float, order='C')
    # histograms = np.empty((0, (n_bins ** 3)), dtype={'name': np.array()}, order='C')
    names = np.array([os.listdir(path)]).reshape(-1)
    for i in os.listdir(path):
        img = cv2.imread(path + '/' + i)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h = myhist3(img, n_bins).reshape(-1)
        # h = np.append(h, path + '/' + i)
        histograms = np.vstack([histograms, h])
    return histograms, names


def distances(histograms, measure, compare_img):
    dst = np.empty(1)
    hist0 = histograms[compare_img]
    for hist in histograms:
        ln = compare_histograms(hist0, hist, len(hist0), measure)
        dst = np.append(dst, ln)
    return dst

bins = 8
rnge = np.power(bins, 3)
compare_img = 80

histograms, names_unsorted = make_histograms('dataset', bins)

euclidean_distances_unsorted = distances(histograms, "euclidean", compare_img)[1:]
idx1 = np.argsort(euclidean_distances_unsorted)
names1 = np.array(names_unsorted)[idx1]
histograms1 = np.array(histograms)[idx1]
euclidean_distances = np.array(euclidean_distances_unsorted)[idx1]

plt.subplot(2, 6, 1)
obj01 = cv2.imread('dataset/' + names1[0])
obj01 = cv2.cvtColor(obj01, cv2.COLOR_BGR2RGB)
plt.imshow(obj01)
plt.subplot(2, 6, 2)
obj02 = cv2.imread('dataset/' + names1[1])
obj02 = cv2.cvtColor(obj02, cv2.COLOR_BGR2RGB)
plt.imshow(obj02)
plt.subplot(2, 6, 3)
obj03 = cv2.imread('dataset/' + names1[2])
obj03 = cv2.cvtColor(obj03, cv2.COLOR_BGR2RGB)
plt.imshow(obj03)
plt.subplot(2, 6, 4)
obj04 = cv2.imread('dataset/' + names1[3])
obj04 = cv2.cvtColor(obj04, cv2.COLOR_BGR2RGB)
plt.imshow(obj04)
plt.subplot(2, 6, 5)
obj05 = cv2.imread('dataset/' + names1[4])
obj05 = cv2.cvtColor(obj05, cv2.COLOR_BGR2RGB)
plt.imshow(obj05)
plt.subplot(2, 6, 6)
obj06 = cv2.imread('dataset/' + names1[5])
obj06 = cv2.cvtColor(obj06, cv2.COLOR_BGR2RGB)
plt.imshow(obj06)
plt.subplot(2, 6, 7)
plt.bar(np.arange(0, rnge, 1), histograms1[0], width=4)
plt.title(str(round(euclidean_distances[0], 2)))
plt.subplot(2, 6, 8)
plt.bar(np.arange(0, rnge, 1), histograms1[1], width=4)
plt.title(str(round(euclidean_distances[1], 2)))
plt.subplot(2, 6, 9)
plt.bar(np.arange(0, rnge, 1), histograms1[2], width=4)
plt.title(str(round(euclidean_distances[2], 2)))
plt.subplot(2, 6, 10)
plt.bar(np.arange(0, rnge, 1), histograms1[3], width=4)
plt.title(str(round(euclidean_distances[3], 2)))
plt.subplot(2, 6, 11)
plt.bar(np.arange(0, rnge, 1), histograms1[4], width=4)
plt.title(str(round(euclidean_distances[4], 2)))
plt.subplot(2, 6, 12)
plt.bar(np.arange(0, rnge, 1), histograms1[5], width=4)
plt.title(str(round(euclidean_distances[5], 2)))
plt.suptitle("Euclidean")
plt.show()



chi_distances_unsorted = distances(histograms, "chi", compare_img)[1:]
idx2 = np.argsort(chi_distances_unsorted)
names2 = np.array(names_unsorted)[idx2]
histograms2 = np.array(histograms)[idx2]
chi_distances = np.array(chi_distances_unsorted)[idx2]

plt.subplot(2, 6, 1)
obj11 = cv2.imread('dataset/' + names2[0])
obj11 = cv2.cvtColor(obj11, cv2.COLOR_BGR2RGB)
plt.imshow(obj11)
plt.subplot(2, 6, 2)
obj12 = cv2.imread('dataset/' + names2[1])
obj12 = cv2.cvtColor(obj12, cv2.COLOR_BGR2RGB)
plt.imshow(obj12)
plt.subplot(2, 6, 3)
obj13 = cv2.imread('dataset/' + names2[2])
obj13 = cv2.cvtColor(obj13, cv2.COLOR_BGR2RGB)
plt.imshow(obj13)
plt.subplot(2, 6, 4)
obj14 = cv2.imread('dataset/' + names2[3])
obj14 = cv2.cvtColor(obj14, cv2.COLOR_BGR2RGB)
plt.imshow(obj14)
plt.subplot(2, 6, 5)
obj15 = cv2.imread('dataset/' + names2[4])
obj15 = cv2.cvtColor(obj15, cv2.COLOR_BGR2RGB)
plt.imshow(obj15)
plt.subplot(2, 6, 6)
obj16 = cv2.imread('dataset/' + names2[5])
obj16 = cv2.cvtColor(obj16, cv2.COLOR_BGR2RGB)
plt.imshow(obj16)
plt.subplot(2, 6, 7)
plt.bar(np.arange(0, rnge, 1), histograms2[0], width=4)
plt.title(str(round(chi_distances[0], 2)))
plt.subplot(2, 6, 8)
plt.bar(np.arange(0, rnge, 1), histograms2[1], width=4)
plt.title(str(round(chi_distances[1], 2)))
plt.subplot(2, 6, 9)
plt.bar(np.arange(0, rnge, 1), histograms2[2], width=4)
plt.title(str(round(chi_distances[2], 2)))
plt.subplot(2, 6, 10)
plt.bar(np.arange(0, rnge, 1), histograms2[3], width=4)
plt.title(str(round(chi_distances[3], 2)))
plt.subplot(2, 6, 11)
plt.bar(np.arange(0, rnge, 1), histograms2[4], width=4)
plt.title(str(round(chi_distances[4], 2)))
plt.subplot(2, 6, 12)
plt.bar(np.arange(0, rnge, 1), histograms2[5], width=4)
plt.title(str(round(chi_distances[5], 2)))
plt.suptitle("Chi-Squared")
plt.show()



intersection_distances_unsorted = distances(histograms, "intersection", compare_img)[1:]
idx3 = np.argsort(intersection_distances_unsorted)
names3 = np.array(names_unsorted)[idx3]
histograms3 = np.array(histograms)[idx3]
intersection_distances = np.array(intersection_distances_unsorted)[idx3]

plt.subplot(2, 6, 1)
obj21 = cv2.imread('dataset/' + names3[0])
obj21 = cv2.cvtColor(obj21, cv2.COLOR_BGR2RGB)
plt.imshow(obj21)
plt.subplot(2, 6, 2)
obj22 = cv2.imread('dataset/' + names3[1])
obj22 = cv2.cvtColor(obj22, cv2.COLOR_BGR2RGB)
plt.imshow(obj22)
plt.subplot(2, 6, 3)
obj23 = cv2.imread('dataset/' + names3[2])
obj23 = cv2.cvtColor(obj23, cv2.COLOR_BGR2RGB)
plt.imshow(obj23)
plt.subplot(2, 6, 4)
obj24 = cv2.imread('dataset/' + names3[3])
obj24 = cv2.cvtColor(obj24, cv2.COLOR_BGR2RGB)
plt.imshow(obj24)
plt.subplot(2, 6, 5)
obj25 = cv2.imread('dataset/' + names3[4])
obj25 = cv2.cvtColor(obj25, cv2.COLOR_BGR2RGB)
plt.imshow(obj25)
plt.subplot(2, 6, 6)
obj26 = cv2.imread('dataset/' + names3[5])
obj26 = cv2.cvtColor(obj26, cv2.COLOR_BGR2RGB)
plt.imshow(obj26)
plt.subplot(2, 6, 7)
plt.bar(np.arange(0, rnge, 1), histograms3[0], width=4)
plt.title(str(round(intersection_distances[0], 2)))
plt.subplot(2, 6, 8)
plt.bar(np.arange(0, rnge, 1), histograms3[1], width=4)
plt.title(str(round(intersection_distances[1], 2)))
plt.subplot(2, 6, 9)
plt.bar(np.arange(0, rnge, 1), histograms3[2], width=4)
plt.title(str(round(intersection_distances[2], 2)))
plt.subplot(2, 6, 10)
plt.bar(np.arange(0, rnge, 1), histograms3[3], width=4)
plt.title(str(round(intersection_distances[3], 2)))
plt.subplot(2, 6, 11)
plt.bar(np.arange(0, rnge, 1), histograms3[4], width=4)
plt.title(str(round(intersection_distances[4], 2)))
plt.subplot(2, 6, 12)
plt.bar(np.arange(0, rnge, 1), histograms3[5], width=4)
plt.title(str(round(intersection_distances[5], 2)))
plt.suptitle("Intersection")
plt.show()



hellinger_distances_unsorted = distances(histograms, "hellinger", compare_img)[1:]
idx4 = np.argsort(hellinger_distances_unsorted)
names4 = np.array(names_unsorted)[idx4]
histograms4 = np.array(histograms)[idx4]
hellinger_distances = np.array(hellinger_distances_unsorted)[idx4]

plt.subplot(2, 6, 1)
obj31 = cv2.imread('dataset/' + names4[0])
obj31 = cv2.cvtColor(obj31, cv2.COLOR_BGR2RGB)
plt.imshow(obj31)
plt.subplot(2, 6, 2)
obj32 = cv2.imread('dataset/' + names4[1])
obj32 = cv2.cvtColor(obj32, cv2.COLOR_BGR2RGB)
plt.imshow(obj32)
plt.subplot(2, 6, 3)
obj33 = cv2.imread('dataset/' + names4[2])
obj33 = cv2.cvtColor(obj33, cv2.COLOR_BGR2RGB)
plt.imshow(obj33)
plt.subplot(2, 6, 4)
obj34 = cv2.imread('dataset/' + names4[3])
obj34 = cv2.cvtColor(obj34, cv2.COLOR_BGR2RGB)
plt.imshow(obj34)
plt.subplot(2, 6, 5)
obj35 = cv2.imread('dataset/' + names4[4])
obj35 = cv2.cvtColor(obj35, cv2.COLOR_BGR2RGB)
plt.imshow(obj35)
plt.subplot(2, 6, 6)
obj36 = cv2.imread('dataset/' + names4[5])
obj36 = cv2.cvtColor(obj36, cv2.COLOR_BGR2RGB)
plt.imshow(obj36)
plt.subplot(2, 6, 7)
plt.bar(np.arange(0, rnge, 1), histograms4[0], width=4)
plt.title(str(round(hellinger_distances[0], 2)))
plt.subplot(2, 6, 8)
plt.bar(np.arange(0, rnge, 1), histograms4[1], width=4)
plt.title(str(round(hellinger_distances[1], 2)))
plt.subplot(2, 6, 9)
plt.bar(np.arange(0, rnge, 1), histograms4[2], width=4)
plt.title(str(round(hellinger_distances[2], 2)))
plt.subplot(2, 6, 10)
plt.bar(np.arange(0, rnge, 1), histograms4[3], width=4)
plt.title(str(round(hellinger_distances[3], 2)))
plt.subplot(2, 6, 11)
plt.bar(np.arange(0, rnge, 1), histograms4[4], width=4)
plt.title(str(round(hellinger_distances[4], 2)))
plt.subplot(2, 6, 12)
plt.bar(np.arange(0, rnge, 1), histograms4[5], width=4)
plt.title(str(round(hellinger_distances[5], 2)))
plt.suptitle("Hellinger")
plt.show()


# Question: Which distance is in your opinion best suited for image retrieval? How
# does the retrieved sequence change if you use a different number of bins? Is the
# execution time affected by the number of bins?
# It seems like Hellinger or Chi Squared are the best for image retrieval, Hellinger being better in some cases.
# The more bins I use the more it is accurate, but it also takes a lot longer to compute.


# e --------------------------------------------------------------------------------------------------------------------

plt.subplot(1, 2, 1)
plt.plot(euclidean_distances_unsorted)
index = np.argpartition(euclidean_distances_unsorted, 6)
plt.plot(euclidean_distances_unsorted, markevery=index[:6], marker='o', fillstyle='none', linestyle='none')
plt.subplot(1, 2, 2)
plt.plot(euclidean_distances)
plt.plot(euclidean_distances[0:6], marker='o', fillstyle='none', linestyle='none')
plt.suptitle('Euclidean')
plt.show()

plt.subplot(1, 2, 1)
plt.plot(chi_distances_unsorted)
index = np.argpartition(chi_distances_unsorted, 6)
plt.plot(chi_distances_unsorted, markevery=index[:6], marker='o', fillstyle='none', linestyle='none')
plt.subplot(1, 2, 2)
plt.plot(chi_distances)
plt.plot(chi_distances[0:6], marker='o', fillstyle='none', linestyle='none')
plt.suptitle('Chi-Square')
plt.show()

plt.subplot(1, 2, 1)
plt.plot(intersection_distances_unsorted)
index = np.argpartition(intersection_distances_unsorted, 6)
plt.plot(intersection_distances_unsorted, markevery=index[:6], marker='o', fillstyle='none', linestyle='none')
plt.subplot(1, 2, 2)
plt.plot(intersection_distances)
plt.plot(intersection_distances[0:6], marker='o', fillstyle='none', linestyle='none')
plt.suptitle('Intersection')
plt.show()

plt.subplot(1, 2, 1)
plt.plot(hellinger_distances_unsorted)
index = np.argpartition(hellinger_distances_unsorted, 6)
plt.plot(hellinger_distances_unsorted, markevery=index[:6], marker='o', fillstyle='none', linestyle='none')
plt.subplot(1, 2, 2)
plt.plot(hellinger_distances)
plt.plot(hellinger_distances[0:6], marker='o', fillstyle='none', linestyle='none')
plt.suptitle('Hellinger')
plt.show()


# f --------------------------------------------------------------------------------------------------------------------

hist_sum = np.sum(histograms, axis=0)
hist_sum /= np.sum(hist_sum)

w = np.copy(hist_sum)
lam = 512
w = np.exp(-lam * hist_sum)
# w /= np.sum(w)

plt.subplot(1, 2, 1)
plt.bar(np.arange(0, rnge, 1), hist_sum, width=4)
plt.title("Sum of all histograms")
plt.subplot(1, 2, 2)
plt.bar(np.arange(0, rnge, 1), w, width=4)
plt.title("Weights lambda = " + str(lam))
plt.show()



plt.subplot(3, 6, 1)
obj = cv2.imread('dataset/' + names4[0])
obj = cv2.cvtColor(obj, cv2.COLOR_BGR2RGB)
plt.imshow(obj)
plt.subplot(3, 6, 2)
obj = cv2.imread('dataset/' + names4[1])
obj = cv2.cvtColor(obj, cv2.COLOR_BGR2RGB)
plt.imshow(obj)
plt.subplot(3, 6, 3)
obj = cv2.imread('dataset/' + names4[2])
obj = cv2.cvtColor(obj, cv2.COLOR_BGR2RGB)
plt.imshow(obj)
plt.subplot(3, 6, 4)
obj = cv2.imread('dataset/' + names4[3])
obj = cv2.cvtColor(obj, cv2.COLOR_BGR2RGB)
plt.imshow(obj)
plt.subplot(3, 6, 5)
obj = cv2.imread('dataset/' + names4[4])
obj = cv2.cvtColor(obj, cv2.COLOR_BGR2RGB)
plt.imshow(obj)
plt.subplot(3, 6, 6)
obj = cv2.imread('dataset/' + names4[5])
obj = cv2.cvtColor(obj, cv2.COLOR_BGR2RGB)
plt.imshow(obj)
plt.subplot(3, 6, 7)
plt.bar(np.arange(0, rnge, 1), histograms4[0], width=4)
plt.title(str(round(hellinger_distances[0], 2)))
plt.subplot(3, 6, 8)
plt.bar(np.arange(0, rnge, 1), histograms4[0], width=4)
plt.title(str(round(hellinger_distances[1], 2)))
plt.subplot(3, 6, 9)
plt.bar(np.arange(0, rnge, 1), histograms4[0], width=4)
plt.title(str(round(hellinger_distances[2], 2)))
plt.subplot(3, 6, 10)
plt.bar(np.arange(0, rnge, 1), histograms4[0], width=4)
plt.title(str(round(hellinger_distances[3], 2)))
plt.subplot(3, 6, 11)
plt.bar(np.arange(0, rnge, 1), histograms4[0], width=4)
plt.title(str(round(hellinger_distances[4], 2)))
plt.subplot(3, 6, 12)
plt.bar(np.arange(0, rnge, 1), histograms4[0], width=4)
plt.title(str(round(hellinger_distances[5], 2)))
plt.subplot(3, 6, 13)
hist = np.multiply(histograms4[0], w)
hist /= np.sum(hist)
plt.bar(np.arange(0, rnge, 1), hist, width=4)
plt.subplot(3, 6, 14)
hist = np.multiply(histograms4[1], w)
hist /= np.sum(hist)
plt.bar(np.arange(0, rnge, 1), hist, width=4)
plt.subplot(3, 6, 15)
hist = np.multiply(histograms4[2], w)
hist /= np.sum(hist)
plt.bar(np.arange(0, rnge, 1), hist, width=4)
plt.subplot(3, 6, 16)
hist = np.multiply(histograms4[3], w)
hist /= np.sum(hist)
plt.bar(np.arange(0, rnge, 1), hist, width=4)
plt.subplot(3, 6, 17)
hist = np.multiply(histograms4[4], w)
hist /= np.sum(hist)
plt.bar(np.arange(0, rnge, 1), hist, width=4)
plt.subplot(3, 6, 18)
hist = np.multiply(histograms4[5], w)
hist /= np.sum(hist)
plt.bar(np.arange(0, rnge, 1), hist, width=4)
plt.show()

i = 5
img = cv2.imread('dataset/' + names_unsorted[i])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(2, 3, 1)
plt.imshow(img)
plt.subplot(2, 3, 2)
plt.bar(np.arange(0, rnge, 1), histograms[i], width=4)
plt.subplot(2, 3, 3)
hist = np.multiply(histograms[i], w)
hist /= np.sum(hist)
plt.bar(np.arange(0, rnge, 1), hist, width=4)

i = 10
img = cv2.imread('dataset/' + names_unsorted[i])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(2, 3, 4)
plt.imshow(img)
plt.subplot(2, 3, 5)
plt.bar(np.arange(0, rnge, 1), histograms[i], width=4)
plt.subplot(2, 3, 6)
hist = np.multiply(histograms[i], w)
hist /= np.sum(hist)
plt.bar(np.arange(0, rnge, 1), hist, width=4)
plt.show()