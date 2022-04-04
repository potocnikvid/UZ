import numpy as np
import cv2
from matplotlib import pyplot as plt
import warnings

# a
bird = cv2.imread('images/bird.jpg')
bird = cv2.cvtColor(bird, cv2.COLOR_BGR2RGB)
bird_gray = cv2.cvtColor(bird, cv2.COLOR_RGB2GRAY)

threshold = 74
# 1
bird_gray[bird_gray < threshold] = 0
bird_gray[bird_gray >= threshold] = 1
# 2
bird_gray2 = np.copy(bird_gray)
np.where([bird_gray2 < threshold], 0, 1)

plt.subplot(1, 2, 1)
plt.imshow(bird, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(bird_gray2, cmap='gray')
plt.show()

# b
def my_hist_uin8(i_gray, bins):
    h = np.zeros(bins)
    i_gray2 = np.copy(i_gray)
    i_arr = i_gray2.reshape(-1)
    i_arr //= int(np.ceil(256/bins))
    np.add.at(h, i_arr, 1)
    return h/sum(h)


bird = cv2.imread('images/bird.jpg')
bird = cv2.cvtColor(bird, cv2.COLOR_BGR2RGB)
bird_gray = cv2.cvtColor(bird, cv2.COLOR_RGB2GRAY)

bins1 = 20
bins2 = 100

plt.subplot(1, 3, 1)
plt.imshow(bird_gray, cmap='gray')
plt.subplot(1, 3, 2)
h1 = my_hist_uin8(bird_gray, bins1)
plt.bar(np.arange(0, bins1, 1, np.uint8), h1)
plt.subplot(1, 3, 3)
h2 = my_hist_uin8(bird_gray, bins2)
plt.bar(np.arange(0, bins2, 1, np.uint8), h2)
plt.show()
# Question: The histograms are usually normalized by dividing the result by the
# sum of all cells. Why is that?
# So that images of different sizes are easily comparable. Also to see % of bins (more useful info)

# c
def my_hist(i_gray, bins):
    h = np.zeros(bins)
    i_gray2 = np.copy(i_gray)
    i_arr = i_gray2.reshape(-1)
    min = np.amin(i_arr)
    max = np.amax(i_arr)
    full_rng = max - min + 1
    bin_rng = int(np.ceil(full_rng/bins))
    min_ix = min // bin_rng
    i_arr //= int(np.ceil(full_rng/bins))
    i_arr -= min_ix
    np.add.at(h, i_arr, 1)
    # print(len(h))
    # print(len(i_arr))
    return h/sum(h)


bird_gray[bird_gray < 50] = 50
bird_gray[bird_gray > 100] = 200

plt.subplot(1, 3, 1)
plt.imshow(bird_gray, cmap='gray')
plt.subplot(1, 3, 2)
h1 = my_hist(bird_gray, bins1)
plt.bar(np.arange(0, bins1, 1, np.uint8), h1)
plt.subplot(1, 3, 3)
h2 = my_hist(bird_gray, bins2)
plt.bar(np.arange(0, bins2, 1, np.uint8), h2)
plt.show()


# d
me = cv2.imread('images/my_photo-2.jpg')
me = cv2.cvtColor(me, cv2.COLOR_BGR2RGB)
me_gray = cv2.cvtColor(me, cv2.COLOR_RGB2GRAY)
me2 = cv2.imread('images/my_photo-3.jpg')
me2 = cv2.cvtColor(me2, cv2.COLOR_BGR2RGB)
me2_gray = cv2.cvtColor(me2, cv2.COLOR_RGB2GRAY)
me3 = cv2.imread('images/my_photo-4.jpg')
me3 = cv2.cvtColor(me3, cv2.COLOR_BGR2RGB)
me3_gray = cv2.cvtColor(me3, cv2.COLOR_RGB2GRAY)
me4 = cv2.imread('images/my_photo-5.jpg')
me4 = cv2.cvtColor(me4, cv2.COLOR_BGR2RGB)
me4_gray = cv2.cvtColor(me4, cv2.COLOR_RGB2GRAY)

plt.subplot(4, 2, 1)
plt.imshow(me, cmap='gray')
plt.subplot(4, 2, 2)
me_hist = my_hist_uin8(me_gray, 20)
plt.bar(np.arange(0, 20, 1, np.uint8), me_hist)

plt.subplot(4, 2, 3)
plt.imshow(me2, cmap='gray')
plt.subplot(4, 2, 4)
me_hist = my_hist_uin8(me2_gray, 20)
plt.bar(np.arange(0, 20, 1, np.uint8), me_hist)

plt.subplot(4, 2, 5)
plt.imshow(me3, cmap='gray')
plt.subplot(4, 2, 6)
me_hist = my_hist_uin8(me3_gray, 20)
plt.bar(np.arange(0, 20, 1, np.uint8), me_hist)
2
plt.subplot(4, 2, 7)
plt.imshow(me4, cmap='gray')
plt.subplot(4, 2, 8)
me_hist = my_hist_uin8(me4_gray, 20)
plt.bar(np.arange(0, 20, 1, np.uint8), me_hist)
plt.show()

# e


def otsu(img_gray):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        final_thresh = -1
        final_value = -1
        hist, bins = np.histogram(img_gray, np.array(range(0, 257)))
        hist = 1.0 * hist/np.sum(hist)
        for t in bins[1:-1]:
            bg_weight = np.sum(hist[:t])
            fg_weight = np.sum(hist[t:])
            bg_mean = np.sum(np.arange(0, t, 1) * hist[:t]) / bg_weight
            fg_mean = np.sum(np.arange(t, 256, 1) * hist[t:]) / fg_weight
            #between class variance
            bcw = bg_weight * fg_weight * np.power(bg_mean - fg_mean, 2)
            if bcw > final_value:
                final_thresh = t
                final_value = bcw
        final_img = img_gray.copy()
        final_img[img_gray > final_thresh] = 255
        final_img[img_gray < final_thresh] = 0
        return final_img, final_thresh - 1


bird = cv2.imread('images/bird.jpg')
bird = cv2.cvtColor(bird, cv2.COLOR_BGR2RGB)
bird_gray = cv2.cvtColor(bird, cv2.COLOR_RGB2GRAY)

ret, bird_thresh = cv2.threshold(bird_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
plt.subplot(1, 2, 1)
plt.xlabel(ret)
plt.imshow(bird_thresh, cmap='gray')
plt.subplot(1, 2, 2)
otsu_img, otsu_thresh = otsu(bird_gray)
plt.xlabel(otsu_thresh)
plt.imshow(otsu_img, cmap='gray')
plt.show()

harewood = cv2.imread('images/harewood.jpg')
harewood = cv2.cvtColor(harewood, cv2.COLOR_BGR2RGB)
harewood_gray = cv2.cvtColor(harewood, cv2.COLOR_RGB2GRAY)

ret, harewood_thresh = cv2.threshold(harewood_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
plt.subplot(1, 2, 1)
plt.xlabel(ret)
plt.imshow(harewood_thresh, cmap='gray')
plt.subplot(1, 2, 2)
otsu_img, otsu_thresh = otsu(harewood_gray)
plt.xlabel(otsu_thresh)
plt.imshow(otsu_img, cmap='gray')
plt.show()

eagle = cv2.imread('images/eagle.jpg')
eagle = cv2.cvtColor(eagle, cv2.COLOR_BGR2RGB)
eagle_gray = cv2.cvtColor(eagle, cv2.COLOR_RGB2GRAY)

ret, eagle_thresh = cv2.threshold(eagle_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
plt.subplot(1, 2, 1)
plt.xlabel(ret)
plt.imshow(eagle_thresh, cmap='gray')
plt.subplot(1, 2, 2)
otsu_img, otsu_thresh = otsu(eagle_gray)
plt.xlabel(otsu_thresh)
plt.imshow(otsu_img, cmap='gray')
plt.show()