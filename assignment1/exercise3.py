import numpy as np
import cv2
from matplotlib import pyplot as plt

# a
mask = cv2.imread('images/mask.png')

n = 5
SE5 = np.ones((n, n), np.uint8) # create a square structuring element
mask_eroded = cv2.erode(mask, SE5)
mask_dilated = cv2.dilate(mask, SE5)
plt.subplot(1, 2, 1)
plt.imshow(mask_eroded, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(mask_dilated, cmap='gray')
plt.show()

n = 10
SE10 = np.ones((n, n), np.uint8) # create a square structuring element
mask_eroded = cv2.erode(mask, SE10)
mask_dilated = cv2.dilate(mask, SE10)
plt.subplot(1, 2, 1)
plt.imshow(mask_eroded, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(mask_dilated, cmap='gray')
plt.show()

n = 5
SE = np.ones((n, n), np.uint8) # create a square structuring element
mask_eroded = cv2.erode(mask, SE)
mask_dilated = cv2.dilate(mask_eroded, SE)
mask_opening = np.copy(mask_dilated)

mask_dilated = cv2.dilate(mask, SE)
mask_eroded = cv2.erode(mask_dilated, SE)
mask_closing = np.copy(mask_eroded)

plt.subplot(1, 2, 1)
plt.imshow(mask_opening, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(mask_closing, cmap='gray')
plt.show()

# Question: Based on the results, which order of erosion and dilation operations
# produces opening and which closing?
#  Eroding then dilating produces opening and vice-versa in closing

# b
bird = cv2.imread('images/bird.jpg')
bird = cv2.cvtColor(bird, cv2.COLOR_BGR2RGB)
bird_gray = cv2.cvtColor(bird, cv2.COLOR_RGB2GRAY)
ret, bird_thresh = cv2.threshold(bird_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

n = 23
SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
bird_dilated = cv2.dilate(bird_thresh, SE)
bird_eroded = cv2.erode(bird_dilated, SE)
plt.imshow(bird_eroded, cmap='gray')
plt.show()


# c
def immask(picture, i_mask):
    image = np.copy(picture)
    image[i_mask == 0] = (0, 0, 0)
    return image


mask = np.copy(bird_eroded)
bird_masked = immask(bird, mask)
plt.imshow(bird_masked)
plt.show()

# d
eagle = cv2.imread('images/eagle.jpg')
eagle = cv2.cvtColor(eagle, cv2.COLOR_BGR2RGB)
eagle_gray = cv2.cvtColor(eagle, cv2.COLOR_RGB2GRAY)
ret, eagle_thresh = cv2.threshold(eagle_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

n = 4
SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
eagle_eroded = cv2.erode(eagle_thresh, SE)
eagle_eroded = cv2.erode(eagle_eroded, SE)
eagle_eroded = cv2.erode(eagle_eroded, SE)
eagle_dilated = cv2.dilate(eagle_eroded, SE)
eagle_dilated = cv2.dilate(eagle_dilated, SE)
eagle_dilated = cv2.dilate(eagle_dilated, SE)
plt.imshow(eagle_dilated, cmap='gray')
plt.show()

eagle_mask = np.copy(eagle_dilated)
eagle_img = immask(eagle, eagle_mask)
plt.imshow(eagle_img)
plt.show()

# Question: Why is the background included in the mask and not the object? How
# would you fix that in general? (just inverting the pmask if necessary doesn’t count)
# We could count the number of black and white pixels and always invert the mask if
# there are less black pixels. But that would not work if the object was closer in perspective
# (less pixels in the background)

# e

coins = cv2.imread('images/coins.jpg')
coins = cv2.cvtColor(coins, cv2.COLOR_BGR2RGB)
coins_gray = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)
ret, coins_thresh = cv2.threshold(coins_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
coins_thresh = ~coins_thresh

n = 7
SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
coins_dilated = cv2.dilate(coins_thresh, SE)
coins_dilated = cv2.dilate(coins_dilated, SE)
coins_eroded = cv2.erode(coins_dilated, SE)
coins_eroded = cv2.erode(coins_eroded, SE)

output = cv2.connectedComponentsWithStats(coins_eroded, 4, cv2.CV_32S)
num_labels = output[0] - 1
labels = output[1]
stats = output[2]
# removes black pixels
sizes = stats[1:, -1]
max_size = 700
for i in range(0, num_labels):
    if sizes[i] > max_size:
        labels[labels == i + 1] = 0
    else:
        labels[labels == i + 1] = 1

coins_masked = immask(coins, labels)
plt.subplot(1, 2, 1)
plt.imshow(coins)
plt.subplot(1, 2, 2)
plt.imshow(coins_masked)
plt.show()