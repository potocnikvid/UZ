import numpy as np
import cv2
from matplotlib import pyplot as plt
import math


def disparity(pz, f, T):
    disparities = []
    for p in pz:
        disparities.append(f * T / p)
    return disparities


# a

pz = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
d = disparity(pz, 0.25, 12)
plt.plot(pz, d)
plt.xlabel("Pz")
plt.ylabel("disparities")
plt.show()

