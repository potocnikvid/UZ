import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
from a5_utils import normalize_points, draw_epiline, get_grid
import glob

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


def fundamental_matrix(matches):
    A = []
    points_1, t_1 = normalize_points(matches[:, :2])
    points_2, t_2 = normalize_points(matches[:, 2:])

    for i in range(len(matches)):
        A.append([points_1[i][0] * points_2[i][0],
                points_1[i][1] * points_2[i][0],
                points_2[i][0],
                points_1[i][0] * points_2[i][1],
                points_1[i][1] * points_2[i][1],
                points_2[i][1],
                points_1[i][0],
                points_1[i][1],
                1])

    # print(A)
    _, _, vt = np.linalg.svd(A)
    v_n = vt.transpose()[:, -1]
    f_t = np.reshape(v_n, (3, 3))

    u, d, vt = np.linalg.svd(f_t)
    d = np.diag([*d[:2], 0])
    f = u @ d @ vt

    u, d, vt = np.linalg.svd(f_t)

    e1 = vt.transpose()[:, 2] / vt.transpose()[2, 2]
    e2 = u[:, 2] / u[2, 2]

    f = t_2.transpose() @ f @ t_1
    return f, e1, e2

def distance(point, line):
    return np.abs(line[0] * point[0] + line[1] * point[1] + line[2]) / \
           np.sqrt(np.square(line[0]) + np.square(line[1]))


def get_inliers(f, points, threshold):
    inliers = []
    outliers = []
    for i in range(len(points)):
        line2 = f @ np.append(points[i, :2], 1)
        line1 = f.transpose() @ np.append(points[i, 2:], 1)
        dist1 = distance(points[i, :2], line1)
        dist2 = distance(points[i, 2:], line2)
        if dist1 < threshold and dist2 < threshold:
            inliers.append(points[i].tolist())
        else:
            outliers.append(points[i].tolist())
    return np.array(inliers), np.array(outliers)


def ransac_fundamental(points, threshold, k):
    max_inliers = -1
    best_f = None
    best_inliers = None
    best_outliers = None
    for i in range(k):
        np.random.shuffle(points)
        sample = points[:8, :]
        f, e1, e2 = fundamental_matrix(sample)
        inliers, outliers = get_inliers(f, points, threshold)
        if len(inliers) >= max_inliers:
            max_inliers = len(inliers)
            best_inliers = inliers
            best_outliers = outliers
            best_f = f
    return best_f, best_inliers, best_outliers

def find_matches(img1, img2):
    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

    matcher = cv2.FlannBasedMatcher_create()
    matches = matcher.knnMatch(descriptors_1, descriptors_2, 2)
    res = []
    for m1, m2 in matches:
        if m1.distance < 0.7 * m2.distance:
            x1 = keypoints_1[m1.queryIdx].pt[0]
            y1 = keypoints_1[m1.queryIdx].pt[1]
            x2 = keypoints_2[m1.trainIdx].pt[0]
            y2 = keypoints_2[m1.trainIdx].pt[1]
            match = [x1, y1, x2, y2]
            res.append(match)
    return np.array(res)



def shear_symmetric_form(x):
    return np.array([[0, -1, x[1]], [1, 0, -x[0]], [-x[1], x[0], 0]])


def triangulate(x, p_1, p_2):
    points_3d = []
    A = np.zeros((4, 4))
    for i in range(len(x)):
        product_1 = shear_symmetric_form(x[i, :2]) @ p_1
        product_2 = shear_symmetric_form(x[i, 2:]) @ p_2
        A[0] = product_1[0]
        A[1] = product_1[1]
        A[2] = product_2[0]
        A[3] = product_2[1]
        u, d, vt = np.linalg.svd(A)
        X = vt[-1]
        X = X / X[-1]
        points_3d.append(X.tolist())
    return np.array(points_3d)



# a
def a():
    img1 = cv2.imread('data/epipolar/house1.jpg')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread('data/epipolar/house2.jpg')
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    points = read_data("./data/epipolar/house_points.txt")
    calibration_1 = read_data("./data/epipolar/house1_camera.txt")
    calibration_2 = read_data("./data/epipolar/house2_camera.txt")

    points_3d = triangulate(points, calibration_1, calibration_2)

    fig = plt.figure(figsize=(15, 12))

    plt.subplot(1, 3, 1)
    plt.imshow(img1, cmap='gray')
    plt.scatter(points[:, 0], points[:, 1], color='red', marker='o')
    plt.subplot(1, 3, 2)
    plt.imshow(img2, cmap='gray')
    plt.scatter(points[:, 2], points[:, 3], color='red', marker='o')

    ax = fig.add_subplot(133, projection='3d')  # define 3D subplot
    ax.view_init(23, -23)
    T = np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]])  # transformation matrix
    points_3d = np.dot(points_3d[:, :3], T)
    for i, pt in enumerate(points_3d):
        plt.plot([pt[0]], [pt[1]], [pt[2]], 'r.')  # plot points
        ax.text(pt[0], pt[1], pt[2], str(i))  # plot indices
    plt.show()




# b

def calibrate(images):
    objp = get_grid()
    imgpoints = []
    objpoints = []
    plt.figure(figsize=(15, 12))
    i = 0
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, circles = cv2.findCirclesGrid(gray, (4, 11), flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(circles)
            cv2.drawChessboardCorners(img, (4, 11), circles, ret)
        i += 1
        plt.subplot(3, 3, i)
        plt.imshow(img, cmap="gray")
    plt.show()
    return objpoints, imgpoints


def b():
    images = glob.glob('*/epipolar/pointa/*')
    objpoints, imgpoints = calibrate(images)


    img1 = cv2.imread('data/epipolar/PXL_20211219_193141053.jpg')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread('data/epipolar/PXL_20211219_193136665.jpg')
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    h1, w1 = np.shape(img1)
    h2, w2 = np.shape(img2)

    ret, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints, img1.shape[::-1], None, None)
    newcameramtx1, roi1 = cv2.getOptimalNewCameraMatrix(mtx1, dist1, (w1, h1), 1, (w1, h1))
    dst1 = cv2.undistort(img1, mtx1, dist1, None, newcameramtx1)
    x, y, w, h = roi1
    dst1 = dst1[y:y + h, x:x + w]
    plt.subplot(1, 2, 1)
    plt.imshow(dst1, cmap="gray")

    ret, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints, img2.shape[::-1], None, None)
    newcameramtx2, roi2 = cv2.getOptimalNewCameraMatrix(mtx2, dist2, (w2, h2), 1, (w2, h2))
    dst2 = cv2.undistort(img2, mtx2, dist2, None, newcameramtx2)
    x, y, w, h = roi2
    dst2 = dst2[y:y + h, x:x + w]
    plt.subplot(1, 2, 2)
    plt.imshow(dst2, cmap="gray")
    plt.show()



    points = find_matches(dst1, dst2)

    calibration_1 = read_data("./data/epipolar/house1_camera.txt")
    calibration_2 = read_data("./data/epipolar/house2_camera.txt")

    points_3d = triangulate(points, calibration_1, calibration_2)

    fig = plt.figure(figsize=(15, 12))
    plt.subplot(1, 3, 1)
    plt.imshow(img1, cmap='gray')
    plt.scatter(points[:, 0], points[:, 1], color='red', marker='o')
    plt.subplot(1, 3, 2)
    plt.imshow(img2, cmap='gray')
    plt.scatter(points[:, 2], points[:, 3], color='red', marker='o')
    ax = fig.add_subplot(133, projection='3d')  # define 3D subplot
    ax.view_init(20, -20)
    T = np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]])  # transformation matrix
    points_3d = np.dot(points_3d[:, :3], T)
    for i, pt in enumerate(points_3d):
        plt.plot([pt[0]], [pt[1]], [pt[2]], 'r.')  # plot points
        ax.text(pt[0], pt[1], pt[2], str(i))  # plot indices
    plt.show()

a()
b()