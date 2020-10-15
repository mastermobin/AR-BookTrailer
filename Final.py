import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eig, inv, norm


MIN_RAW_MATCH = 60
MIN_INLIER_MATCH = 45
RANSAC_THRESH = 10.0
PERSIST_FRAME = 8
EXPORT_RES = (960, 540)
DEBUG = True

cap = cv.VideoCapture('Data/Test/13.MOV')
# Read All Trailers
altVideos = [None, None, None, None, None, None, None, None, None, None, None]
for i in range(10):
    altVideos[i] = cv.VideoCapture('Data/Source/' + str(i) + '.mp4')
writer = cv.VideoWriter('Test.mp4', -1, 30, EXPORT_RES)

# Init Matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(check=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
flann.clear()

# Init Feature Descriptor & Read Samples
sift = cv.xfeatures2d.SIFT_create()
kps = []
deses = []
sample = []
for num in range(10):
    src = cv.imread("Data/Source/" + str(num) + ".jpg")
    # src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    kpt, des = sift.detectAndCompute(src, None)
    kps.append(kpt)
    deses.append(des)
    sample.append(src)

Ms = [None, None, None, None, None, None, None, None, None, None, None]
counter = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
checker = [False, False, False, False,
           False, False, False, False, False, False]
frame = None
i = 0


def fixImage(img):
    equ = cv.equalizeHist(img)
    return equ


def getMatches(frameDes, desS, num=-1):
    matches = flann.knnMatch(queryDescriptors=fdes,
                             trainDescriptors=desS,
                             k=2)
    qualified = []
    try:
        for pair in matches:
            if len(pair) == 2:
                m = pair[0]
                n = pair[1]
                if m != None and n != None:
                    if m.distance < 0.7*n.distance:
                        qualified.append(m)
            elif len(pair) == 1:
                qualified.append(pair[0])
        if num != -1:
            print(str(num) + "-Raw Size: " + str(len(qualified)))
    except:
        return qualified

    return qualified


def getTransformationMatrix(matches, sps, fkp, num=-1):
    if len(matches) < 20:
        return None

    # Calculate Inliers
    src_pts = np.float32(
        [fkp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [sps[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, RANSAC_THRESH)
    matchesMask = mask.ravel().tolist()
    inliers = matchesMask.count(1)

    if len(matches) > MIN_RAW_MATCH:
        if(inliers > MIN_INLIER_MATCH):
            if num != -1:
                print(str(num) + "-\tInl Size: " + str(inliers) + " ✔")
            return M
        else:
            if num != -1:
                print(str(num) + "-\tInl Size: " +
                      str(inliers) + " Fallback To Ratio")

    ratio = float(inliers) / len(matches)
    if (ratio > 0.6) and (inliers > 15):
        if num != -1:
            print(str(num) + "-\tInl Ratio: " +
                  str(inliers / len(matches)) + " ✔")
        return M

    return None


def fill_holes(img):
    im_th = img.copy()

    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    cv.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv

    return im_out


def getArea(img):
    x = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, contours, _ = cv.findContours(
        x, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    max_id = -1
    max_area = -1
    for i in range(len(contours)):
        area = cv.contourArea(contours[i])
        if max_area < area:
            max_area = area
            max_id = i

    return max_area


def writeFrame(matrix, num, frame):
    if matrix is None:
        return

    ret, alter = altVideos[num].read()
    alter = cv.cvtColor(alter, cv.COLOR_BGR2BGRA)
    alter = cv.rotate(alter, cv.ROTATE_90_CLOCKWISE)

    sampleH, sampleW, d = sample[num].shape
    res = cv.resize(alter, (sampleW, sampleH), interpolation=cv.INTER_CUBIC)

    h, w, d = frame.shape
    wrapped = cv.warpPerspective(res, matrix, (w, h))

    t1 = sample[num]
    t1 = cv.cvtColor(t1, cv.COLOR_BGR2GRAY)
    t1 = fixImage(t1)
    t1 = cv.GaussianBlur(t1, (19, 19), 7)
    if DEBUG:
        cv.namedWindow("Test1", cv.WINDOW_NORMAL)
        cv.imshow('Test1', t1)
        cv.resizeWindow("Test1", 450, 800)
        cv.waitKey(1)

    t2 = cv.warpPerspective(frame, inv(
        matrix), (sampleW, sampleH))
    t2 = cv.cvtColor(t2, cv.COLOR_BGR2GRAY)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    canny = cv.morphologyEx(
        cv.Canny(t2, 0, 250), cv.MORPH_DILATE, kernel)

    t2 = fixImage(t2)
    t2 = cv.GaussianBlur(t2, (19, 19), 7)
    if DEBUG:
        cv.namedWindow("Test2", cv.WINDOW_NORMAL)
        cv.imshow('Test2', t2)
        cv.resizeWindow("Test2", 450, 800)
        cv.waitKey(1)

    result = ((t1[:, :].astype('int32') -
               t2[:, :].astype('int32')) < 50).astype('uint8') * 255

    if DEBUG:
        cv.namedWindow("Test3", cv.WINDOW_NORMAL)
        cv.imshow('Test3', result)
        cv.resizeWindow("Test3", 450, 800)
        cv.waitKey(1)

    if DEBUG:
        cv.namedWindow("Test3", cv.WINDOW_NORMAL)
        cv.imshow('Test3', canny)
        cv.resizeWindow("Test3", 450, 800)
        cv.waitKey(1)

    result = cv.subtract(result, canny)

    if DEBUG:
        cv.namedWindow("Test3", cv.WINDOW_NORMAL)
        cv.imshow('Test3', result)
        cv.resizeWindow("Test3", 450, 800)
        cv.waitKey(1)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    result = cv.morphologyEx(result, cv.MORPH_OPEN, kernel)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (8, 8))
    result = cv.morphologyEx(result, cv.MORPH_CLOSE, kernel)

    if DEBUG:
        cv.namedWindow("Test3", cv.WINDOW_NORMAL)
        cv.imshow('Test3', result)
        cv.resizeWindow("Test3", 450, 800)
        cv.waitKey(1)

    # result = fill_holes(result)
    _, contours, _ = cv.findContours(
        result, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    hull = []
    max_id = -1
    max_area = -1
    for i in range(len(contours)):
        area = cv.contourArea(contours[i])
        if max_area < area:
            max_area = area
            max_id = i
        hull.append(cv.convexHull(contours[i], False))

    drawing = np.zeros((result.shape[0], result.shape[1], 3), np.uint8)
    color = (255, 255, 255)
    cv.drawContours(drawing, hull, max_id, color, -1, 8)

    hullArea = getArea(drawing)
    print(hullArea)
    ratio = hullArea / (sampleH * sampleW)
    print(ratio)
    if(ratio < 0.45) or (ratio > 0.85):
        drawing = np.ones(
            (result.shape[0], result.shape[1], 3), np.uint8) * 255

    if DEBUG:
        cv.namedWindow("Test3", cv.WINDOW_NORMAL)
        cv.imshow('Test3', drawing)
        cv.resizeWindow("Test3", 450, 800)
        cv.waitKey(1)

    result = cv.warpPerspective(drawing, matrix, (w, h))

    ret, res = cv.threshold(res, 0, 255, cv.THRESH_BINARY)
    res = cv.warpPerspective(res, matrix, (w, h))
    frameCpy = frame[::]

    y1, y2 = 0, wrapped.shape[0]
    x1, x2 = 0, wrapped.shape[1]

    alpha_s = result[:, :, 2] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        frameCpy[y1:y2, x1:x2, c] = (alpha_s * wrapped[:, :, c] +
                                     alpha_l * frameCpy[y1:y2, x1:x2, c])

    frameCpy = cv.resize(frameCpy, EXPORT_RES,
                         interpolation=cv.INTER_CUBIC)
    return frameCpy


lframe = None
lfdes = None
lfkp = None
GM = None
while(cap.isOpened()):
    print(i, flush=True)
    ret, frame = cap.read()

    if(ret == False or i > 1000):
        break

    frame = cv.resize(frame, EXPORT_RES, interpolation=cv.INTER_CUBIC)
    fkp, fdes = sift.detectAndCompute(frame, None)

    for num in range(10):
        if (not checker[num]) and (i % 10 != 0):
            continue

        qualified = getMatches(des, deses[num], num)
        M = getTransformationMatrix(qualified, kps[num], fkp, num)

        if (M is None) and (Ms[num] is not None) and (counter[num] < PERSIST_FRAME):
            counter[num] += 1
            if GM is None:
                M = Ms[num]
            else:
                print("GM IN USE")
                print(GM)
                M = Ms[num] * GM
        else:
            counter[num] = 0
            if M is not None:
                Ms[num] = M.copy()
                GM = None
                checker[num] = True
            else:
                Ms[num] = None

        temp = writeFrame(M, num, frame)
        if temp is not None:
            frame = temp[::]

    writer.write(frame)
    cv.namedWindow("Frame", cv.WINDOW_NORMAL)
    cv.imshow('Frame', frame)
    cv.resizeWindow("Frame", 960, 540)
    cv.waitKey(1)
    print("--------")
    lfdes = fdes
    lfkp = fkp
    lframe = frame
    i += 1

cv.destroyAllWindows()
writer.release()
cap.release()
