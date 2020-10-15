import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eig, inv, norm


MIN_RAW_MATCH = 70
MIN_INLIER_MATCH = 45
RANSAC_THRESH = 10.0
PERSIST_FRAME = 7
EXPORT_RES = (960, 540)

cap = cv.VideoCapture('Data/Test/6.MOV')
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
frame = None
i = 0


def getMatches(frameDes, desS, num):
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
                print(str(num) + "-\tInl Size: " + str(inliers))
    elif inliers > len(matches) * 0.65:
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
    if img[0, 0] != 0:
        print("WARNING: Filling something you shouldn't")
    cv.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv

    return im_out


def writeFrame(matrix, num, frame):
    if matrix is None:
        return

    ret, alter = altVideos[num].read()
    alter = cv.cvtColor(alter, cv.COLOR_BGR2BGRA)
    alter = cv.rotate(alter, cv.ROTATE_90_CLOCKWISE)

    sampleH, sampleW, d = sample[num].shape
    res = cv.resize(alter, (sampleW, sampleH), interpolation=cv.INTER_CUBIC)

    h, w, d = frame.shape
    wrapped = cv.warpPerspective(res, Ms[num], (w, h))

    t1 = sample[num]
    t1 = cv.cvtColor(t1, cv.COLOR_BGR2GRAY)

    t2 = cv.warpPerspective(frame, inv(
        Ms[num]), (sampleW, sampleH))
    t2 = cv.cvtColor(t2, cv.COLOR_BGR2GRAY)

    test = cv.bitwise_and(t1, t1)
    test[:, :] = (abs(t1[:, :] - t2[:, :]) > 190) * 255

    result = test[:, :]

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    result = cv.morphologyEx(result, cv.MORPH_OPEN, kernel)

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

    result = cv.warpPerspective(drawing, Ms[num], (w, h))

    ret, res = cv.threshold(res, 0, 255, cv.THRESH_BINARY)
    res = cv.warpPerspective(res, Ms[num], (w, h))
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


while(cap.isOpened()):
    print(i, flush=True)
    i += 1
    ret, frame = cap.read()

    if(ret == False or i > 1000):
        break

    frame = cv.resize(frame, EXPORT_RES, interpolation=cv.INTER_CUBIC)
    # frameGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    fkp, fdes = sift.detectAndCompute(frame, None)

    for num in range(10):
        qualified = getMatches(des, deses[num], num)
        M = getTransformationMatrix(qualified, kps[num], fkp, num)

        if (M is None) and (Ms[num] is not None) and (counter[num] < PERSIST_FRAME):
            counter[num] += 1
            M = Ms[num]
        else:
            counter[num] = 0
            Ms[num] = M

        temp = writeFrame(M, num, frame)
        if temp is not None:
            frame = temp[::]

    writer.write(frame)
    cv.namedWindow("Frame", cv.WINDOW_NORMAL)
    cv.imshow('Frame', frame)
    cv.resizeWindow("Frame", 960, 540)
    cv.waitKey(1)
    print("--------")

cv.destroyAllWindows()
writer.release()
cap.release()
