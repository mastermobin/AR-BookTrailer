import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


MIN_RAW_MATCH = 70
MIN_INLIER_MATCH = 45
RANSAC_THRESH = 10.0
PERSIST_FRAME = 7
EXPORT_RES = (960, 540)

cap = cv.VideoCapture('Data/Test/1.MOV')
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
    src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    kpt, des = sift.detectAndCompute(src, None)
    kps.append(kpt)
    deses.append(des)
    sample.append(src)

Ms = [None, None, None, None, None, None, None, None, None, None, None]
counter = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
frame = None
i = 0


def getMatches(frameDes, num):
    matches = flann.knnMatch(queryDescriptors=fdes,
                             trainDescriptors=deses[num],
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
        print(str(num) + "-Raw Size: " + str(len(qualified)))
    except:
        return qualified

    return qualified


def getTransformationMatrix(matches, num, fkp):
    if len(matches) < 20:
        return None

    # Calculate Inliers
    src_pts = np.float32(
        [fkp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kps[num][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, RANSAC_THRESH)
    matchesMask = mask.ravel().tolist()
    inliers = matchesMask.count(1)

    if len(matches) > MIN_RAW_MATCH:
        if(inliers > MIN_INLIER_MATCH):
            print(str(num) + "-\tInl Size: " + str(inliers) + " ✔")
            return M
        else:
            print(str(num) + "-\tInl Size: " + str(inliers))
    elif inliers > len(matches) * 0.65:
        print(str(num) + "-\tInl Ratio: " + str(inliers / len(matches)) + " ✔")
        return M

    return None


def writeFrame(matrix, num, frame):
    if matrix is None:
        return

    ret, alter = altVideos[num].read()
    alter = cv.cvtColor(alter, cv.COLOR_BGR2BGRA)
    alter = cv.rotate(alter, cv.ROTATE_90_CLOCKWISE)

    sampleH, sampleW = sample[num].shape
    res = cv.resize(alter, (sampleW, sampleH), interpolation=cv.INTER_CUBIC)

    h, w, d = frame.shape
    wrapped = cv.warpPerspective(res, Ms[num], (w, h))

    ret, res = cv.threshold(res, 0, 255, cv.THRESH_BINARY)
    res = cv.warpPerspective(res, Ms[num], (w, h))
    frameCpy = frame[::]

    y1, y2 = 0, wrapped.shape[0]
    x1, x2 = 0, wrapped.shape[1]

    alpha_s = wrapped[:, :, 3] / 255.0
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
    frameGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    fkp, fdes = sift.detectAndCompute(frameGray, None)

    for num in range(10):
        qualified = getMatches(des, num)
        M = getTransformationMatrix(qualified, num, fkp)

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
