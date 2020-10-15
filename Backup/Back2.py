import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

MIN_MATCH_COUNT = 70
MIN_GOOOD = 40

cap = cv.VideoCapture('Data/Test/9.MOV')
rep = [None] * 11
for i in range(10):
    rep[i] = cv.VideoCapture('Data/Source/' + str(i) + '.mp4')

writer = cv.VideoWriter('Test.mp4', -1, 5, (960, 540))

FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH = 6
# index_params = dict(algorithm=FLANN_INDEX_LSH,
#                     table_number=6,  # 12
#                     key_size=12,     # 20
#                     multi_probe_level=1)  # 2
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict()
flann = cv.FlannBasedMatcher(index_params, search_params)
flann.clear()

sift = cv.xfeatures2d.SIFT_create()
kps = []
deses = []
sources = []
for num in range(10):
    src = cv.imread("Data/Source/" + str(num) + ".jpg")
    src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    kpt, des1 = sift.detectAndCompute(src, None)
    kps.append(kpt)
    deses.append(des1)
    sources.append(src)

frame = None
i = 0
while(cap.isOpened()):
    print(i, flush=True)
    i += 1
    ret, frame = cap.read()

    frame = cv.resize(frame, (960, 540), interpolation=cv.INTER_CUBIC)

    if(ret == False or i > 1000):
        break
    frameGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    kp2, des2 = sift.detectAndCompute(frameGray, None)

    for num in range(10):
        matches = flann.knnMatch(queryDescriptors=des2,
                                 trainDescriptors=deses[num],
                                 k=2)

        good = []
        try:
            for pair in matches:
                if len(pair) == 2:
                    m = pair[0]
                    n = pair[1]
                    if m != None and n != None:
                        if m.distance < 0.7*n.distance:
                            good.append(m)
                elif len(pair) == 1:
                    good.append(pair[0])
            print(len(good))
        except:
            print("Error")

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32(
                [kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kps[num][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 8.0)
            matchesMask = mask.ravel().tolist()
            soGood = matchesMask.count(1)
            print("M", soGood)
            if(soGood > MIN_GOOOD):
                ret2, repf = rep[num].read()

                repf = cv.cvtColor(repf, cv.COLOR_BGR2BGRA)
                repf = cv.rotate(repf, cv.ROTATE_90_CLOCKWISE)
                h, w = sources[num].shape

                res = cv.resize(repf, (w, h), interpolation=cv.INTER_CUBIC)
                h, w, d = frame.shape
                wrapped = cv.warpPerspective(res, M, (w, h))

                ret, res = cv.threshold(res, 0, 255, cv.THRESH_BINARY)
                res = cv.warpPerspective(res, M, (w, h))

                frameCpy = frame[::]

                y1, y2 = 0, wrapped.shape[0]
                x1, x2 = 0, wrapped.shape[1]

                alpha_s = wrapped[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s

                for c in range(0, 3):
                    frameCpy[y1:y2, x1:x2, c] = (alpha_s * wrapped[:, :, c] +
                                                 alpha_l * frameCpy[y1:y2, x1:x2, c])

                frameCpy = cv.resize(frameCpy, (960, 540),
                                     interpolation=cv.INTER_CUBIC)

                frame = frameCpy[::]
    writer.write(frame)
    cv.namedWindow("Frame", cv.WINDOW_NORMAL)
    cv.imshow('Frame', frame)
    cv.resizeWindow("Frame", 960, 540)
    cv.waitKey(1)
    print("--------")
cv.destroyAllWindows()
writer.release()
cap.release()
