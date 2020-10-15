import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

MIN_MATCH_COUNT = 10

cap = cv.VideoCapture('Data/Test/0.MOV')

frame = None
if(cap.isOpened()):
    ret, frame = cap.read()

    cv.namedWindow("Frame", cv.WINDOW_NORMAL)
    cv.imshow('Frame', frame)
    cv.resizeWindow("Frame", 960, 540)
    cv.waitKey(0)

cap.release()

source = cv.imread("Data/Source/1.jpg")
rep = cv.imread("Data/Source/5.jpg")
rep = cv.cvtColor(rep, cv.COLOR_BGR2BGRA)


sift = cv.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(source,None)
kp2, des2 = sift.detectAndCompute(frame,None)
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
# Need to draw only good matches, so create a mask
# matchesMask = [[0,0] for i in range(len(matches))]
# ratio test as per Lowe's paper
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    print(matchesMask.count(1))
    h,w,d = source.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    img2 = cv.polylines(frame,[np.int32(dst)],True,255,3, cv.LINE_AA)


    res = cv.resize(rep,(w, h), interpolation = cv.INTER_CUBIC)
    h,w,d = frame.shape
    wrapped = cv.warpPerspective(res, M, (w, h))
    cv.namedWindow("Frame2", cv.WINDOW_NORMAL)
    cv.imshow('Frame2', wrapped)
    cv.resizeWindow("Frame2", 960, 540)
    cv.waitKey(0)

    ret, res = cv.threshold(res,0,255,cv.THRESH_BINARY)
    res = cv.warpPerspective(res, M, (w, h))


    frameCpy = frame[::]

    y1, y2 = 0, wrapped.shape[0]
    x1, x2 = 0, wrapped.shape[1]

    alpha_s = wrapped[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        frameCpy[y1:y2, x1:x2, c] = (alpha_s * wrapped[:, :, c] +
                                alpha_l * frameCpy[y1:y2, x1:x2, c])
    
    cv.namedWindow("Frame3", cv.WINDOW_NORMAL)
    cv.imshow('Frame3', frameCpy)
    cv.resizeWindow("Frame3", 960, 540)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
img3 = cv.drawMatches(source,kp1,frame,kp2,good,None,**draw_params)
plt.imshow(img3, 'gray'),plt.show()