import cv2
import numpy as np
import cv2 as cv
sift = cv.xfeatures2d_SIFT.create(1000)
# sift function using n=1000 to get the most accurate matching points and make the process faster
cv.namedWindow('main1',cv.WINDOW_NORMAL)  # using window normal to fit the image to the window

img0 = cv.imread('yard0.png', cv.IMREAD_GRAYSCALE)#grayscaling the image

x_small = int(img0.shape[1] ) #original size of the images
y_small =  int(img0.shape[0] )#
print(x_small,y_small)
x_final = x_small * 5 # final dimensions(designed for max 5 images)(5 times width)
y_final = y_small * 3 # 3 times height
Center = np.array([[1, 0, x_small * 2], [0, 1, y_small]], dtype=np.float32)  # array that brings the image to the center
img0 = cv.warpAffine(img0, Center, (x_final, y_final))  # moving image to the center
cv.imshow('main1', img0)
cv.waitKey(0)
kp0 = sift.detect(img0)
desc0 = sift.compute(img0, kp0)

img1 = cv.imread('yard1.png', cv.IMREAD_GRAYSCALE)
img1 = cv.warpAffine(img1, Center, (x_final, y_final))
cv.namedWindow('main2',cv.WINDOW_NORMAL)
cv.imshow('main2', img1)
cv.waitKey(0)
kp1 = sift.detect(img1)
desc1 = sift.compute(img1, kp1)

img2 = cv.imread('yard2.png', cv.IMREAD_GRAYSCALE)
img2 = cv.warpAffine(img2, Center, (x_final, y_final))
cv.namedWindow('main3',cv.WINDOW_NORMAL)
cv.imshow('main3', img2)
cv.waitKey(0)

kp2 = sift.detect(img2)
desc2 = sift.compute(img2, kp2)

img3 = cv.imread('yard3.png', cv.IMREAD_GRAYSCALE)
img3 = cv.warpAffine(img3, Center, (x_final, y_final))
cv.namedWindow('main4',cv.WINDOW_NORMAL)
cv.imshow('main4', img3)
cv.waitKey(0)
kp3 = sift.detect(img3)
desc3 = sift.compute(img3, kp3)


def match2(d0, d1):
    n0 = d0.shape[0]
    n1 = d1.shape[0]

    matches = []
    matches1 = []
    matches2 = []
    for i in range(n1):
        fv = d1[i, :]
        diff = d0 - fv

        diff = np.abs(diff)
        distances = np.sum(diff, axis=1)

        i2 = np.argmin(distances)
        mindist2 = distances[i2]

        distances[i2] = np.inf  # infinity

        i3 = np.argmin(distances)
        mindist3 = distances[i3]

        if mindist2 / mindist3 < 0.5:
            x = cv.DMatch(i, i2, mindist2)
            matches1.append(x)
    for k in range(n0):
        lv = d0[k, :]
        diff2 = d1 - lv

        diff2 = np.abs(diff2)
        distances2 = np.sum(diff2, axis=1)

        k2 = np.argmin(distances2)
        mindist22 = distances2[k2]

        distances2[k2] = np.inf  # infinity

        k3 = np.argmin(distances2)
        mindist33 = distances2[k3]

        if mindist22 / mindist33 < 0.5:
            y = cv.DMatch(k, k2, mindist22)
            matches2.append(cv.DMatch(k, k2, mindist22))
            for j in range(len(matches1)):
                if y.trainIdx == matches1[j].queryIdx :

                    matches.append(matches1[j])#take the second image’s array point
                    print(matches1[j].queryIdx,matches1[j].trainIdx)
                    break

    return matches




matches = match2(desc0[1], desc1[1])

dimg = cv.drawMatches(img1, desc1[0], img0, desc0[0], matches, None)#draw lines between the common points
cv.namedWindow('main5', cv.WINDOW_NORMAL)
cv.imshow('main5', dimg)
cv.waitKey(0)
img_pt1 = []
img_pt2 = []
for x in matches:
    img_pt1.append(kp0[x.trainIdx].pt)#right image
    img_pt2.append(kp1[x.queryIdx].pt)#left image
img_pt1 = np.array(img_pt1)
img_pt2 = np.array(img_pt2)

M, mask = cv.findHomography(img_pt1, img_pt2, cv.RANSAC)
# find how to convert the first parameter to fit the second

img4 = cv.warpPerspective(img0, M, (img0.shape[1], img0.shape[0]))#transform the image img using the M and put it in img4
img4[:,x_small*2:x_small*3]= img1[:, x_small * 2: x_small * 3]#put img1 in img4 as well

cv.namedWindow('main',cv.WINDOW_NORMAL)
cv.imshow('main', img4)
cv.waitKey(0)

kp4 = sift.detect(img4)
desc4 = sift.compute(img4, kp4)
matches2=match2(desc4[1], desc2[1])

ding = cv.drawMatches(img2, desc2[0], img4, desc4[0], matches2, None)
cv.namedWindow('main5', cv.WINDOW_NORMAL)
cv.waitKey(0)
img_pt1 = []
img_pt2 = []
for x in matches2:
    img_pt1.append(kp4[x.trainIdx].pt)#right
    img_pt2.append(kp2[x.queryIdx].pt)#left
img_pt1 = np.array(img_pt1)
img_pt2 = np.array(img_pt2)

M, mask = cv.findHomography(img_pt1, img_pt2, cv.RANSAC)
img5 = cv.warpPerspective(img4, M, (img4.shape[1], img4.shape[0]))
img5[:, x_small * 2: x_small * 3]= img2[:, x_small * 2: x_small *3]

cv.namedWindow('main',cv.WINDOW_NORMAL)
cv.imshow('main', img5)
cv.waitKey(0)

kp5 = sift.detect(img5)
desc5 = sift.compute(img5, kp5)
matches3= match2(desc5[1], desc3[1])

dilg = cv.drawMatches(img3, desc3[0], img5, desc5[0], matches3, None)
cv.namedWindow('main5', cv.WINDOW_NORMAL)  # για να χωραει στο παραθυρο
cv.imshow('main5', dilg)
cv.waitKey(0)
img_pt1 = []
img_pt2 = []
for x in matches3:
    img_pt1.append(kp5[x.trainIdx].pt)#right
    img_pt2.append(kp3[x.queryIdx].pt)#left
img_pt1 = np.array(img_pt1)
img_pt2 = np.array(img_pt2)

M, mask = cv.findHomography(img_pt2, img_pt1, cv.RANSAC)
#finds how to convert the left image to fit the right

newimg = cv.warpPerspective(img3, M, (img5.shape[1], img5.shape[0]))#transform img3 and put it in newing
img5[:, 0: x_small * 2] = newimg[:, 0: x_small * 2]#put newing to img5(which already has img5 as it is)

cv.namedWindow('main',cv.WINDOW_NORMAL)
cv.imshow('main', img5)
cv.waitKey(0)

cv2.imwrite("b2.jpg", img5)
pass
