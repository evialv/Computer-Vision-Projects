
import cv2
import numpy as np

filename = 'N6.png'
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
img_rgb = cv2.imread(filename, 0)
height, width = img_rgb.shape
print(img.shape) #Τυπώνει τις διαστάσεις της εικόνας

members = [(0, 0)] * 9
newimg = np.copy(img)

for i in range(1, img.shape[0]-1):
    for j in range(1, img.shape[1]- 1):
        members[0] = img[i - 1, j - 1]
        members[1] = img[i - 1, j ]
        members[2] = img[i - 1, j + 1]
        members[3] = img[i , j - 1]
        members[4] = img[i , j]
        members[5] = img[i , j + 1]
        members[6] = img[i + 1, j - 1]
        members[7] = img[i + 1, j ]
        members[8] = img[i + 1, j + 1]
        members.sort()
        newimg[i,j]=members[4]

cv2.imwrite("b1.png", newimg)
cv2.namedWindow('open', )
cv2.imshow('open', newimg)
cv2.waitKey(0)

kernel_test_3 = np.ones((3,3),np.uint8)
# OPENING
open = cv2.morphologyEx(newimg, cv2.MORPH_OPEN, kernel_test_3)
cv2.namedWindow('open', )
cv2.imshow('open', open)
cv2.waitKey(0)


# CLOSING
close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, kernel_test_3)
cv2.namedWindow('close', )
cv2.imshow('close', close)
cv2.waitKey(0)
#calculating integral image before thresholding
integral_image = cv2.integral(close)

#threshold
returnValue,thresholded_img = cv2.threshold(close,48,255,cv2.THRESH_BINARY )
cv2.namedWindow('close', )
cv2.imshow('close', thresholded_img)
cv2.waitKey(0)
# EROSION κανω erosion
kernel_test_9 = np.ones((9,9), np.uint8)
erode = cv2.morphologyEx(thresholded_img, cv2.MORPH_ERODE, kernel_test_9)

open = cv2.morphologyEx(erode, cv2.MORPH_OPEN, kernel_test_9)
cv2.namedWindow('erosion', )
cv2.imshow('erosion', open)
cv2.waitKey(0)
#find contours
image, contours, hierarchy = cv2.findContours(open,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# ---COUNT CELLS IN PICTURE---

delete = []
noncountingcells = 0
cells = hierarchy.shape[1]
flag = 0
contoursized = []
(x, y) = open.shape
blackimage = np.zeros(image.shape, np.uint8)

for i in range(cells):
    mask = np.zeros(image.shape, np.uint8)
    flag = 0
    imgnew=blackimage.copy()
    initing=close.copy()
    cv2.drawContours(imgnew, contours,i,( 255, 255,255), -1)
    dilate = cv2.morphologyEx(imgnew, cv2.MORPH_DILATE, kernel_test_9)
    image2, contours2, hierarchy2 = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contoursized.append(contours2[0])
    mask = np.zeros(dilate.shape, np.uint8)
    cv2.drawContours(mask, [contours2[0]], 0, 255, -1)
    pixelpoints = np.transpose(np.nonzero(mask)) #βολευει στην προσπελαση των στοιχειων μου
    for j in range(len(pixelpoints)):
        if((pixelpoints[j][0] <= 2 or pixelpoints[j][1] <= 2 or pixelpoints[j][0] >= x-2 or pixelpoints[j][1] >= y-2) and flag==0):

            flag = 1
            delete.append(i)
            noncountingcells = noncountingcells + 1
cellsΙn = cells - noncountingcells
print(cellsΙn, " cells in the boundary")



k = 0   #κυτταρα που απορριπτω
incl = 0   #κυτταρα που μετραω
areas = []  #εμβαδον κυτταρων
meanvalue = [] #τιμη διαβαθμισης γκρι
print(noncountingcells)
for j in range(cells):   # Για κάθε κύτταρο
    if j != delete[k]:     # Αν δεν το εχω απορριψει
        areas.append(round(cv2.contourArea(contoursized[j])))     # μετρα επιφανεια μεγαλωμενου
        print('Area of cell', incl + 1, 'is', areas[incl], 'pixels')
        [x, y, w, h] = cv2.boundingRect(contoursized[j])
        gray = integral_image[(y + h), (x + w)] - integral_image[(y + h), x] - integral_image[y, (x + w)] + integral_image[(y - 1), (x - 1)]
        meanvalue.append(gray / (h * w))   # πεταω την τιμη στο meanvalue
        print('Mean grayscale value of box of cell', incl+ 1, 'is:', meanvalue[incl])
        incl = incl + 1
        begining= close.copy()
        final = close.copy()
        cv2.drawContours(begining, contours, j, (255, 255, 255), -1)
        cv2.drawContours(final, contoursized, j, (255, 255, 255), -1)

        cv2.imshow('contour', begining)
        cv2.waitKey(0)
        cv2.imshow('contour', final)
        cv2.waitKey(0)

    else:   # Αν πρόκειται για κύτταρο που έχει εξαιρεθεί
        k = k + 1   # Θα ψάξουμε το επόμενο εξαιρεθέν κύτταρο
        print(k)
        if k>= noncountingcells:   # Αν ξεπεραστεί ο αριθμός των κυττάρων αυτών, να μην βγει το delete εκτος οριων αφου ξεκιναω το αρραυ απο το 0 αλλα τα κ φτανουν μεχρι το 8 και το δελετε ειναι 0-7
            k = 0

cv2.destroyAllWindows()