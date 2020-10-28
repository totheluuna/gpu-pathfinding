import cv2 as cv
import sys
img = cv.imread(cv.samples.findFile("dataset/da2-png/ht_keep.png"))
if img is None:
    sys.exit("Could not read the image.")
# cv.imshow("Display window", img)
# k = cv.waitKey(0)
# if k == ord("s"):

print('Original Dimensions : ',img.shape)
 
scale_percent = 10 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
resized = cv.resize(img, dim, interpolation = cv.INTER_AREA) 
print('Resized Dimensions : ',resized.shape) 

# convert to grayscale
gray_image = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
print('Grayscale Dimensions : ',gray_image.shape)

hist = cv.calcHist([gray_image],[0],None,[256],[0,256])
# for i in range(len(hist)):
#     if hist[i][0] > 0:
#         print(i)

ret,thresh1 = cv.threshold(gray_image,127,255,cv.THRESH_BINARY)
# print(thresh1)

walls = []
for i in range(thresh1.shape[0]):
    for j in range(thresh1.shape[1]):
        # if the current value = 0 (meaning black) append to list of walls
        if thresh1[i][j] == 0:
            walls.append((i,j))
# print(walls)

def checkWalls(walls):
    return len(walls) < thresh1.shape[0]*thresh1.shape[1]

print("All walls? ", checkWalls(walls))
cv.imwrite("test.png", thresh1)