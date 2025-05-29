import cv2
from matplotlib import pyplot as plt

img = cv2.imread("roads/road3.jpg")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

stopSign_data = cv2.CascadeClassifier("haarcascades/haarcascade_stopsign.xml")

stopSigns = stopSign_data.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)
print(stopSigns)
print(len(stopSigns))
if len(stopSigns) > 0:
    print("True")
else:
    print("False")

for (x, y, width, height) in stopSigns:
    cv2.circle(img_rgb, (x + (width//2), y + (height//2)), width//2, (0,255,0), 5)


plt.subplot(1,1,1)
plt.imshow(img_rgb)
plt.show()
