# import cv2
# from matplotlib import pyplot as plt
#
# # opens image
# img = cv2.imread("faces/people1.png")
#
# # converting images to black and white format
# # leaving one more as rgb
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# # loading the haar cascade data into program
# face_data = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt2.xml")
#
# #result extraction
# faces = face_data.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)
# print(faces)
#
# # outlining
# for (x, y, width, height) in faces:
#     cv2.circle(img_rgb, (x + (width//2), y+ (height // 2), width //2, (0,255,0),5))
#
#
# plt.subplot(1,1,1)
# plt.imshow(img_rgb)
# plt.show()


import cv2
from matplotlib import pyplot as plt

# opens image
img = cv2.imread("faces/people3.jpg")

# converting images to black and white format
# leaving one more as rgb
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# loading the haar cascade data into program
face_data = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt2.xml")

# result extraction
faces = face_data.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)
print(faces)

# outlining
for (x, y, width, height) in faces:
    cv2.circle(img_rgb, (x + (width//2), y + (height//2)), width//2, (0,255,0), 5)


plt.subplot(1,1,1)
plt.imshow(img_rgb)
plt.show()
