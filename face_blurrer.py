import cv2 as cv
import matplotlib.pyplot as plt

global FACE_CASCADE
FACE_CASCADE = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

def blur_faces(img):
    ''' takes image as input, returns same image with faces blurred '''
    # convert to gray scale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # get faces
    faces = FACE_CASCADE.detectMultiScale(gray, 1.01, 4)
    for (x, y, h, w) in faces:
        crop = img[y:y+w, x:x+h, :]
        img[y:y+w, x:x+h, :] = cv.blur(crop, (15,15))
    return img

# img = cv.imread('images/ferguson.jpeg')
img = cv.imread('images/me.jpg')
blur_img = blur_faces(img)
plt.imshow(img)
plt.show()