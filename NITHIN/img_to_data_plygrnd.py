import pytesseract
import cv2

image=cv2.imread("/home/nithing/PycharmProjects/New1/images_another_movie/image740.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.bitwise_not(gray)
data = pytesseract.image_to_data(image)
text = pytesseract.image_to_string(image)

print(data)