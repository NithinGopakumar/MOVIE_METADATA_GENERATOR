import pytesseract
import cv2
import os
# import folder_cr

# myfile="/home/nithing/PycharmProjects/New1/photos"
# list=os.listdir("/home/nithing/PycharmProjects/New1/photos")
# print(list)
list1=["/home/nithing/PycharmProjects/New1/photos/images37.jpg" , "/home/nithing/PycharmProjects/New1/photos/images45.jpg" , "/home/nithing/PycharmProjects/New1/photos/images151.jpg"]
print(list1)
for image in range(len(list1)):
    # ret, image = cap.read()
    image = cv2.bitwise_not(image)
    cv2.imshow("", image)
    cv2.waitKey(20)
    text = pytesseract.image_to_data(image)

    print(text)
    # print(list(values))
image=image+1