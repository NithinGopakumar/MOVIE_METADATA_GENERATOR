import cv2
import numpy as np
# Opens the Video file
import pytesseract


def sharpener(image):
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -3, kernel_sharpening)
    return sharpened


temp_text = ""
cap = cv2.VideoCapture('/home/nithing/Desktop/test30sec.mp4')
i = 0
while (cap.isOpened()):
    ret, image = cap.read()
    image = sharpener(image)
    image = cv2.bitwise_not(image)
    cv2.imshow("", image)
    cv2.waitKey(5)
    text = pytesseract.image_to_string(image)
    # print(text)
    if text == '':
        continue

    if text is not temp_text:
        temp_text = text
    if temp_text == '':
        continue

    f = open("datasheet.txt", "a+")
    f.write(temp_text + " ,")
    # if text == '':
    #     continue
    # else:
    #     temp_text=text
    # print(temp_text)

    # outF = open("myOutFile.txt", "a+")
    # outF.writelines(text)
    # outF.close()

    if ret == False:
        break
    if i % 10 == 0:
        cv2.imwrite('kang' + str(i) + '.jpg', image)
    i += 1
#
# outF = open("./file.txt", "w+")
# for line in temp_text:
#     print >>outF, line
# outF.close()



#
# def gen():
#     for i in range(text):
#         yield i
#
#
# print(i)
#
#
# def datacollect():
#     name = []
#     for i in range(len(name)):
#         if name[i] != data():
#             name.append(data())
#         else:
#             break
#     return name
#
#
# print(datacollect())

cap.release()
cv2.destroyAllWindows()
