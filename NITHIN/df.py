# # import pandas as pd
# # from final_ocr import temp_text
# #
# # print(temp_text)
#
#
# import cv2
# import pytesseract
#
# cap=cv2.imread('home/nithing/PycharmProjects/New/kang images/kang150.jpg')
# #  while (cap.isOpened()):
# #     ret, image = cap.read()
# #
# #     if ret == False:
# #         break
# #     if i % 24 == 0:
#         # image = cv2.bitwise_not(image)
# cv2.imshow("", cap)
# cv2.waitKey(20)
# text = pytesseract.image_to_string(cap)
# print(text)





import cv2
import pytesseract

f = open("data_1.txt", "a+")
f.truncate(0)
image=cv2.imread("/home/nithing/PycharmProjects/New/kang images/kang150.jpg")
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image = cv2.bitwise_not(gray)

# cv2.imshow("GrayImage",gray)
# cv2.waitKey(0)
# cv2.imshow("Grge",image)
# cv2.waitKey(0)
box=pytesseract.image_to_data(image)
dictionary=pytesseract.osd_to_dict(box)
f.write(box)

# # cv2.imshow("Gray",box)
# # cv2.waitKey(0)
# f.close()

import pandas as pd

# import the StrinIO function
# from io module
from io import StringIO

# wrap the string data in StringIO function
StringData = StringIO(box)

# let's read the data using the Pandas
# read_csv() function
df = pd.read_csv(StringData, sep="\t")
# Print the dataframe
print(df[["line_num","text",'level']])