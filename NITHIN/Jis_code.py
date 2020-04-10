# ------------Gray and level 5 words only---------------


import cv2
import pytesseract
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

from pathlib import Path

# pathlist = Path("/home/hashimabdulla/PycharmProjects/OCR/images/image640.jpg").glob('**/*.jpg')

image = cv2.imread(r"C:\Users\nithi\PycharmProjects\NEW_latest\photos\images113.jpg")
# image = cv2.imread("./images/image420.jpg")
y,x,_=image.shape

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.bitwise_not(gray)
data = pytesseract.image_to_data(image)
text = pytesseract.image_to_string(image)
# for i in range(x):
    # for j in range(y):
        # b,g,r=img[j][i]
        # if b>150 and g>150 and r>150:
            # img[j][i]=255
        # else:
            # img[j][i]=0
cv2.imshow("",image)
cv2.waitKey()
custom_config = r'--oem 3 --psm 12'
text= pytesseract.image_to_string(image,config=custom_config)
# print(text)
d = pytesseract.image_to_data(image,config=custom_config, output_type=Output.DICT)
n_boxes = len(d['level'])
# print(n_boxes)
print(d['text'])
tmp_list=[]
list_sent=[]
final_list=[]
tmp_count=0

for i in range(n_boxes):
    box=(d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    # print(box)
    if box in tmp_list:
        tmp_count=tmp_count+1
    else:
        tmp_list=[]
        tmp_count=0
        tmp_list.append(box)

    if tmp_count>0 and box not in list_sent:
        # print("appended")
        list_sent.append(box)
    if d['level'][i]==5 and d['text'][i]!=None:
        final_list.append(d["text"][i])
        print(final_list)

print(len(final_list))
# using list comprehension
final_STR = ' '.join([str(elem) for elem in final_list])

print(final_STR)
print(len(final_STR))
# print(text)

print(len(list_sent))
for i in range(len(list_sent)):
    (x, y, w, h) = list_sent[i]
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow("",image)
cv2.waitKey()