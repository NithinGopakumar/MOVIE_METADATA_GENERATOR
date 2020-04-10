# ------------Bounding box-----------


import pytesseract
from pytesseract import Output
import cv2
from pathlib import Path

# pathlist = Path("/home/hashimabdulla/PycharmProjects/OCR/images/image640.jpg").glob('**/*.jpg')

img = cv2.imread("/home/nithing/PycharmProjects/New1/images_another_movie/image520.jpg")
y,x,_=img.shape
# for i in range(x):
# for j in range(y):
# b,g,r=img[j][i]
# if b>150 and g>150 and r>150:
# img[j][i]=255
# else:
# img[j][i]=0
cv2.imshow("",img)
cv2.waitKey()
custom_config = r'--oem 3 --psm 12 --alpha off'
text= pytesseract.image_to_string(img,config=custom_config)
print(text)
d = pytesseract.image_to_data(img,config=custom_config, output_type=Output.DICT)
n_boxes = len(d['level'])
tmp_list=[]
list_sent=[]
tmp_count=0
for i in range(n_boxes):
    box=(d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    print(box)
    if box in tmp_list:
        tmp_count=tmp_count+1
    else:
        tmp_list=[]
        tmp_count=0
        tmp_list.append(box)

    if tmp_count>0 and box not in list_sent:
        print("appended")
        list_sent.append(box)

print(len(list_sent))
for i in range(len(list_sent)):
    (x, y, w, h) = list_sent[i]
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow("",img)
cv2.imwrite("result/bounding_box.jpg",img)
cv2.waitKey()