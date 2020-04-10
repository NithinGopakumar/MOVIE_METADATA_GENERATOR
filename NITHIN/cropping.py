# -------------Crop Text Area--------------

import pytesseract
from pytesseract import Output
import cv2
from pathlib import Path

# pathlist = Path("/home/hashimabdulla/PycharmProjects/OCR/images/image640.jpg").glob('**/*.jpg')

# image = cv2.imread("./Kang/magpool60.jpg")
# image = cv2.imread("./images/image420.jpg")
from scaled_cropping import ScaledCropper

org_image = cv2.imread(r"C:\Users\nithi\PycharmProjects\NEW_latest\photos\images113.jpg")
bound_image=org_image.copy()
y, x, _ = org_image.shape

gray = cv2.cvtColor(org_image, cv2.COLOR_BGR2GRAY)
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
# cv2.imshow("", image)
# cv2.waitKey()
custom_config = r'--oem 3 --psm 12 --alpha off'
text = pytesseract.image_to_string(image, config=custom_config)
# print(text)
d = pytesseract.image_to_data(image, config=custom_config, output_type=Output.DICT)
n_boxes = len(d['level'])
# print(n_boxes)
# print(d['level'])
tmp_list = []
list_sent = []

tmp_count = 0

for i in range(n_boxes):
    box = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    # print(box)
    if box in tmp_list:
        tmp_count = tmp_count + 1
    else:
        tmp_list = []
        tmp_count = 0
        tmp_list.append(box)

    if tmp_count > 0 and box not in list_sent:
        # print("appended")
        list_sent.append(box)
    if d['level'][i] == 5 and d['text'][i] != None:
        print(d["text"][i])

# print(list_sent)
for i in range(len(list_sent)):
    (x, y, w, h) = list_sent[i]
    rec=cv2.rectangle(bound_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    box=[x-10, y,x+w+10,y,x + w+10, y + h,x-10, y + h]
    croppe=ScaledCropper()
    cro_image=croppe.crop(image,[box])

    custom_config1 = r'--oem 3 --psm 7 --alpha off'

    print("*******************")
    print(pytesseract.image_to_string(cro_image,config=custom_config1))
    print("*******************")
    cv2.imshow("cropped", cro_image)
    cv2.imwrite("Cropped_result/croped_{}.jpg".format(i),cro_image)
    cv2.waitKey(1)

cv2.imshow("", bound_image)
cv2.waitKey()
cv2.imwrite("bounding_box_image2.jpg",bound_image)