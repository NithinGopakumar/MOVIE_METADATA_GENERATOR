# -----------JIS'S CODE 1-------------------
import pytesseract
from pytesseract import Output
import cv2
from pathlib import Path

pathlist = Path("/home/nithing/PycharmProjects/New1/images_another_movie").glob('**/*.jpg')
count=0
for path in pathlist:

    img = cv2.imread(str(path))
    custom_config = r'--oem 3 --psm 12 --alpha off'
    d = pytesseract.image_to_data(img,config=custom_config,output_type=Output.DICT)
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

        if tmp_count>1 and box not in list_sent:
            print("appended")
            list_sent.append(box)

    print(list_sent)
    for i in range(len(list_sent)):
        (x, y, w, h) = list_sent[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    count=count+1

    cv2.imwrite("/home/nithing/PycharmProjects/New1/result/{}.jpg".format(count), img)