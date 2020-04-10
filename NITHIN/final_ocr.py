import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


cap = cv2.VideoCapture(r'C:\Users\nithi\PycharmProjects\NEW_latest\fallen1.mp4')
i = 0
f = open("datasheet1_to_hocr.txt", "a+")
f.truncate(0)
temp_text = ""
while (cap.isOpened()):
    ret, image = cap.read()

    if ret == False:
        break
    if i % 60 == 0:
        image = cv2.bitwise_not(image)
        cv2.imshow("", image)
        cv2.waitKey(20)
        text = pytesseract.image_to_string(image)
        print(text)
        if temp_text==text or text=="\n" or text=="" or text=="\n\n":
            continue
        else:
            f.write(text + " ,")
            temp_text = text

    i += 1
f.close()

cap.release()
cv2.destroyAllWindows()
