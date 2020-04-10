# ----------video_to_frames---------------

import cv2
import os
import shutil
# Opens the Video file
pathaddress=os.getcwd()
folderName="images_another_movie"
newfolder=pathaddress+"/"+folderName
foldercheck=os.path.exists(newfolder)
if foldercheck==False:
    os.mkdir(folderName)
else:
    print("folder already exist")
    shutil.rmtree(newfolder)
    os.mkdir(folderName)

print(newfolder)
print(pathaddress)
print(foldercheck)
cap = cv2.VideoCapture('/home/nithing/Documents/wrester.mp4')
i = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    if i%20==0:
        cv2.imwrite(folderName+'/image' + str(i) + '.jpg', frame)
    i+= 1
cap.release()
cv2.destroyAllWindows()