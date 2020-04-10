import cv2
import os
import shutil
myfile=r"C:\Users\nithi\PycharmProjects\NEW_latest\photos"
foldercheck=os.path.exists(r"C:\Users\nithi\PycharmProjects\NEW_latest\photos")
if foldercheck==True:
    for root, dirs, files in os.walk(myfile):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
    shutil.rmtree(myfile)
    print("No file detected")

# else:
#     folder=os.mkdir("/home/nithing/PycharmProjects/New1/photos")
#     print("yo")
#     os.remove("/home/nithing/PycharmProjects/New1/photos")
#     break
#  if foldercheck==False:
folder=os.mkdir(r"C:\Users\nithi\PycharmProjects\NEW_latest\photos")
#      os.remove("/photos/*")
# shutil.rmtree("/home/nithing/PycharmProjects/New1/photos")
# print(foldercheck)
# path=os.getcwd()
# print(path)
list=os.listdir(r"C:\Users\nithi\PycharmProjects\NEW_latest\photos")
print(list)

# folder=os.mkdir("photos")

cap=cv2.VideoCapture(r'C:\Users\nithi\PycharmProjects\NEW_latest\Vids\age of.mp4')
i=0
while (cap.isOpened()):
    ret,frame=cap.read()
    if ret == False:
        break
    if i%1==0:
        cv2.imwrite('photos/images'+str(i)+'.jpg',frame)
    i+=1

cap.release()
cv2.destroyAllWindows()