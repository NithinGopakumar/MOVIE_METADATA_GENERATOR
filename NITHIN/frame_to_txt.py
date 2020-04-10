from io import StringIO
import pytesseract
import cv2
import os
import pandas as pd
import re
import json
# f=open("")

f=open("dataset.txt","a+")
f.truncate(0)
# Get current location
path=os.getcwd()
# folder name
folder="images"
# create a new folder named images in path
imagefolder=path+"/"+folder
# imagefolder1=path+"/Kang"
# list is a list of .jpg files in imagefolder
list=os.listdir(imagefolder)
# perform tesseract in each images in the list and save the data in a frame.txt file
a={}
l=[]
m=[]
for i in range(len(list)):
    img=list[i]
    image=cv2.imread(imagefolder+"/"+img)

#   print(image)
#   print(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.bitwise_not(gray)
    data=pytesseract.image_to_data(image)
    text=pytesseract.image_to_string(image)
    # b = {img: text}
    # a.update(b)
    m.append(text)
    # cv2.imshow("image",image)
    # cv2.waitKey()
    # print(text)
    # StringData = StringIO()
    # text=text.split("\n")
    b={img:m}
    a.update(b)
    # text_df = df[df.text.notnull()]
    abc_list=l.append(text)
    f.write(text+",")
# f=f("\n\n","',")
# z=l.split()?
# print(z)
# dtst=pd.read_csv(f,header=None, delim_whitespace=True)
# df1=pd.DataFrame(dtst,index=[0])
# dtst.to_csv("dtst.csv",index=None)
df=pd.DataFrame(a)
df.to_csv('text.csv')

print(m)
print("Length of a:",len(m))
print(a)
print("Length of a:",len(a))
# print(df)
print(text)

print(l)
print("length of l:",len(l))
# print(df1)
    # x=list(text_df.columns)
    # print(x)
    # line=df.head(8)
    # print(line)
    # print(text_df)
# a=StringIO(a)

f.close()
# with open(df)as json_file:
#     dataset=json.load(json_file)
#     print(dataset)
# print(a)
# print("lenght of dict : ",len(a.keys()))
# print("lenght of list : ",len(list))
# print(a.keys())
# line = df.head(8)