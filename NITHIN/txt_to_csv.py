# import random
#
# amnt = int(raw_input('Please enter the amount of numbers you would like:'))
# data = (random.randint(0,100000000) for _ in range(amnt))
# data = (str(datum) for datum in data)
# data = ','.join(data) + '\n'
# with open("random.csv", "w") as fp:
#     fp.write(data)



import csv

f=open("datasheet1.txt")
x=f.readlines()
s=[]


for i in x:
    i=i.replace("\n",'')
    j=i.replace(",,",",")
    k=j.replace(" ","")
    s.append(j)

csvex=csv.writer(open("to_csv","w"), delimiter=",", quoting=csv.QUOTE_ALL)
csvex.writerow(s)
