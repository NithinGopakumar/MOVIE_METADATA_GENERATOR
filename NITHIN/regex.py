import pandas as pd
import re
from Jis_code import final_list
# data=open("datasheet1.txt")
# df=pd.DataFrame(data)
# print(df)
# print(final_list)

# clean = open('datasheet1_to_hocr.txt').read().replace('\n', ',')
clean = final_list


file_lst = ['cats1.fa', 'cats2.fa', 'dog1.fa', 'dog2.fa']
file_lst_trimmed =[]

clean1=clean.sub(",,", ",")
clean2=clean1.replace("»", ",")
clean3=clean2.replace("«", ",")
clean4=clean3.replace("+", ",")
clean5=clean4.replace("-", ",")
print(clean1)
print(clean2)
print(clean3)
print(clean4)
print(clean5)


# s = io.StringIO.io.StringIO(clean5)
# with open('fileName.csv', 'w') as f:
#     for line in s:
#         f.write(line)

