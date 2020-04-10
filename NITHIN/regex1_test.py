# import re
#
# file_lst = ['cats1.fa', 'cats2.fa', 'dog1.fa', 'dog2.fa']
# file_lst_trimmed =[]
#
# for file in file_lst:
#     file_lst_trimmed = re.sub(r'1.fa', '', file)
#
# print(file_lst_trimmed)


# Python program to convert a list to string

# Function to convert

s = ['I', 'want', 4, 'apples', 'and', 18, 'bananas']

# using list comprehension
listToStr = ' '.join([str(elem) for elem in s])

print(listToStr)
