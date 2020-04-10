# traint_data=[]
#
# def create_data(cmindx,length,sent,label):
#     sentence=(sent, {"entities": [(0, cmindx, label), (cmindx+1, length, "PERSON")]})
#     traint_data.append(sentence)
#
#
#
# filepath = '/home/nithing/datasetcr1'
# with open(filepath) as fp:
#     for cnt, line in enumerate(fp):
#         LABEL = "DESIGNATION"
#         text=line.strip('\n')
#         # print(text)
#         cmindx = text.index(',')
#         length = len(text)
#         create_data(cmindx, length, text, LABEL)
#
# print(traint_data)




traint_data=[]

def create_data(cmindx,length,sent,label):
    print(sent)
    sentence=(sent, {"entities": [(0, cmindx,"PERSON" ),(cmindx+1, length, label)]})
    traint_data.append(sentence)



filepath = "/home/nithing/datasetcr1"

with open(filepath,'r') as fp:
    for cnt, line in enumerate(fp):
        LABEL = "DESIGNATION"
        sent=line.strip('\n')
        # print(text)
        length = len(sent)

        cmindx = sent.index(",")
        sent = sent[cmindx + 1:length] + ',' + sent[0:cmindx]
        cmindx = sent.index(",")
        length = len(sent)
        create_data(cmindx, length, sent, LABEL)

print(traint_data)