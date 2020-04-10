# import os
# import glob
#
# files = glob.glob('/home/nithing/PycharmProjects/New1/photos')
# for f in files:
#     os.remove(f)


import os
import shutil

myfile="/home/nithing/PycharmProjects/New1/photos"
for root, dirs, files in os.walk(myfile):
    for f in files:
        os.unlink(os.path.join(root, f))
    for d in dirs:
        shutil.rmtree(os.path.join(root, d))

shutil.rmtree(myfile)

        # if os.path.isfile(myfile):
        #     shutil.rmtree(myfile)
        # else:    ## Show an error ##
        #     print("Error: %s file not found" % myfile)
