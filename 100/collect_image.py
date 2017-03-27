import matplotlib.image as imglib
import numpy as np
import os  

pwd = os.getcwd()

acc = np.empty([300, 240, 320, 4])
index = 0
for fn in os.listdir(pwd + '/photo'):
    # img = imglib.imread(pwd + 'photo/' + fn)
     # if os.path.isfile(fn):
    photonum = os.path.splitext(fn)[0]
    img = imglib.imread(pwd + '/photo/' + photonum + ".jpg")
    depth = imglib.imread(pwd + '/depth/' + photonum + ".png")
    acc[index] = np.dstack((img,depth))
    index += 1

np.save('merged_100.npy', acc)