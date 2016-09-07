import numpy as np

x_path = '/home/ben/dev/AgePredictor/faces/data/'
y_path = '/home/ben/dev/AgePredictor/faces/labels/'
rand = np.random.choice(10000, 8000, replace=False)
count = np.arange(10000)
remainder = [x for x in count if x not in rand]
temp = []
tempy = []
for i in range(0,10):
    x = np.load(x_path+'face_' + str(i) + '.npy')
    y = np.load(y_path+'face_labels_' + str(i) + '.npy')
    for j in range(0, 1000):
        temp.append(x[j])
        tempy.append(y[j])

mat = np.array(temp)
maty = np.array(tempy)

del temp
del tempy
#X_train = np.take(mat, mat[rand, :])
#y_train = np.take(maty, maty[rand, :])
#X_test = np.take(mat, mat[reaminder, :])
#y_test = np.take(maty, maty[remainder, :])

X_train = mat[rand, :]
y_train = maty[rand, :]
X_test = mat[remainder, :]
y_test = maty[remainder, :]
np.save('/home/ben/dev/AgePredictor/faces/data/faces_train.npy', X_train)
np.save('/home/ben/dev/AgePredictor/faces/labels/labels_train.npy', y_train)
np.save('/home/ben/dev/AgePredictor/faces/data/faces_test.npy', X_test)
np.save('/home/ben/dev/AgePredictor/faces/labels/labels_test.npy', y_test)
