import scipy.io as sio
data =  sio.loadmat("car_devkit/devkit/cars_train_annos.mat")
dataframe = data['annotations'][0]
print(dataframe)