from mtcnn import MTCNN
from PIL import Image
from numpy import asarray
from numpy import savez_compressed
import cv2 
from os import listdir
from os.path import isdir
from matplotlib import pyplot
import csv
from numpy import load
from numpy.core.fromnumeric import mean
from numpy.lib.npyio import load
from keras.models import load_model
from numpy.lib.shape_base import expand_dims
import os
TRAIN_DATA = []


'''to extract from the image we are going to create a method 
that will hendle that '''
def extract_faces(filename, required_size=(160,160)) -> list:
    # loading the image from the file
    image = Image.open(filename)
    # convert the image to rgb
    image = image.convert('RGB')
    # converting to array
    pixels = asarray(image)
    detactor = MTCNN()
    data= detactor.detect_faces(pixels)
    x1,y1,width,height= data[0]['box']
    x1,y1 = abs(x1), abs(y1)
    x2,y2 = x1+width, y1+height
    faces = pixels[y1:y2,x1:x2]
    # sizing the images
    image = Image.fromarray(faces)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array



def load_faces(filename):
    faces = list()
    for file in listdir(filename):
        path = filename + file
        face = extract_faces(path)
        faces.append(face)
        
    return faces


def load_dataset(foldername):
    x,y = list(),list()
    
    for subdir in listdir(foldername):
        # path
        path = foldername + subdir + "/"
        if not isdir(path):
            continue
        # loading the faces
        faces = load_faces(path)
        labels = [subdir for _ in range(len(faces))]
        print(f"loaded {len(faces)} examples for class: {subdir}")
        x.extend(faces)
        y.extend(labels)
    return  asarray(x), asarray(y)
    

def get_embedding(model,face_pixels):
    face_pixels =  face_pixels.astype('float32')
    mean,std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean)/std
    samples = expand_dims(face_pixels,axis=0)
    yhat = model.pridict(samples)
    return yhat[0]
# defining the file
filename = 'demoreader/src/images/'

filename_test = 'demoreader/src/test/'
train_x, train_y = load_dataset(filename)
print(train_x.shape,train_y.shape)
train_data = train_x.shape,train_y.shape
with open('demoreader/train_data.csv',mode='w') as file:
    csv_out = csv.writer(file)
    for row in train_data:
        csv_out.writerow(row)
    
print("after the test data")
test_x, test_y = load_dataset(filename_test)
savez_compressed('demoreader/student_data_faces.npz',train_x,train_y,test_x,test_y)
print(test_x.shape,test_y.shape)
test_data = test_x.shape,test_y.shape
with open('demoreader/test_data.csv',mode='w') as file:
    csv_out = csv.writer(file)
    for row in test_data:
        csv_out.writerow(row)
        
# now i am going to load the dataset from the npz file
# with the help of load() method
data = load('demoreader/student_data_faces.npz')
train_x,train_y,test_x,test_y =     data['arr_0'],data['arr_1'],data['arr_2'],data['arr_3']

print("Loaded:", train_x.shape,train_y.shape,test_x.shape,test_y.shape)
# # demoreader/model/facenet_keras.h5
# # loading the facenet model
# f_net_model = load_model('demoreader/model/facenet_keras.h5')
# print("Loaded Model")

# # now i am going to train every set of face to an embedding
# new_train_x = list()

# for face_pixels in train_x:
#     embedding = get_embedding(f_net_model,face_pixels)
#     new_train_x.append(embedding)
# new_train_x = asarray(new_train_x)
# print(new_train_x.shape)

# # for the test data
# new_test_x = list()

# for face_pixels in test_x:
#     embedding = get_embedding(f_net_model,face_pixels)
#     new_test_x.append(new_test_x)
# new_test_x = asarray(new_test_x)
# print(new_test_x)

print("Fcace classification")

print(f"Dataset : train={train_x.shape[0]},test={test_x.shape[0]}")








# i = 1
# for file in listdir(filename):
#     path = filename + file
#     face = extract_faces(path)
#     print(i,face.shape)
#     pyplot.subplot(2 , 7, i)
#     pyplot.axis('off')
#     pyplot.imshow(face)
#     i += 1
# pyplot.show()

# print(exels)



