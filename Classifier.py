from PIL import Image
import numpy as np
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from keras.models import model_from_json
from keras.datasets import cifar10
import os
import cv2

TRAIN_PATH = "Images"

def from_files_to_array():
    counter = 0
    data = []
    labels = []
    files_path = os.listdir(TRAIN_PATH)
    files_path = files_path
    for dog_type in files_path:
        dogs_images = os.listdir(TRAIN_PATH+"/"+dog_type)
        for dog in dogs_images:
            imag = cv2.imread(TRAIN_PATH+"/"+dog_type + "/" + dog)
            img_from_ar = Image.fromarray(imag, 'RGB')
            resized_image = img_from_ar.resize((50, 50))
            data.append(np.array(resized_image))
            labels.append(counter)
        counter += 1
    animals = np.array(data)
    labels = np.array(labels)
    np.save("animals", animals)
    np.save("labels", labels)
    return animals, labels

def create_animal_arrays():
    return from_files_to_array()
    # data = []
    # labels = []
    # cats = os.listdir()
    # for cat in cats:
    # 	imag = cv2.imread("bla"+"/" + cat)
    # 	img_from_ar = Image.fromarray(imag, 'RGB')
    # 	resized_image = img_from_ar.resize((50, 50))
    # 	data.append(np.array(resized_image))
    # 	labels.append(0)
    # dogs = os.listdir("Dogs")
    # for dog in dogs:
    # 	imag = cv2.imread("Dogs/" + dog)
    # 	img_from_ar = Image.fromarray(imag, 'RGB')
    # 	resized_image = img_from_ar.resize((50, 50))
    # 	data.append(np.array(resized_image))
    # 	labels.append(1)
    #
    # butterflies = os.listdir("Butterflies")
    # for butterfly in butterflies:
    # 	imag = cv2.imread("Butterflies/" + butterfly)
    # 	img_from_ar = Image.fromarray(imag, 'RGB')
    # 	resized_image = img_from_ar.resize((50, 50))
    # 	data.append(np.array(resized_image))
    # 	labels.append(2)
    #
    # elephants = os.listdir("Elephants")
    # for elephant in elephants:
    # 	imag = cv2.imread("Elephants/" + elephant)
    # 	img_from_ar = Image.fromarray(imag, 'RGB')
    # 	resized_image = img_from_ar.resize((50, 50))
    # 	data.append(np.array(resized_image))
    # 	labels.append(3)
    #
    # animals = np.array(data)
    # labels = np.array(labels)
    # np.save("animals",animals)
    # np.save("labels",labels)
    # return animals, labels

def load_animal_array():
    animals = np.load("animals.npy")
    labels = np.load("labels.npy")
    return animals, labels

def shuffle_data(animals, labels):
    s=np.arange(animals.shape[0])
    np.random.shuffle(s)
    animals=animals[s]
    labels=labels[s]
    return animals, labels

def divide_data(animals, labels):
    """ 90% of data in train set and 10% in test set """
    num_classes=len(np.unique(labels)) #total number of animal categories
    data_length=len(animals) #size of dataset

    (x_train,x_test)=animals[(int)(0.1*data_length):],animals[:(int)(0.1*data_length)]
    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255
    # train_length=len(x_train)
    # test_length=len(x_test)

    (y_train,y_test)=labels[(int)(0.1*data_length):],labels[:(int)(0.1*data_length)]

    #One hot encoding
    y_train=keras.utils.to_categorical(y_train,num_classes)
    y_test=keras.utils.to_categorical(y_test,num_classes)

    return x_train, y_train, x_test, y_test

#make model
def make_model():
    model=Sequential()
    model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32, kernel_size=2, padding="same",
                     activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(500,activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(12, activation="softmax"))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy'])
    return model

def fit_model(model, x_train, y_train):
    model.fit(x_train,y_train,batch_size=10,epochs=50,verbose=1)

def score(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=1)
    print('\n', 'Test accuracy:', score[1])

def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


# predict on one image

def convert_to_array(img):
    im = cv2.imread(img)
    img = Image.fromarray(im, 'RGB')
    image = img.resize((50, 50))
    return np.array(image)

def get_animal_name(label):
    dog_types = os.listdir(TRAIN_PATH)
    return dog_types[label]
    # if label==0:
    # 	return "Cat"
    # if label==1:
    # 	return "Dog"
    # if label==2:
    # 	return "Butterfly"
    # if label==3:
    # 	return "Elephant"


def predict_animal(file, model):
    print("Predicting .................................")
    ar=convert_to_array(file)
    ar=ar/255
    # label=1
    a = list()
    a.append(ar)
    a=np.array(a)
    print("**********")
    score=model.predict(a,verbose=1)
    print(score)
    print("**********")
    label_index=np.argmax(score)
    print(label_index)
    acc=np.max(score)
    animal=get_animal_name(label_index)
    print(animal)
    print("The predicted Animal is a "+animal+" with accuracy =    "+str(acc))
    save_model(model)

def first_time_run(image):
    animals, labels = create_animal_arrays()
    animals, labels = shuffle_data(animals, labels)
    x_train, y_train, x_test, y_test = divide_data(animals, labels)
    model =make_model()
    fit_model(model, x_train, y_train)
    score(model, x_test, y_test)
    files_path = os.listdir(image)
    for i, animal in enumerate(files_path):
        if not "ini" in animal:
            print()
            print("index in folder: ", i)
            print("animal is: ",animal)
            predict_animal(image + "/" + animal, model)
    # predict_animal(image, model)

def second_time_run(image):
    animals, labels = load_animal_array()
    animals, labels = shuffle_data(animals, labels)
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy'])
    files_path = os.listdir(image)
    for animal in files_path:
        if not "ini" in animal:
            print("animal is: ",animal)
            predict_animal(image + "/" + animal, loaded_model)

if __name__ == '__main__':
    first_time_run("test")
    # second_time_run("test")





