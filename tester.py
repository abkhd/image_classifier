import tensorflow as tf
import os
import cv2
import imghdr
from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.metrics import Precision, Recall, BinaryAccuracy
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox


model = load_model(os.path.join('models', 'image-sentiment-model.h5'))


def classify_image():
    filename = openfile()
    img = cv2.imread(filename)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    resize = tf.image.resize(img, (256, 256))
    plt.show()
    extra_dim = np.expand_dims(resize / 255, 0)  # expands dimensions so model can read it
    yhat = model.predict(extra_dim)
    if yhat > 0.5:
        messagebox.showinfo(title='Classifier', message='Prediction: Sad')
    else:
        messagebox.showinfo(title='Classifier', message='Prediction: Happy')



def openfile():
    filepath = filedialog.askopenfilename(title='Open File') #gets filename
    return filepath

window = Tk()
window.geometry('500x50')
label = Label(text='classify an image of a person as happy or sad')
label.pack()
button1 = Button(window, text='classify image', command=classify_image)
button1.pack()
window.mainloop()

