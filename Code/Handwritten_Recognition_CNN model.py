

#A simple CNN model for handwritten recognition using mnist dataset
 
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dropout,Conv2D, Dense, Flatten, MaxPooling2D
from keras.optimizers import SGD
import keras
from matplotlib import pyplot as plt

#Load the MNIST dataset
(x_train, y_train),(x_test, y_test)=mnist.load_data()
print(x_train.shape)

#Use opencv to display 6 random images from dataset
import numpy as np
import cv2
for i in range(0,6):
    rand_num=np.random.randint(0, len(x_train))
    img=x_train[rand_num]
    display="Random image ##"+str(i)
    cv2.imshow(display ,img)
    cv2.waitKey()

cv2.destroyAllWindows()

#Getting our dataset in the right shape needed for keras
img_rows=x_train[0].shape[0]
img_cols=x_train[1].shape[0]

x_train=x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test=x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

#Store the shape of a single image 
input_shape=(img_rows, img_cols, 1)

#Change our image type to float32 datatype
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')

#Normalize our data by changing the range from (0,255) to(0,1)
x_train /=255
x_test /=255

print("x_train samples: ",x_train.shape)
print(x_train.shape[0], "Train samples")
print(x_test.shape[0], "Test samples")

#Now we hot encode outputs
from keras.utils import np_utils

y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)

#Lets count the no. of columns in our hot encoded matrix
print("No. of classes:"+ str(y_test.shape[1]))

num_classes=y_test.shape[1]
num_pixels=x_train.shape[1]*x_train.shape[2]

#Create model
model=Sequential()

model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3,3), activation='relu' ))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=SGD(0.01), metrics=['accuracy'] )
print(model.summary())

#Train our model
batch_size=32
epochs=7

history=model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, 
                   validation_data=(x_test, y_test))
score=model.evaluate(x_test, y_test, verbose=0)
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])

#Plotting our loss charts
history_dic=history.history
loss_values=history_dic['loss']
val_loss_values=history_dic['val_loss']
epochs=range(1, len(loss_values)+1)
line1=plt.plot(epochs, val_loss_values, label='Validation/test loss')
line2=plt.plot(epochs, loss_values, label='Training loss')
plt.setp(line1, linewidth=2.0, marker='+', markersize=10.0 )
plt.setp(line2, linewidth=2.0, marker='*', markersize=10.0 )
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.show()

#Plotting our accuracy charts
history_dic=history.history
acc_values=history_dic['accuracy']
val_acc_values=history_dic['val_accuracy']
epochs=range(1, len(loss_values)+1)
line1=plt.plot(epochs, val_acc_values, label='Validation/test accuracy')
line2=plt.plot(epochs, acc_values, label='Training accuracy')
plt.setp(line1, linewidth=2.0, marker='+', markersize=10.0 )
plt.setp(line2, linewidth=2.0, marker='*', markersize=10.0 )
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.show()

#Save our model
model.save(r"C:\Users\Shruti\Documents\Neural Networks\Simple_CNN_mnist.py")
print("Model Saved")

#Loading our model
from keras.models import load_model
classifier=load_model(r"C:\Users\Shruti\Documents\Neural Networks\Simple_CNN_mnist.py")

#Lets input some of our test data to classifier
def draw_test(name, pred, input_img):
    BLACK=[0,0,0]
    expanded_image=cv2.copyMakeBorder(input_img, 0,0,0, imageL.shape[0], 
                                      cv2.BORDER_CONSTANT, value=BLACK)
    expanded_image=cv2.cvtColor(expanded_image, cv2.COLOR_GRAY2BGR)
    cv2.putText(expanded_image, str(pred), (152, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 
                4, (0,255,0), 2)
    cv2.imshow(name, expanded_image)
    
for i in range(0, 10):
    rand=np.random.randint(0, len(x_test))
    input_img=x_test[rand]
    
    imageL=cv2.resize(input_img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    input_img=input_img.reshape(1,28,28,1)
    
    #Get prediction
    res=str(classifier.predict_classes(input_img, 1, verbose=0)[0])
    
    if display:
        draw_test("Prediction", res, imageL)
        cv2.waitKey(0)
cv2.destroyAllWindows()

#Making classification_report and confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
y_pred=model.predict_classes(x_test)

print(classification_report(np.argmax(y_test, axis=1), y_pred))
print(confusion_matrix(np.argmax(y_test, axis=1), y_pred))

#Use numpy to create array that stores a value of 1 when any misclassification occurs
(x_train, y_train),(x_test, y_test)=mnist.load_data()
result=np.absolute(y_test-y_pred)
result_indices=np.nonzero(result>0)

print("Indices of misclassified \n\n"+ str(result_indices))

#Displaying misclassifications
def draw_test(name, pred, input_img, true_label):
    BLACK=[0,0,0]
    expanded_image=cv2.copyMakeBorder(input_img, 0,0,0, imageL.shape[0]*2, 
                                      cv2.BORDER_CONSTANT, value=BLACK)
    expanded_image=cv2.cvtColor(expanded_image, cv2.COLOR_GRAY2BGR)
    cv2.putText(expanded_image, str(pred), (152, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 
                4, (0,255,0), 2)
    cv2.putText(expanded_image, str(true_label), (250, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 
                4, (0,0,255), 2)
    cv2.imshow(name, expanded_image)
    
    
for i in range(0, 10):
    
    input_img=x_test[result_indices[0][i]]
    
    imageL=cv2.resize(input_img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    input_img=input_img.reshape(1,28,28,1)
    
    #Get prediction
    res=str(classifier.predict_classes(input_img, 1, verbose=0)[0])
    
    if display:
        draw_test("Prediction", res, imageL, y_test[result_indices[0][i]])
        cv2.waitKey(0)
cv2.destroyAllWindows()








