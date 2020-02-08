import tensorflow as tf
from tensorflow import keras
import os,gzip
import numpy as np
import matplotlib.pyplot as plt
fashion_mnist=keras.datasets.fashion_mnist
path='./data'
files=['train-labels-idx1-ubyte.gz',
       'train-images-idx3-ubyte.gz',
       't10k-labels-idx1-ubyte.gz',
       't10k-images-idx3-ubyte.gz',]
def load_data(data_folder,files):
    paths=[]
    for fname in files:
        paths.append(os.path.join(data_folder,fname))
    ##frombuffer将data以流的形式读入转化成ndarray对象
    #第一参数为stream,第二参数为返回值的数据类型，第三参数指定从stream的第几位开始读入
    with gzip.open(paths[0],'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(),np.uint8,offset=8)
    with gzip.open(paths[1],'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(),np.uint8,offset=16).reshape(len(y_train),28,28)
    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(paths[3],'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(),np.uint8,offset=16).reshape(len(y_test),28,28)
    return (x_train,y_train),(x_test,y_test)
(train_images,train_labels),(test_images,test_labels)=load_data(path,files)
class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Shirt','Sneaker','Bag','Ankle boot']
train_images=train_images/255.0
test_images=test_images/255.0
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
# plt.show()
#建模
model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(10,activation='softmax')])
#编译训练模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images,train_labels,epochs=10)
#评估模型及预测
test_loss,test_acc = model.evaluate(test_images,test_labels,verbose=2)
print('\nTest accuracy:',test_acc)
predictions = model.predict(test_images)
print(predictions[1])
print(np.argmax(predictions[1]))
print(test_labels[1])
def plot_image(i,predictions_array,true_label,img):
    predictions_array,true_label,img=predictions_array,true_label[i],img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img,cmap=plt.cm.binary)
    predicted_label=np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color='red'
    plt.xlabel('{}{:2.0f}%({})'.format(class_names[predicted_label],
                                       100*np.max(predictions_array),
                                       class_names[true_label]),
               color=color)
def plot_value_array(i,predictions_array,true_label):
    predictions_array,true_label=predictions_array,true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot=plt.bar(range(10),predictions_array,color='#777777')
    plt.ylim([0,1])
    predicted_label=np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
# for i in range(10):
#     plt.figure(figsize=(6,3))
#     plt.subplot(1,2,1)
#     plot_image(i,predictions[i],test_labels,test_images)
#     plt.subplot(1,2,2)
#     plot_value_array(i,predictions[i],test_labels)
#     plt.show()
#
num_rows=5
num_cols=3
num_images=num_rows*num_cols
plt.figure(figsize=(2*2*num_cols,2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows,2*num_cols,2*i+1)
    plot_image(i,predictions[i],test_labels,test_images)
    plt.subplot(num_rows,2*num_cols,2*i+2)
    plot_value_array(i,predictions[i],test_labels)
plt.tight_layout()
plt.show()
