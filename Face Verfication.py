import random
import numpy as np
import cv2
import os
from PySide2.QtWidgets import QApplication, QMainWindow, QPushButton, QPlainTextEdit,QComboBox
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets, Sequential
import glob

from skimage import io, transform
import csv
import time

import dlib



IMGSIZE = 160
size=160

# Create folder
def createdir(*args):
    for item in args:
        if not os.path.exists(item):
            os.makedirs(item)

def relight(imgsrc, alpha=1, bias=0):
    imgsrc = imgsrc.astype(float)
    imgsrc = imgsrc * alpha + bias
    imgsrc[imgsrc < 0] = 0
    imgsrc[imgsrc > 255] = 255
    imgsrc = imgsrc.astype(np.uint8)
    return imgsrc

def getfacefromcamera(name,outdir):
    createdir(outdir)
    camera = cv2.VideoCapture(0)
    haar = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    n = 1
    while 1:
        # 创建200张64*64图片
        if n <= 2000:
            print('It`s processing %s image.' % n)
            success, img = camera.read()
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = haar.detectMultiScale(gray_img, 1.3, 5)
            # f_x, f_y, f_w, f_h分别为获取面部的左上角x, y坐标值，宽高值（原点（0， 0）在图片左上角）
            for f_x, f_y, f_w, f_h in faces:
                # 截取面部图片，先写y方向，再写x方向，别写反了（可以尝试尝试写反获取的图片）
                face = img[f_y:f_y+f_h, f_x:f_x+f_w]
                # 修改图片大小为64*64
                face = cv2.resize(face, (IMGSIZE, IMGSIZE))
                # 随机改变图片的明亮程度，增加图片复杂度
                face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
                # 保存图片
                cv2.imwrite(os.path.join(outdir, str(n)+'.jpg'), face)
                # 在原图img面部上方20处写下你的名字
                cv2.putText(img, name, (f_x, f_y-20), cv2.FONT_HERSHEY_COMPLEX, 1, 255, 2)
                # 画出方框，框选出你的face
                img = cv2.rectangle(img, (f_x, f_y), (f_x + f_w, f_y + f_h), (255, 0, 0), 2)
                n += 1
            cv2.imshow('img', img)
            key = cv2.waitKey(30) & 0xff
            if key == 27:
                break
        else:
            break
    camera.release()
    cv2.destroyAllWindows()

def button_getface():
    name = text_name.toPlainText()
    if len(name)>0:
        getfacefromcamera(name, os.path.join('face_images', name))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

root_img = './face_images'
img_mean = tf.constant([0.485, 0.456, 0.406])
img_std = tf.constant([0.229, 0.224, 0.225])

# 由卷积层和全连接层组成，中间用Flatten层进行平铺, inputs: [None, 64, 64, 3]
my_layers = [

    layers.Conv2D(32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

    layers.GlobalMaxPool2D(),

    layers.Dense(512, activation=tf.nn.relu),
    layers.Dropout(rate=0.5),
    layers.Dense(2, activation=None)
]

# root为我们之前获得图片数据的根目录face_images，filename为我们要加载的csv文件，
# name2label为我们获取的图片类型字典
def load_csv(root, filename, name2label):
    # 如果根目录root下不存在filename文件，那么创建一个filename文件
    if not os.path.exists(os.path.join(root, filename)):
        # 创建一个图片路径的列表images
        images = []
        # 遍历字典里所有的元素
        for name in name2label.keys():
            # 将路径下所有的jpg图片的路径写至images列表中
            images += glob.glob(os.path.join(root, name, '*.jpg'))
            # print('addr:', glob.glob(os.path.join(root, name, '*.jpg')))
        print(len(images), images)
        # 对images进行随机打乱
        random.shuffle(images)
        with open(os.path.join(root, filename), mode='w', newline='') as f:
            writer = csv.writer(f)
            for img in images:
                # 获取路径最底层文件夹的名字
                # os.sep为路径的分隔符，split()函数以给定分隔符进行切片（默认以空格切片），取从右往左数第二个
                # img = '...a/b/c/haha.jpg' =>['...a', 'b', 'c', 'haha.jpg'], -2指的是'c'
                name = img.split(os.sep)[-2]
                # 查找字典对应元素的值
                label = name2label[name]
                # 添加到路径的后面
                writer.writerow([img, label])
            print('written into csv file:', filename)
    # 如果存在filename文件，将其读取至imgs, labels这两个列表中
    imgs, labels = [], []
    with open(os.path.join(root, filename)) as f:
        reader = csv.reader(f)
        for row in reader:
            # 读取路径和对应的label值
            img, label = row
            label = int(label)
            # 将其分别压入列表中，并返回出来
            imgs.append(img)
            labels.append(label)
    return imgs, labels

def load_faceimg(root, mode='train'):
    # 创建图片类型字典，准备调用load_csv()方法
    name2label = {}
    for name in sorted(os.listdir(os.path.join(root))):
        # 跳过root目录下不是文件夹的文件
        if not os.path.isdir(os.path.join(root, name)):
            continue
        # name为根目录下各个文件夹的名字
        # name2label.keys()表示字典name2label里所有的元素，len()表示字典所有元素的个数
        # 一开始字典是没有元素的，所以p1的值为0, 之后字典元素有个一个，所以p2的值为1
        name2label[name] = len(name2label.keys())
    # 调用load_csv（）方法，返回值images为储存图片的目录的列表，labels为储存图片种类(0, 1两种)的列表
    images, labels = load_csv(root, 'images.csv', name2label)
    # 我们将前60%取为训练集，后20%取为验证集，最后20%取为测试集，并返回
    if mode == 'train':
        images = images[:int(0.6 * len(images))]
        labels = labels[:int(0.6 * len(labels))]
    elif mode == 'val':
        images = images[int(0.6 * len(images)):int(0.8 * len(images))]
        labels = labels[int(0.6 * len(labels)):int(0.8 * len(labels))]
    else:
        images = images[int(0.8 * len(images)):]
        labels = labels[int(0.8 * len(labels)):]
    return images, labels, name2label

def normalize(x, mean=img_mean, std=img_std):
    # 标准化
    # x: [64, 64, 3]
    # mean: [64, 64, 3], std: [3]
    x = (x - mean) / std
    return x

def denormalize(x, mean=img_mean, std=img_std):
    # 标准化的逆过程
    x = x * std + mean
    return x


# x：图片的路径List, y: 图片种类的数字编码List
def get_tensor(x, y):
    # 创建一个列表ims
    ims = []
    for i in x:
        # 读取路径下的图片
        p = tf.io.read_file(i)
        # 对图片进行解码，RGB，3通道
        p = tf.image.decode_jpeg(p, channels=3)
        # 修改图片大小为64*64
        p = tf.image.resize(p, [64, 64])
        # 将图片压入ims列表中
        ims.append(p)
    # 将List类型转换为tensor类型，并返回
    ims = tf.convert_to_tensor(ims)
    y = tf.convert_to_tensor(y)
    return ims, y


# 预处理函数，x, y均为tensor类型
def preprocess(x, y):
    # 数据增强
    x = tf.image.random_flip_left_right(x)  # 左右镜像
    x = tf.image.random_crop(x, [64, 64, 3])  # 随机裁剪
    # x: [0,255]=>0~1，将其值转换为float32
    x = tf.cast(x, dtype=tf.float32) / 255.
    # 0~1 => D(0, 1)
    x = normalize(x)
    # 将其值转换为int32
    y = tf.cast(y, dtype=tf.int32)
    return x, y

# 加载图片，获得图片路径与图片种类编码的列表


def model_train(model_name='model1'):
    images_train, labels_train, name2label = load_faceimg(root_img, mode='train')
    images_val, labels_val, _ = load_faceimg(root_img, mode='val')
    images_test, labels_test, _ = load_faceimg(root_img, mode='test')

    print('images_train:', images_train)

    # 从对应路径读取图片，并将列表转换为张量
    x_train, y_train = get_tensor(images_train, labels_train)
    x_val, y_val = get_tensor(images_val, labels_val)
    x_test, y_test = get_tensor(images_test, labels_test)

    # 可输出查看它们的shape
    print('x_train:', x_train.shape, 'y_train:', y_train.shape)
    print('x_val:', x_val.shape, 'y_val:', y_val.shape)
    print('x_test:', x_test.shape, 'y_test:', y_test.shape)
    # print('images:', len(images_test))
    # print('labels:', len(labels_test))
    # print('name2label', name2label)

    # 切分传入参数的第一个维度，并进行随机打散，预处理和打包处理
    db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    db_train = db_train.shuffle(1000).map(preprocess).batch(50)

    db_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    db_val = db_val.map(preprocess).batch(10)

    db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    db_test = db_test.map(preprocess).batch(10)

    # 创建一个迭代器，可以查看其shape大小
    sample_train = next(iter(db_train))
    sample_val = next(iter(db_val))
    sample_test = next(iter(db_test))
    print('sample_train:', sample_train[0].shape, sample_train[1].shape)
    print('sample_val:', sample_val[0].shape, sample_val[1].shape)
    print('sample_test:', sample_test[0].shape, sample_test[1].shape)
    my_net = Sequential(my_layers)
    
    my_net.build(input_shape=[None, 64, 64, 3])
    my_net.summary()
    
    optimizer = optimizers.Adam(learning_rate=1e-3)
    acc_best = 0
    patience_num = 1
    no_improved_num = 0
    for epoch in range(100):
        for step, (x, y) in enumerate(db_train):
            with tf.GradientTape() as tape:
                out = my_net(x)
                # print('out', out.shape)
                logits = out
                y_onehot = tf.one_hot(y, depth=2)
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, my_net.trainable_variables)
            optimizer.apply_gradients(zip(grads, my_net.trainable_variables))
    
            if step % 5 == 0:
                print(epoch, step, 'loss:', float(loss))
        total_num = 0
        total_correct = 0
        for x2, y2 in db_test:
            out = my_net(x2)
            logits = out
            prob = tf.nn.softmax(logits, axis=1)
            # tf.argmax() : axis=1 表示返回每一行最大值对应的索引, axis=0 表示返回每一列最大值对应的索引
            pred = tf.argmax(prob, axis=1)
            # 将pred转化为int32数据类型，便于后面与y2进行比较
            pred = tf.cast(pred, dtype=tf.int32)
    
            correct = tf.cast(tf.equal(pred, y2), dtype=tf.int32)
            correct = tf.reduce_sum(correct)
    
            total_num += x2.shape[0]
            total_correct += int(correct)
        acc = total_correct / total_num
        if acc > acc_best:
            acc_best = acc
            no_improved_num = 0
            my_net.save(model_name+'.h5')
        else:
            no_improved_num += 1
        print(epoch, 'acc:', acc, 'no_improved_num:', no_improved_num)
        if no_improved_num >= patience_num:
            break
def train_button():
    model_name = text_model.toPlainText()
    if len(model_name)>0:
        model_train(model_name)
        ComboBox.addItem(model_name+'.h5')
    else:
        model_train()

def get_accurate(lfw_filename):
    detector = dlib.get_frontal_face_detector()
    my_net = tf.keras.models.load_model('model1.h5')

    sum_accurate = 0
    count = 0

    for filename in os.listdir(r"./"+lfw_filename):
        lfw_img = cv2.imread(lfw_filename + "/" + filename)
        gray_img = cv2.cvtColor(lfw_img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_img, 1)
        for i, d in enumerate(faces):
            # 获得面部图片

            x1 = d.top() if d.top() > 0 else 0
            y1 = d.bottom() if d.bottom() > 0 else 0
            x2 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0
            face = lfw_img[x1:y1,x2:y2]

            # 修改图片大小为64*64
            face = cv2.resize(face, (size, size))
            # 将图片数据类型转换为tensor类型，完成后shape为[64, 64, 3]
            face_tensor = tf.convert_to_tensor(face)
            # 在0维度左侧增加一个维度，即[64, 64, 3]=>[1, 64, 64, 3]
            face_tensor = tf.expand_dims(face_tensor, axis=0)
            # 将tensor类型从uint8转换为float32
            face_tensor = tf.cast(face_tensor, dtype=tf.float32)
            # print('face_tensor', face_tensor)
            # 输入至网络
            logits = my_net(face_tensor)
            # 将每一行进行softmax
            in_data = logits.numpy()
            in_data = max(abs(max(in_data)))
            sum_accurate = in_data + sum_accurate
        count = count + 1

    avg_acc = sum_accurate/count
    return avg_acc

def namelist(img_file):
    NameList=[]
    files=os.listdir(img_file)
    print('files:',files)

    for f in files:
        if os.path.isdir(img_file + '/'+f):   #这里是绝对路径，该句判断目录是否是文件夹
            NameList.append(f)

    Name1 = NameList[0]
    Name2 = NameList[1]
    return Name1, Name2

def run(model_name='model1.h5'):
    name1, name2 = namelist('./face_images')

    p1 = [0]
    p2 = [1]
    p1 = tf.convert_to_tensor(p1, dtype=tf.int32)
    p2 = tf.convert_to_tensor(p2, dtype=tf.int32)

    detector = dlib.get_frontal_face_detector()
    my_net = tf.keras.models.load_model('model1.h5')

    acc = get_accurate('lfw_faces')

    camera = cv2.VideoCapture(0)
    #haar = cv2.CascadeClassifier('./haarcascade_frontalface_alt2.xml')
    n = 1

    while True:
        if n <= 2000:
            print('It`s processing %s image.' % n)
            success, img = camera.read()

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #faces = haar.detectMultiScale(gray_img, 1.3, 5)
            faces = detector(gray_img, 1)

            #for f_x, f_y, f_w, f_h in faces:
            for i, d in enumerate(faces):
                # 获得面部图片
                #face = img[f_y:f_y + f_h, f_x:f_x + f_w]

                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0
                face = img[x1:y1,x2:y2]

                # 修改图片大小为64*64
                face = cv2.resize(face, (size, size))
                # 将图片数据类型转换为tensor类型，完成后shape为[64, 64, 3]
                face_tensor = tf.convert_to_tensor(face)
                # 在0维度左侧增加一个维度，即[64, 64, 3]=>[1, 64, 64, 3]
                face_tensor = tf.expand_dims(face_tensor, axis=0)
                # 将tensor类型从uint8转换为float32
                face_tensor = tf.cast(face_tensor, dtype=tf.float32)
                # print('face_tensor', face_tensor)
                # 输入至网络
                logits = my_net(face_tensor)
                # 将每一行进行softmax
                print('logits:', logits)
                in_data = logits.numpy()
                in_data = max(abs(max(in_data)))
                print('abc:', in_data)

                if in_data<(acc+acc*0.2):
                    
                    prob = tf.nn.softmax(logits, axis=1)
                    # 取出prob中每一行最大值对应的索引
                    pred = tf.argmax(prob, axis=1)
                    print('pred:', pred)
                    pred = tf.cast(pred, dtype=tf.int32)
                    print('pred:', pred)
                    if tf.equal(pred, p1):
                        cv2.putText(img, name1, (x2+10, y1-50), cv2.FONT_HERSHEY_COMPLEX, 1, 255, 2)
                        #cv2.putText(img, 'weijia', (f_x, f_y - 20), cv2.FONT_HERSHEY_COMPLEX, 1, 255, 2)
                    if tf.equal(pred, p2):
                        cv2.putText(img, name2, (x2+10, y1-50), cv2.FONT_HERSHEY_COMPLEX, 1, 255, 2)
                        #cv2.putText(img, 'zichen', (f_x, f_y - 20), cv2.FONT_HERSHEY_COMPLEX, 1, 255, 2)
                    #img = cv2.rectangle(img, (f_x, f_y), (f_x + f_w, f_y + f_h), (255, 0, 0), 2)
                    img = cv2.rectangle(img, (x2, x1), (y2,y1), (255, 0, 0), 2)
                    n += 1
                else:
                    cv2.putText(img, 'No Found in Dataset', (x2+10, y1-50), cv2.FONT_HERSHEY_COMPLEX, 1, 255, 2)
                    img = cv2.rectangle(img, (x2, x1), (y2,y1), (255, 0, 0), 2)
            cv2.imshow('Press esc to exit', img)
            key = cv2.waitKey(30) & 0xff
            if key == 27:
                break
        else:
            break
    camera.release()
    cv2.destroyAllWindows()
def check_face_button():
    model=ComboBox.currentText()
    run(model_name=model)


app=QApplication([])
window=QMainWindow()
window.resize(500,210)
window.setWindowTitle('Face Verification')

text_name=QPlainTextEdit(window)
text_name.setPlaceholderText('Please type in your name')
text_name.move(10,25)
text_name.resize(300,50)

button1=QPushButton('Get Face',window)
button1.move(350,35)
button1.clicked.connect(button_getface)

text_model=QPlainTextEdit(window)
text_model.setPlaceholderText('Please type in the model name you like')
text_model.move(10,80)
text_model.resize(300,50)

ComboBox = QComboBox(window)
ComboBox.move(180,135)
ComboBox.addItem("model1.h5")

button2=QPushButton('Train Model',window)
button2.move(350,85)
button2.clicked.connect(train_button)



button3=QPushButton('Check My Face',window)
button3.move(180,170)
button3.clicked.connect(run)

window.show()

app.exec_()