import numpy as np
import tensorflow as tf
import sys
import time
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
   sys.path.remove(ros_path)
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

def single(img):
    image_single = img
    maxarx = np.argmax(img)
    i = int(maxarx/128)
    j = maxarx % 128
    maxone = img[i][j]
    minarx = np.argmin(img)
    i = int(minarx/128)
    j = minarx % 128
    minone = img[i][j]
    delta = 255/(maxone-minone)
    for i in range(128):
        for j in range(128):
            image_single[i][j] = int((img[i][j]-minone)*delta)
    return image_single

# 接下来开始使用神经网络模型
# 生成卷积核
def filterweight(data_shape):
    randomfilter = tf.random.normal(data_shape, stddev=0.01)
    return tf.Variable(randomfilter)


# 生成权重误差
def biasweight(data_shape):
    randomfilter = tf.random.normal(data_shape)
    return tf.Variable(randomfilter)


# 生成卷积层
def conv2d(input, filter):
    return tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')


# 生成池化层
def pool2d(input):
    return tf.nn.max_pool2d(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 避免过拟合
def avoid_overfit(input, keep_pro):
    return tf.nn.dropout(input, keep_pro)


# 初始化
pro1 = tf.compat.v1.placeholder(tf.float32)
pro2 = tf.compat.v1.placeholder(tf.float32)
pro3 = tf.compat.v1.placeholder(tf.float32)
pro4 = tf.compat.v1.placeholder(tf.float32)
pro5 = tf.compat.v1.placeholder(tf.float32)
pro6 = tf.compat.v1.placeholder(tf.float32)
proconnect = tf.compat.v1.placeholder(tf.float32)


# 创建网络
def create_neural(input):
    # 第一层卷积
    filter1 = filterweight([3, 3, 1, 32])  # 卷积核大小(3,3)， 输入通道(1)， 输出通道(32)
    bias1 = biasweight([32])
    conv1 = conv2d(input, filter1)
    conv1_handle = tf.nn.relu(conv1 + bias1)
    pool1 = pool2d(conv1_handle)
    layer1 = avoid_overfit(pool1, pro1)  # 避免过拟合

    # 第二层卷积
    filter2 = filterweight([3, 3, 32, 32])  # 卷积核大小(3,3)， 输入通道(32)， 输出通道(64)
    bias2 = biasweight([32])
    conv2 = conv2d(layer1, filter2)
    conv2_handle = tf.nn.relu(conv2 + bias2)
    pool2 = pool2d(conv2_handle)
    layer2 = avoid_overfit(pool2, pro2)  # 避免过拟合

    # 第三层卷积
    filter3 = filterweight([3, 3, 32, 64])  # 卷积核大小(3,3)， 输入通道(64)， 输出通道(64)
    bias3 = biasweight([64])
    conv3 = conv2d(layer2, filter3)
    conv3_handle = tf.nn.relu(conv3 + bias3)
    pool3 = pool2d(conv3_handle)
    layer3 = avoid_overfit(pool3, pro3)

    # 第四层卷积
    filter4 = filterweight([3, 3, 64, 64])  # 卷积核大小(3,3)， 输入通道(64)， 输出通道(64)
    bias4 = biasweight([64])
    conv4 = conv2d(layer3, filter4)
    conv4_handle = tf.nn.relu(conv4 + bias4)
    pool4 = pool2d(conv4_handle)
    layer4 = avoid_overfit(pool4, pro3)

    # 第五层卷积
    filter5 = filterweight([3, 3, 64, 128])  # 卷积核大小(3,3)， 输入通道(64)， 输出通道(64)
    bias5 = biasweight([128])
    conv5 = conv2d(layer4, filter5)
    conv5_handle = tf.nn.relu(conv5 + bias5)
    pool5 = pool2d(conv5_handle)
    layer5 = avoid_overfit(pool5, pro3)

    # 第六层卷积
    filter6 = filterweight([3, 3, 128, 64])  # 卷积核大小(3,3)， 输入通道(64)， 输出通道(64)
    bias6 = biasweight([64])
    conv6 = conv2d(layer5, filter6)
    conv6_handle = tf.nn.relu(conv6 + bias6)
    pool6 = pool2d(conv6_handle)
    layer6 = avoid_overfit(pool6, pro3)

    # 全连接层
    filter_connect = filterweight([256, 1024])
    bias_connect = biasweight([1024])
    layer6_flat = tf.reshape(layer6, [-1, 256])
    connect = tf.matmul(layer6_flat, filter_connect)
    connect_handle = tf.nn.relu(connect + bias_connect)
    dense = avoid_overfit(connect_handle, proconnect)

    # 输出层
    filterout = filterweight([1024, 3])
    biasout = biasweight([3])
    out = tf.matmul(dense, filterout)
    result = tf.add(out, biasout)
    return result

'''''''''''
def Readlabel2(path, label):
    number = 0
    for file in os.listdir(path):
        label_sigle = np.zeros(2)
        ID.append(file)
        label_sigle[number] = 1
        label.append(label_sigle)
        number += 1
        print(number)
    print(ID)
    # print(label)
'''''''''''
ID = []
def Readlabel(label):
    ID.append('1811299')
    ID.append('1711485')
    ID.append('somebody else')
    label_sigle = np.zeros(3)
    label_sigle[0] = 1
    label.append(label_sigle)
    label_sigle = np.zeros(3)
    label_sigle[1] = 1
    label.append(label_sigle)
    label_sigle = np.zeros(3)
    label_sigle[2] = 1
    label.append(label_sigle)


def test_withcamera():
    cap = cv2.VideoCapture(0)
    label = []
    Readlabel(label=label)
    input = tf.compat.v1.placeholder(tf.float32, [None, 128, 128, 1])
    result = create_neural(input)
    predict = tf.argmax(result, 1)
    print("create neural network successfully")
    saver = tf.train.Saver()
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.8  # 分配50%
    while 1:
        hx, img = cap.read()
        if hx == 1:
            faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imlocation = []
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(32, 32)
            )

            if len(faces) > 0:
                imgall = []
                for (x, y, w, h) in faces:
                    # 将人圈起来
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    one = img[y:y + h, x:x + w]
                    one = cv2.resize(one, (128, 128), interpolation=cv2.INTER_CUBIC)
                    image_height = one.shape[0]
                    image_weight = one.shape[1]
                    dest = np.zeros(one.shape, np.uint8)
                    # 翻转人脸的方向
                    for i in range(image_height):
                        for j in range(image_weight):
                            dest[i, j] = one[i, image_weight - 1 - j]
                    dest = cv2.cvtColor(dest, cv2.COLOR_RGB2GRAY)
                    dest = dest[:, :, np.newaxis]
                    dest = single(dest)
                    imgall.append(dest)
                    imlocation.append((x, y))

                # tf.reset_default_graph()
                with tf.Session(config=tfconfig) as sess:

                    sess.run(tf.global_variables_initializer())
                    saver.restore(sess, 'train_faces_target')
                    # is_res=1
                    print('restore ok')
                    out = sess.run(predict,
                                   feed_dict={input: imgall, pro1: 1.0, pro2: 1.0, pro3: 1.0, pro4: 1.0, pro5: 1.0,
                                              proconnect: 1.0})
                    # print(out)
                    num = len(out)
                    for i in range(num):
                        cv2.putText(img, ID[out[i]], imlocation[i], cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 255, 0), 5)
                    print(ID[out[0]])
                #cv2.imshow('video', img)
                # print('a',a)
            cv2.imshow('img', img)
            cv2.waitKey(10)


def test_withvideo(video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter('oto_other.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, size)

    label = []
    Readlabel(label=label)
    input = tf.compat.v1.placeholder(tf.float32, [None, 128, 128, 1])
    result = create_neural(input)
    predict = tf.argmax(result, 1)
    print("create neural network successfully")
    saver = tf.train.Saver()
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.8  # 分配80%
    while video.isOpened():
        hx, img = video.read()
        #print(img)
        #img = np.rot90(img)
        #print(img)
        if hx == 1:
            faceCascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imlocation = []
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(32, 32)
            )

            if len(faces) > 0:
                imgall = []
                for (x, y, w, h) in faces:
                    # 将人圈起来
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 5)
                    one = img[y:y + h, x:x + w]
                    one = cv2.resize(one, (128, 128), interpolation=cv2.INTER_CUBIC)
                    image_height = one.shape[0]
                    image_weight = one.shape[1]
                    dest = np.zeros(one.shape, np.uint8)
                    # 翻转人脸的方向
                    for i in range(image_height):
                        for j in range(image_weight):
                            dest[i, j] = one[i, image_weight - 1 - j]
                    dest = cv2.cvtColor(dest, cv2.COLOR_RGB2GRAY)
                    dest = dest[:, :, np.newaxis]
                    dest = single(dest)
                    imgall.append(dest)
                    imlocation.append((x, y))
                    #cv2.imshow('video', dst)
                # tf.reset_default_graph()
                with tf.Session(config=tfconfig) as sess:

                    sess.run(tf.global_variables_initializer())
                    saver.restore(sess, 'train_faces_target')
                    # is_res=1
                    print('restore ok')
                    out = sess.run(predict,
                                   feed_dict={input: imgall, pro1: 1.0, pro2: 1.0, pro3: 1.0, pro4: 1.0, pro5: 1.0,
                                              proconnect: 1.0})
                    # print(out)
                    num = len(out)
                    for i in range(num):
                        cv2.putText(img, ID[out[i]], imlocation[i], cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 255, 0), 5)
                    print(ID[out[0]])
                    # print('a',a)
                cv2.imshow('img',img)
                cv2.waitKey(int(1000/int(fps)))
        videoWriter.write(img)

    video.release()
    videoWriter.release()
    cv2.destroyAllWindows()

def test_withphoto(photo_path):
    image = cv2.imread(photo_path)
    label = []
    Readlabel('/home/qloveo/face-recon-qcx/processed_test', label=label)
    input = tf.compat.v1.placeholder(tf.float32, [None, 128, 128, 1])
    result = create_neural(input)
    predict = tf.argmax(result, 1)
    print("create neural network successfully")
    saver = tf.train.Saver()
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.8  # 分配80%
    while True:
        hx = 1
        if hx == 1:
            faceCascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            imlocation = []
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(32, 32)
            )

            if len(faces) > 0:
                imgall = []
                for (x, y, w, h) in faces:
                    # 将人圈起来
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    one = image[y:y + h, x:x + w]
                    one = cv2.resize(one, (128, 128), interpolation=cv2.INTER_CUBIC)
                    image_height = one.shape[0]
                    image_weight = one.shape[1]
                    dst = np.zeros(one.shape, np.uint8)
                    # 翻转人脸的方向
                    for i in range(image_height):
                        for j in range(image_weight):
                            dst[i, j] = one[i, image_weight - 1 - j]
                    imgall.append(dst)
                    imlocation.append((x, y))
                    cv2.imshow('video', dst)
                # tf.reset_default_graph()
                with tf.Session(config=tfconfig) as sess:

                    sess.run(tf.global_variables_initializer())
                    saver.restore(sess, 'train_faces_target')
                    # is_res=1
                    print('restore ok')
                    out = sess.run(predict,
                                   feed_dict={input: imgall, pro1: 1.0, pro2: 1.0, pro3: 1.0, pro4: 1.0, pro5: 1.0,
                                              proconnect: 1.0})
                    # print(out)
                    num = len(out)
                    for i in range(num):
                        cv2.putText(image, ID[out[i]], imlocation[i], cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 255, 0), 5)
                    print(ID[out[0]])
                    # print('a',a)
                # cv2.imshow('img',img)
            cv2.waitKey(10)


if __name__ == '__main__':
    time_start = time.time()
    test_withphoto('1.jpg')
    test_withcamera()
    time_end = time.time()
    print(time_end - time_start)