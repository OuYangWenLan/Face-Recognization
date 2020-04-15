# from skimage import exposure, img_as_float, io
import tensorflow as tf
import sys

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2

sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import os
import time
import random
import matplotlib.pyplot as plt
import pickle
# 为了分割数据集需要导入train_test_split
from sklearn.model_selection import train_test_split

# 用于存放学号
ID = []


# 这里的path是指存放每个人照片的路径，也就是processed
# 根据测试，我们读取所有的照片需要28.3s-35
# 根据测试，图片和标签可以对上
# 归一化,将图片的像素变为0-255的区间范围，为了避免光照的影响

# 读取已经处理好的文件
def Readdataprocessed(images, label):
    for i in range(3):
        f1 = open('image' + str(i + 1) + '.txt', 'rb')
        f2 = open('label' + str(i + 1) + '.txt', 'rb')
        image_tmp = pickle.load(f1)
        label_tmp = pickle.load(f2)
        # 不喜欢用归一化之后的结果，还是就到255之间吧
        for j in range(len(image_tmp)):
            image_tmp[j] = 255 * image_tmp[j]
        images.extend(image_tmp)
        label.extend(label_tmp)
        f1.close()
        f2.close()
        print("done " + str(i + 1))


def Photoprocessed(path):
    # 生成人脸识别的模板
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    number = 0
    # 读取path路径下所有的图片
    for file in os.listdir(path):
        # 读取图片
        img = cv2.imread(file)
        # 转化为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 人脸检测接口
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(32, 32)
        )
        # 对检测到的人脸进行提取
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # 将人脸圈起来
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # one就是单纯人脸那一小块
                one = img[y:y + h, x:x + w]
                # 把人脸变成128*128的大小
                one = cv2.resize(one, (128, 128), interpolation=cv2.INTER_CUBIC)
                dest = np.zeros(one.shape, np.uint8)
                # 灰度转化
                dest = cv2.cvtColor(dest, cv2.COLOR_RGB2GRAY)
                # 加一个维度
                dest = dest[:, :, np.newaxis]
                # 调用自己写的single函数来进行归一化，把所有的照片归一化到0-255之间
                dest = single(dest)
                # 把修改之后的灰度图像写到图片中去
                cv2.imwrite(str(number + 1) + '.jpg', dest)
                number += 1


# 提取已经处理好的图片，先用Photoprocessed把彩色的图片提取成灰度图并且归一化，再用dataprocessed函数来把所有的图片信息存在一个数组里面
def dataprocessed(path, name):
    image = []
    for file in os.listdir(path):
        img = cv2.imread(file)
        image.append(img)
    f = open(str(name) + '.txt', 'rb')
    pickle.dump(image, f)


# 归一化函数
def single(img):
    image_single = img
    maxarx = np.argmax(img)
    i = int(maxarx / 128)
    j = maxarx % 128
    maxone = img[i][j]
    minarx = np.argmin(img)
    i = int(minarx / 128)
    j = minarx % 128
    minone = img[i][j]
    delta = 255 / (maxone - minone)
    for i in range(128):
        for j in range(128):
            image_single[i][j] = int((img[i][j] - minone) * delta)
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



def train():
    # 设定照片和标签
    images = []
    label = []
    # 读取已经处理好的照片，所有的图片都被提取出来，每张图片都有自己对应的标签
    Readdataprocessed(images=images, label=label)
    print("ReadData OK")

    # 把照片和标签都转化为数组
    images = np.array(images)
    label = np.array(label)

    # 将训练集和测试集以9:1的比例分开
    train_x, test_x, train_y, test_y = train_test_split(images, label, test_size=0.1,
                                                        random_state=random.randint(0, 100))
    print("split data successfully")

    # 每一批训练时的数量
    batch_size = 128
    test_size = 128
    # 获取一共训练或者测试多少批
    num_batch = len(train_x) // batch_size
    num_test = len(test_x) // test_size

    # [每一批训练的数量，照片高，照片宽，照片的通道数] img_as_float
    input = tf.compat.v1.placeholder(tf.float32, [None, 128, 128, 1])

    # [每一批训练的数量，标签的种类]
    truelabel = tf.compat.v1.placeholder(tf.float32, [None, 3])

    # 创建神经网络进行训练
    result = create_neural(input)
    print("create neural network successfully")

    # 定义优化器，这一块知道有这么一个优化器就行了，不需要深究
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=result, labels=truelabel))
    train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(result, 1), tf.argmax(truelabel, 1)), tf.float32))
    # 下面就是一些参数，不用去管
    tf.summary.scalar('loss', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
    # 记录训练次数和准确率，便于作图
    training_time = []
    accuracy_line = []
    # 网络的保存器的初始化
    saver = tf.train.Saver()
    print("Begin train")
    # 下面两行是配置信息，不用看
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.8  # 分配80%的计算资源

    # 下面开始训练了
    with tf.Session(config=tfconfig) as sess:
        # 初始化
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, './NET/train_faces100-58399')
        summary_writer = tf.summary.FileWriter('./tmp', graph=tf.get_default_graph())
        print("num_batch:", num_batch)
        # 开始正式的训练
        for n in range(200):
            # 每次取128(batch_size)张图片
            print("已经完成", n)
            for i in range(num_batch):
                # batch_x是训练的样本照片数据
                # batch_y是训练的样本标签数据
                # batch_testx是测试的照片数据
                # batch_testy是测试的标签数据
                batch_x = train_x[i * batch_size: (i + 1) * batch_size]
                batch_y = train_y[i * batch_size: (i + 1) * batch_size]
                batch_testx = test_x[(i % num_test) * test_size: ((i % num_test) + 1) * test_size]
                batch_testy = test_y[(i % num_test) * test_size: ((i % num_test) + 1) * test_size]

                # 开始训练数据，同时训练三个变量，返回三个数据
                _, loss, summary = sess.run([train_step, cross_entropy, merged_summary_op],
                                            feed_dict={input: batch_x, truelabel: batch_y, pro1: 0.5, pro2: 0.5,
                                                       pro3: 0.5, pro4: 0.5, pro5: 0.5, proconnect: 0.75})
                summary_writer.add_summary(summary, n * num_batch + i)
                # 打印损失
                # print('loss',n * num_batch + i, loss)
                # 获取测试数据的准确率，将测试数据代入网络中进行检测
                acc = accuracy.eval(
                    {input: batch_testx, truelabel: batch_testy, pro1: 1.0, pro2: 1.0, pro3: 1.0, pro4: 1.0, pro5: 1.0,
                     proconnect: 1.0})
                # print('result',sess.run(tf.argmax(result, 1) ,feed_dict={input: batch_x, truelabel: batch_y, pro1: 0.5, pro2: 0.5,
                #                                       pro3: 0.5, pro4: 0.5, pro5: 0.5, proconnect: 0.75}))
                # print('truelabel',sess.run(tf.argmax(truelabel, 1),feed_dict={input: batch_x, truelabel: batch_testy, pro1: 0.5, pro2: 0.5,
                #                                       pro3: 0.5, pro4: 0.5, pro5: 0.5, proconnect: 0.75}))
                training_time.append(n * num_batch + i)
                accuracy_line.append(acc)
                print('accuracy', n * num_batch + i, acc)
            # 每10次就存放一次模型
            if acc >= 1:
                saver.save(sess, 'train_faces_target')
                # break
            if (n + 1) % 50 == 0:
                saver.save(sess, 'train_faces' + str(n + 1), global_step=n * num_batch + i)
        plt.plot(training_time, accuracy_line)
        plt.xlabel("training number")
        plt.ylabel("accuracy")
        plt.show()



if __name__ == '__main__':
    time_start = time.time()
    train()
    time_end = time.time()
    print(time_end - time_start)