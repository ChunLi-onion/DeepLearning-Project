import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
minist = input_data.read_data_sets("MNIST_data", one_hot=True)

# mini_batch 的每个批次大小
batch_size = 500
# 计算一共有多少批次
n_batch = minist.train.num_examples // batch_size

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])  # 784 = 28 * 28, 压平图片两个维度后变为一个维度
y = tf.placeholder(tf.float32, [None, 10])  # one-hot之后10个类别变为10个维度，只有一个为1，其余均为0

# 创建一个神经网络
W1 = tf.Variable(tf.random_normal([784, 30]))  # L1 units个数减少会影响准确率更低
b1 = tf.Variable(tf.random_normal([30]))
A1 = tf.nn.relu(tf.matmul(x, W1) + b1)

W2 = tf.Variable(tf.random_normal([30, 10]))
b2 = tf.Variable(tf.random_normal([10]))
prediction = tf.nn.softmax(tf.matmul(A1, W2) + b2)

# 代价函数
loss = tf.reduce_logsumexp(tf.square(y - prediction))  # reduce_log 似乎比reduce_mean更快更准确些

# 使用Adam来训练
train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

# 预测和真值的对比结果存放在一维列表中, 元素为布尔值
correct_prediction = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(y, axis=1))  # 相等为True，不等为false，输出一个一维列表
# 计算准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # tf.cast转换格式，把布尔值true->1, false->0

# 初始化变量
init = tf.global_variables_initializer()

# 创建会话
with tf.Session() as sess:
    sess.run(init)
    # 全部batch训练41次
    for epoch in range(41):
        for _ in range(n_batch):
            # minist.train.next_batch 自动取下一组batch数据，(0,100) ->(100,200)
            batch_xs, batch_ys = minist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
        acc = sess.run(accuracy, feed_dict={x: minist.test.images, y: minist.test.labels})
        print("Iteraction" + str(epoch) + ": accuracy " + str(acc))