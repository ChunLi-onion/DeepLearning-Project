import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# numpy建立数据样本
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]  # (200, 1),[:,np.newaxis] 添加一个新维度，一维（200，）变二维（200，1）
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

# 定义两个placeholder，为训练的样本输入
x = tf.placeholder(tf.float32, shape=[None, 1])  # 样本个数不确定，用None代表，列数确定为“1”
y = tf.placeholder(tf.float32, shape=[None, 1])

# 定义神经网络中间层
W1 = tf.Variable(tf.random_normal([1, 10]))  # 第一层weights 初始化为 （1，10） 十个单元
b1 = tf.Variable(tf.zeros([1, 10]))  # bias 初始化为0
Z1 = tf.matmul(x, W1) + b1
A1 = tf.nn.tanh(Z1, name='A1')

# 定义神经网络输出层
W2 = tf.Variable(tf.random_normal([10, 1]))  # 第二层weights 为一个单元 （10，1）
b2 = tf.Variable(tf.zeros([1, 1]))
Z1 = tf.matmul(A1, W2) + b2
prediction = tf.nn.tanh(Z1, name='prediction')

# 定义代价函数
loss = tf.reduce_mean(tf.square(y - prediction))

# 指定梯度下降法来优化训练
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.global_variables_initializer()
# 建立会话
with tf.Session() as sess:
    sess.run(init)  # 变量初始化
    plt.figure()
    for _ in range(4000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})

    # 获得预测值
    prediction_value = sess.run(prediction, feed_dict={x: x_data})  # prediction操作不需要feed：y， loss和train需要y
    # 画图
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, 'r-', lw=5)
    plt.show()
