# conding: utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

batch_size = 100

n_batch = mnist.train.num_examples // batch_size

# 参数概要
def variable_sunmmaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(var, mean))))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)  # 直方图


# 命名空间
with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, [None, 784], name="x-input")
    y = tf.placeholder(tf.float32, [None, 10], name="y-input")

with tf.name_scope("layer"):
    # 创建一个简单的神经网络
    with tf.name_scope('weights'):
        W = tf.Variable(tf.truncated_normal([784, 10]), name='W')
        variable_sunmmaries(W)
    with tf.name_scope('biass'):
        b = tf.Variable(tf.zeros([10]), name='bias')
        variable_sunmmaries(b)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x, W) + b

    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(wx_plus_b)

# 交叉熵代价函数
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=wx_plus_b))
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 计算准确度
with tf.name_scope('accuracy'):
    # 对比结果为布尔值组成的一维列表
    predict_compare = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(y,axis=1))
    # 计算准确率，通过一维列表
    accuracy = tf.reduce_mean(tf.cast(predict_compare, tf.float32))
# 初始化
init = tf.global_variables_initializer()

# 合并所有的summary
merged = tf.summary.merge_all()

# 模型主体结束，开始写会话
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/', sess.graph)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys})

        writer.add_summary(summary, global_step=epoch)
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
        print("Iter: " + str(epoch) + ", Test acc: " + str(acc))

# for i in range(2001):
#     #m每个批次100个样本
#     batch_xs,batch_ys = mnist.train.next_batch(100)
#     summary,_ = sess.run([merged,train_step],feed_dict={x:batch_xs,y:batch_ys})
#     writer.add_summary(summary,i)
#     if i%500 == 0:
#         print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))


