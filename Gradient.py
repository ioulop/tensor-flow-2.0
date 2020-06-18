# -*- encoding:utf-8 -*-
import tensorflow as tf
import numpy as np
print(tf.__version__)

'''
自动求导机制
tf.GradientTape详解
tf.GradientTape(
persistent=False,
watch_accessed_variables=True
)

* persistent:用来指定新创建的gradient tape 是否是可持续性的。默认是False，
意味着只能够调用一次gradient()函数
* watch_accessed_variables:表明这个GradientTape是不是会自动追踪任何能被训练(trainable)的变量
默认是True。要是为False的话，意味着你许哟啊手动去直指定你想追踪的那些变量

gradient(target, sources)
作用：根据tape上面的上下文来计算某些或者tensor的梯度参数：
target:被微分的Tensor,你可以理解为loss值(针对深度学习训练来说)
source:Tensor或者Variable列表(当然可以只有一个值)

返回：
一个列表表示各个变量的梯度值，和source中的变量列表一一对应，表明这个变量的梯度
'''
x = tf.constant(3.0)
with tf.GradientTape() as g:
    g.watch(x)  # watch作用：确保某个tensor被tape追踪
    y = x * x
# gradient作用：根据tape上面的上下文来计算某个或者某些tensor的梯度
dy_dx = g.gradient(y, x)
tf.print(dy_dx)

n = tf.constant(2.0)
with tf.GradientTape() as g:
    g.watch(n)
    y = n * n
dy_dn = g.gradient(y, n)
print(dy_dn)

'''
自定义模型训练:用到GradientTape
一般在网络中使用时，不需要显示调用watch函数，使用默认设置，GradientTape会监控可训练变量

'''
# input = tf.keras.Input()
# optimizer = tf.keras.optimizers.Adam()
# loss_object = tf.keras.losses.CategoricalCrossentropy()
# with tf.GradientTape() as g:
#     predictions = model(data)
#     loss = loss_object(labels, predictions)
# gradients = g.gradient(loss, model.trainable_variables)
# apply_gradient(grads_and_vars, name=None)
# 作用：吧计算出来的梯度更新到变量上面去
# 参数含义：grad_and_vars:(gradient, variable)对的列表
# name:操作名，可不写
# optimizer.apply_gradients(zip(gradients, model.trainable_variables))
'''
案例一。模型的自动求导
构建模型(神经网络的前向传播)-->定义损失函数-->定义优化函数-->定义tape-->
模型得到预测值-->前向传播得到loss-->反向传播-->用优化函数将计算出来的梯度更新到变量上面去

'''
# class MyModel(tf.keras.Model):
#     def __init__(self, num_classes=10):
#         super(MyModel, self).__init__(name='my_model')
#         self.num_class=num_classes
#         # 定义自己需要的层
#         self.dense_1 = tf.keras.layers.Dense(32, activation='relu')
#         self.dense_2 = tf.keras.layers.Dense(num_classes)
#
#     def call(self, inputs):
#         # 定义前向传播
#         # 使用在(in'__init__')定义的层
#         x = self.dense_1(inputs)
#         x = self.dense_2(x)
#         return x
#
#
# data = np.random.random((1000, 32))
# labels = np.random.random((1000, 10))
# model = MyModel(num_classes=10)
# loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
# optimizer = tf.keras.optimizers.Adam()
#
# with tf.GradientTape() as tape:
#     predictions = model.call(data)
#     loss = loss_object(predictions, labels)
# gradients = tape.gradient(loss, model.trainable_variables)
# optimizer.apply_gradients(zip(gradients, model.trainable_variables))
# print(model.trainable_variables)

'''
案例2 使用GradientTape自定义训练模型
'''
class MyModel(tf.keras.Model):
    def __init__(self, num_class=10):
        super(MyModel,self).__init__(name='my_model')
        self.dense_1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(10)
        self.num_class = num_class

    def call(self, inputs):
        x = self.dense_1(inputs)
        return self.dense_2(x)


loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.SGD(lr=1e-3)

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

model = MyModel(num_class=10)
batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((data, labels))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
epochs = 3
for epoch in range(epochs):
    print("start of epoch %d" % (epoch,))
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        # 每一个batch_size的操作
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss = loss_object(y_batch_train, logits)
        grad = tape.gradient(loss, model.trainable_weights)

        optimizer.apply_gradients(model.trainable_weights)
        # 每200轮打印一次
        if step % 200 == 0 :
            print('training loss (for one batch) at step %s' % (step, float(loss)))
            print('Seen so far :%s samples' % ((step+1) * 64))
