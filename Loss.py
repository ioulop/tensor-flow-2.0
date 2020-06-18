'''
常用的损失函数
tf.keras.losses
* mean_squared_error(平方差误差损失，用于回归，简写MSE，
类实现的形式为MeanSquaredError和MSE)
* binary_crossentropy(二元交叉熵，用于二分类，类实现形式为：BinaryCrossentropy)
* categorical_crossentropy(类别交叉熵，用于多分类，要求label为onehot编码，
类实现形式为CategoricalCrossentropy)
* sparse_categorical_crossentropy(稀疏类别交叉熵，用于多分类，要求label为序号编码形式类，
类实现形式为SparseCategoricalCrossentropy)
自定义损失函数
两种方式实现：类的方式，函数的方式
'''


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import numpy as np

mnist = np.load('mnist.npz')
x_train, y_train, x_test, y_test = mnist['x_train'], mnist['y_train'], mnist['x_test'], mnist['y_test']
x_train, x_test = x_train / 255.0, x_test / 255.0

import matplotlib.pyplot as plt
fig, ax = plt.subplots(
    nrows=2,
    ncols=5,
    sharex=True,
    sharey=True,)
ax = ax.flatten()
for i in range(10):
    img = x_train[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
# add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
y_train = tf.one_hot(y_train)
y_test = tf.one_hot(y_test)
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 类的实现
class MyModel(Model):
    def __int__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return  self.d2(x)

# 多分类的focal loss 损失函数
class FocalLoss(tf.keras.losses.Loss):
    def __int__(self, gamma=2.0, alpha=0.25):
        self.gamma = gamma
        self.alpha = alpha
        super(FocalLoss, self).__init__()

    def call(self, y_true, y_pred):
        y_pred = tf.nn.softmax(y_pred, axis=-1)
        epsilon = tf.keras.backend.epsilon()  # 非常小的数，把数据归到一个区间
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)

        y_true = tf.cast(y_true, tf.float32)
        loss = y_true * tf.math, pow(1-y_pred, self.gamma) * tf.math.log(y_pred)
        loss = tf.math.reduce_sum(loss, axis=1)
        return loss


model = MyModel()
loss_object = FocalLoss(gamma=2.0, alpha=0.25)
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

epochs = 5
for epoch in range(epochs):
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    for test_image, test_labels in test_ds:
        test_step(test_image, test_labels)

    template = 'Epoch{}, Loss:{}, Accuracy:{}, Test Loss:{}, Test:Accuracy:{}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))

