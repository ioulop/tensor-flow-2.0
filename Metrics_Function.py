'''
常用的评估函数：
回归：
* tf.keras.metrics.MeanSquaredError(平方差误差，用于回归，可以简写为MSE，函数形式为mse)
* tf.keras.metrics.MeanAbsoluteError(绝对值误差，用于回归，可以简写为MAE，函数形式为mae)
* tf.keras.metrics.MeanAbsolutePercentageError(平均百分比误差，用于回归，可以简写为MAPE，函数形式为mape)
* tf.keras.metrics.RootMeanSquaredError(均方根误差，用于回归)
分类：
* tf.keras.metrics.Accuracy(准确率，用于分类，可以使用字符串“Accuracy”表示，
Accuracy=(TP+TN)/(TP+TN+FP+FN),要求y_true和y_pred都为类别序号编码)
* tf.keras.metrics.AUC(ROC曲线(TPR vs FPR)下的面积，用于二分类，直观解释为随机抽取一个正样本和一个负样本，
正样本的预测值大于负样本的概率)
* tf.keras.metrics.Precision(精确率，用于二分类，Precision=TP/(TP+FP))
* tf.keras.metrics.Recall(召回率，用于二分类，Recall=TP/(TP+FN))
* tf.keras.metrics.TopKCategoricalAccuracy(多分类TopK准确率，
要求y_true(label)为onehot编码形式)
'''
import tensorflow as tf
import numpy as np
import matplotlib as plt

'''
自定义评估函数：
自定义评估函数需要继承tf.keras.metrics.Metrics类，并重写__init__,
update_state和result三个方法
* __init__():所有状态变量都应通过以下方法在此方法中创建self.add_weight()
* update_state():对状态变量进行所有更新
* result():根据状态变量计算并返回指标值】
'''
mnist = np.load('mnist.npt')
x_train, y_train, x_test, y_test = mnist['x_train'], mnist['y_train'], mnist['x_test'], mnist['y_test']
x_train, x_test = x_train/255.0, x_test/255.0

fig, ax = plt.subplots(
    nrows=2,
    nclos=5,
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
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


class MyModel(tf.keras.Model):
    def __int__(self):
        super(MyModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.d2 = tf.keras.layers.Dense(10,activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


class CategoricalTruePositives(tf.keras.metrics.Metric):
    def __int__(self, name='categorical_true_positives', **kwargs):
        super(CategoricalTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        values = tf.equal(tf.cast(y_true, 'int32'), tf.cast(y_pred, 'int32'))
        values = tf.cast(values, 'float32')
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, 'float32')
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_states(self):
        self.true_positives.assign(0.)





class SparseCategoricalAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='SparseCategoricalAccuracy', **kwargs):
        super(SparseCategoricalAccuracy, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', dtype=tf.int32, initializer=tf.zeros_initializer())
        self.count = self.add_weight(name='count', dtype=tf.int32, initializer=tf.zeros_initializer())

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = tf.cast(tf.equal(y_true, tf.argmax(y_pred, axis=-1, output_type=tf.int32)), tf.int32)
        self.total.assign_add(tf.shape(y_true)[0])
        self.count.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.count / self.total

    def reset_states(self):
        # the state of the metric will be reset at the start of each epoch
        self.total.assign(0)
        self.count.assign(0)

model = MyModel()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
train_tp = CategoricalTruePositives(name='train_tp')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
test_tp = CategoricalTruePositives(name='test_tp')

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)
    train_tp(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)
    test_tp(labels, predictions)


EPOCHS = 5
for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_accuracy.reset_states()
    train_tp.reset_states()
    test_loss.reset_states()
    test_accuracy.eset_states()
    test_tp.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = 'Epoch{}, Loss:{}, Accuracy:{}, TP:{}, Test Loss:{}, Test TP:{}'
    print(template.format(epoch+1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          train_tp.result(),
                          test_loss.result(),
                          test_accuracy.result() * 100,
                          test_tp.result()))

