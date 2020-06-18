import tensorflow as tf
import numpy as np
# print(tf.__version__)
# print(tf.test.is_gpu_available())
# # 张量
# # rank0
# manmal = tf.Variable("Elemet", tf.string)
# tf.print(tf.rank(manmal))
#
# tf.print(tf.shape(manmal))
# print(tf.rank(manmal))
# print(tf.shape(manmal))
# # rank1
# mystr = tf.Variable([1], tf.int32)
# tf.print(tf.rank(mystr))
# tf.print(tf.shape(mystr))
# # rank2
# mymat = tf.Variable([[11], [7], [2]], tf.int16)
# tf.print(tf.rank(mymat))
# tf.print(tf.shape(mymat))
# # 创建张量
# tf.constant([1, 2, 3], dtype=tf.int16)
#
# '''
# 操作：
# tf.strings
# tf.debugging
# tf.dtypes
# tf.math
# tf.random
# tf.feature_column ：https://www.tensorflow.org/tutorials/structured_data/feature_columns
#                     常用于特征处理
# '''
# # tf.strings
# # 字符切割
# print(tf.strings.bytes_split("hello"))
# # 单词切割
# print(tf.strings.split("hello world"))
# # sstring hash
# print(tf.strings.to_hash_bucket(['hello', 'world'], num_buckets=10))
#
# # tf.debugging ：tf自带debug函数
# a = tf.random.uniform((10, 10))
# print(tf.debugging.assert_equal(x=a.shape, y=(10, 10)))
# # 错误示范
# # print(tf.debugging.assert_equal(x=a.shape, y=(20, 20)))
#
# # tf.random:神经网络必须会用到随机初始化，这个库必须学习
# a = tf.random.uniform(shape=(10, 5), minval=0, maxval=10)
# print(a)
#
# # tf.math
# a = tf.constant([[1, 2], [3, 4]])
# b = tf.constant([[5, 6], [7, 8]])
# tf.print(tf.math.add(a, b))
# tf.print(tf.math.subtract(a, b))
# tf.print(tf.math.multiply(a, b))
# tf.print(tf.math.divide(a, b))
#
# # tf.dtypes
# x = tf.constant([1.8, 2.2], dtype=tf.float32)
# x1 = tf.dtypes.cast(x, tf.int32)
# print(x1)
#
# '''
# 常用层：
# tf.keras.layers
# tf.nn
# 在大多数情况下，可以使用TensorFlow封装的tf.keras.layers构建的一些层建模，keras层是非常有用的
# 可以在文档中查看预先存在的层的完整列表，它包括Dense,Conv2D,LSTM,BatchNormalization,Dropout等
#
# '''
# # 定义文本
# a = tf.random.uniform(shape=(10, 100, 50), minval=-0.5, maxval=0.5)
# x = tf.keras.layers.LSTM(100)(a)
# x = tf.keras.layers.Dense(10)(x)
# x = tf.nn.softmax(x)
#
# # 增加层的配置参数
# # 增加层的激活函数
# tf.keras.layers.Dense(64, activation='relu')
# # or
# tf.keras.layers.Dense(64, activation=tf.nn.relu)

# 将L1正则化系数为0.01的线性层应用于内核矩阵
# tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))
# # 将L2正则化系数为0.01的线性层应用于变差向量:
# tf.keras.layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01))
# # 内核初始化为随机正交矩阵的线性层：
# tf.keras.layers.Dense(64, kernel_initializer='orthogonal')
# # 偏差向量初始化为2.0的线性层：
# tf.keras.layers.Dense(64, bias_initializer=tf.keras.initializers.Constant(2.0))

# 三种建模方式
# Sequential model 太简单，不推荐使用

# No1.Sequential model
from tensorflow.keras import layers

# model = tf.keras.Sequential()
# model.add(layers.Dense(64, activation='relu'))  # 第一层
# model.add(layers.Dense(64, activation='relu'))  # 第二层
# model.add(layers.Dense(10))  # 第三层

# No2.Sequential model
# model = tf.keras.Sequential([
#     layers.Dense(64, activation='relu', input_shape=(32,)),  # 第一层
#     layers.Dense(64, activation='relu'),  # 第二层
#     layers.Dense(10)  # 第3层
# ])
# model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
#               loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
#
# data = np.random.random((100, 32))
# labels = np.random.random((100, 10))
# model.fit(data, labels, epochs=10, batch_size=32)

'''
Functional model  推荐使用
函数式模型式一种创建模型的方法，该模型比tf.keras.Sequential更灵活，函数式模型可以处理具有非线性
拓扑的模型，具有共享层的模型以及具有多个输入和输出的模型等等
深度学习模型通常是层的有向无环图(DAG)的主要思想。因此，函数式模型是一种构建层图的方法
举个例子：(input:32-dimensional vectors) \$ [Dense(64 units,relu activation)] \$ [Sense(64 units,relu activation)]\$
[Dense(64 units,relu activation)] \$ [Dense(10 units,softmax activation)] \$ (output:lofits of a
probalitity distribution over 10 classes)
'''
# inputs1 = tf.keras.Input(shape=(32,))  # 输入1
# inputs2 = tf.keras.Input(shape=(32,))  # 输入2
# x1 = layers.Dense(64, activation='relu')(inputs1)  # 第一层
# x2 = layers.Dense(64, activation='relu')(inputs2)  # 第一层
# x = tf.concat([x1, x2], axis=-1)
# x = layers.Dense(64, activation='relu')(x)  # 第二层
# predictions = layers.Dense(10)(x)  # 第三层
# model = tf.keras.Model(inputs=[inputs1, inputs2], outputs=predictions)
# model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
#               loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
#
# data1 = np.random.random((1000, 32))
# data2 = np.random.random((1000, 32))
# labels = np.random.random((1000, 10))
# model.fit((data1, data2), labels, epochs=5, batch_size=32)


'''
子类化模型（重要！！！）
通过子类化tf.keras.Model和定义自己的前向传播模型来构建完全可制定的模型
和eager Execution模式相辅相成
'''
#
#
# class MyModel(tf.keras.Model):
#     def __init__(self, num_classes=10):
#         super(MyModel, self).__init__(name='my_model')
#         self.num_classed = num_classes
#         # 定义自己需要的层
#         self.dense_1 = layers.Dense(32, activation='relu')
#         self.dense_2 = layers.Dense(num_classes)
#
#     def call(self, inputs):
#         # 定义前向传播= forward
#         # 使用在(in __init__)定义的层
#         x = self.dense_1(inputs)
#         return self.dense_2(x)
#
#
# model = MyModel(num_classes=10)
# model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
#               loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
#
# data = np.random.random((1000, 32))
# labels = np.random.random((1000, 10))
# # Trains for 5 epochs
# model.fit(data, labels, epochs=5, batch_size=32)

'''
案例1 、keras版本训练模型
相关函数
* 构建模型(顺序模型，函数式模型，子类型模型)
* 模型训练:model.fit()
* 模型验证:model.evaluation()
* 模型预测:model.predict()

提供许多内置的优化器，损失和指标。通常，不必从头开始创建自己的损失，指标或优化函数，因为所需的可能已经是kerasAPI的一部分：
优化器：
* AGD()(有或没有动量)
* RMSprop()
* Adam()
损失：
* MeanSquaredError()
* KLDivergence()
* CrosineSimilarity()
指标：
* AUC()
* Precision()
* Recall()
另外，如果想用上述默认设置，那么在很多情况下，可以通过字符串标识符指定优化器，损失函数和指标
'''
# 1.1构建模型
# inputs = tf.keras.Input(shape=(32,))  # batch_size=32,数据维度也是32
# x = tf.keras.layers.Dense(64, activation='relu')(inputs)  # 64个神经元
# x = tf.keras.layers.Dense(64, activation='relu')(x)
# predictions = tf.keras.layers.Dense(10)(x)  # 输出是10类
# model = tf.keras.Model(inputs=inputs, outputs=predictions)
# model.compile(opitimizer=tf.keras.optimizers.Adam(0.001),
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
#
x_train = np.random.random((1000, 32))
y_train = np.random.randint(10, size=(1000,))
# x_val = np.random.random((200, 32))
# y_val = np.random.randint(10, size=(200,))
# x_test = np.random.random((200, 32))
# y_test = np.random.randint(10, size=(200,))
# model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_val, y_val))
# model.fit(x_train, y_train, batch_size=64, validation_split=0.2, epochs=1)
'''
1.3模型验证：返回test loss 和 metrics
'''
# Evaluate the model on the test data using evaluate
# print("\n # Evaluate on test data")
# results = model.evaluate(x_test, y_test, batch_size=128)
# print("test loss, test acc", results)

# Generate prediction (probabilities -- the output of the last layer)
# on new data using 'predict'
# print('\n # Generate predictions for 3 samples')
# predictions = model.predict(x_test[:3])
# print('prediction shape:', predictions.shape)

'''
案例2、使用样本加权和类别加权
除了输入数据和目标函数外，还可以在使用时将样本权重或者类权重传递给模型fit:
从Numpy数据进行训练时：通过sample_weight和class_weight参数。从数据集训练时：通过使用数据集返回一个元组(input_batch,target_batch,sample_weight_batch).
“样本权重”数组是一个数字数组，用于指定批次中每个样本在计算总损失时应具有的权重，它通常用于不平衡的分类问题中(这种想法是为很少见的班级赋予更多的权重)。当所使用的权重为1和0是，
该数组可用作损失函数的掩码(完全丢弃某些样本对总损失的贡献)。
“类别权重”字典是同一个概念的一个更具体的实例：它将类别索引映射到应该用于属于该类别的样本的样本权重，例如，如果在数据中类“0”的表示量少2倍，则可以使用class_weight={0:1, 1:0.5}
这是一个Numpy示例，其中我们使用类权重或者样本权重来更加重视第5类的正确分类

'''


# 2.1 构建模型
def get_uncompiled_model():
    inputs = tf.keras.Input(shape=(32,), name='digits')
    x = tf.keras.layers.Dense(64, activation='relu', name='dense_1')(inputs)
    x = tf.keras.layers.Dense(64, activation='relu', name='dense_2')(x)
    outputs = tf.keras.layers.Dense(10, name='predictions')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def get_compiled_model():
    model = get_uncompiled_model()
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['sparse_categorical_accuracy'])
    return model


# 2.2模型训练
# 类别5：加权

# 类别加权
# class_weight = {0: 1., 1: 1., 2: 1., 3: 1., 4: 1.,
#                 5: 2.,
#                 6: 1., 7: 1., 8: 1., 9: 1.}
# print('Fit with class weight')
# model = get_compiled_model()
# model.fit(x_train, y_train, class_weight=class_weight, batch_size=64, epochs=4)

# 样本加权(有一部分伪标签，权重小，真正的标签，权重为2)
# Here's the same example using 'sample_weight' instead:
# sample_weight = np.ones(shape=(len(y_train),))
# sample_weight[y_train == 5] = 2.
# print('\n Fit with sample weight')
#
# model = get_compiled_model()
# model.fit(x_train, y_train, sample_weight=sample_weight, batch_size=64, epochs=4)

'''
案例3、使用回调函数
keras中的回调是在训练期间(在某个时期开始时，在批处理结束时，在某个时期结束时等)在不同时间点调用的对象，这些对象可用于实现以下行为：
在训练过程中的不同时间点进行验证(除了内置的按时间段的验证)
定期或在超过特定精度阈值时对模型进行检查
当训练似乎停滞不前时，更改模型的学习率
当训练似乎停滞不前时，对顶层进行微调
在训练结束或超出特定性能阈值时发送电子邮件或即时消息通知等等，回调可以作为列表传递给model.fit:
3.1 EarlyStopping(早停)
* monitor:被检测的数据
* min_delta:在被检测的数据中被认为是提升的最小变化，例如，小于min_data的绝对变化会被认为没有提升
* patience:没有进步的训练轮数，在这之后训练就会被停止
* verbose:详细信息模式
* mode:{auto,min,max}其中之一，在min模式中，当被监测的数据停止下降，训练就会停止；早max模式中，当被监测的数据停止上升，训练就会停止；
  在auto模式中，方向会自动从被监测的数据的名字中判断出来

'''
# model = get_compiled_model()
# callbacks = [
#     tf.keras.callbacks.EarlyStopping(
#         # 当val_loss不在下降时，停止训练
#         monitor='val_loss',
#         # “不再下降”被定义为“减少不超过1e-2”
#         min_delta=1e-2,
#         # "不再改善" 进一步定义为 "至少2个epoch"
#         patience=2,
#         verbose=1)
# ]
#
# model.fit(x_train, y_train, epochs=20, batch_size=64, callbacks=callbacks,validation_split=0.2)

'''
许多内置的回调函数可用
* ModelCheckpoint:定期保存模型
* EarlyStopping:当训练不在改善验证指标时，停止训练
* TensorBoard：定期编写可在TensorBoard中可视化的模型日志（更多详细信息，请参见“可视化”部分）
* CSVKLogger:将损失和指标数据流式传输到CSV文件等

3.2checkpoint模型
在相对较大的数据集上训练模型时，至关重要的是要定期保存模型的checkpoint
最简单的方法是使用MedelCheckpoint回调：
'''
# model = get_compiled_model()
# callbacks = [
#     tf.keras.callbacks.ModelCheckpoint(
#         filepath='mymodel_(epoch)',
#         # 模型保存路径
#         # 下面的两个参数意味着当且仅当val_loss 分数提高时，我们才会覆盖当前检查点
#         save_best_only=True,
#         monitor='val_loss',
#         # 加入这个仅仅保存模型权重
#         save_weights_only=True,
#         verbose=1
#     )]
# model.fit(x_train, y_train, epochs=3, batch_size=64, callbacks=callbacks, validation_split=0.2)
'''
3.3使用回调现实动态学习率调整
由于优化程序无法访问验证指标，因此无法使用这些计划对象来实现动态学习率计划（例如，当验证损失不再改善降低学习率）
但是，回调确实可以访问所有指标，包括验证指标！因此，可以通过使用回调来修改优化程序上的当前学习率，从而实现此模式，实际上，它作为ReduceLROnPlateau回调内置的。

ReduceLROnPlateau参数
* monitor：被监测的指标
* factor：学习速率被降低的因数，新的学习速率=学习速率*因数
* patience：没有进步的训练轮数，在这之后训练速率会降低
* verbose：整数0：安静，1：更新信息
* mode：{auto,min,max}其中之一，如果是min模式，学习速率会被降低如果被监测的数据已经停止下降；在max模式，学习速率会被降低，如果被监测的数据已经停止上升，在auto模式
，方向会被从监测的数据中自动推断出来
* min_delta:衡量新的最佳阈值，仅关注重大变化
* cooldown：在学习速率被降低之后，重新恢复正常操作之前等待的训练轮数量
* min_lr：学习速率的下边界
'''
# model = get_compiled_model()
# callbacks = [
#     tf.keras.callbacks.ModelCheckpoint(
#         filepath='mymode_{epoch}',
#         # 模型保存路径
#         # 下面的两个参数意味着当且仅当val_loss分数提高时，我们才会覆盖当前检查点
#         save_best_only=True,
#         monitor='val_loss',
#         # 加入这个仅仅保存模型权重
#         save_weights_only=True,
#         verbose=1),
#     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_sparse_categorical_accuracy',
#                                          vernose=1,
#                                          mode='max',
#                                          factor=0.5,
#                                          patience=3)
#
# ]
# model.fit(x_train, y_train, epochs=3, callbacks=callbacks, batch_size=64, validation_split=0.2)

'''
案例4、将数据传递到多输入，多输出模型
在前面的实例中，我妈正在考虑一个具有单个输入(shape的张量(32,))和单个输出(shape的预测张量(10,))的模型，但是具有多个输入或输出的模型呢？
考虑以下模型，该模型具有形状的图像输入(32，32，3)(即height,width,channels)和形状的时间序列输入(None, 10)(即(timesteps,features))，
我们的模型将具有根据这些输入的组合计算出的两个输出："得分"(形状(1,1))和五类(形状(5,))的概率分布

'''
import pydot

image_input = tf.keras.Input(shape=(32, 32, 3), name='image_input')
timeseires_input = tf.keras.Input(shape=(20, 10), name='ts_Input')

x1 = tf.keras.layers.Conv2D(3, 3)(image_input)
x1 = tf.keras.layers.GlobalMaxPool2D()(x1)

x2 = tf.keras.layers.Conv1D(3, 3)(timeseires_input)
x2 = tf.keras.layers.GlobalMaxPool1D()(x2)

x = tf.keras.layers.concatenate([x1, x2])

score_output = tf.keras.layers.Dense(1, name='score_output')(x)
class_output = tf.keras.layers.Dense(5, name='class_output')(x)

model = tf.keras.Model(inputs=[image_input, timeseires_input], outputs=[score_output, class_output])
tf.keras.utils.plot_model(model, 'multi_input_and_output_model.png', show_shapes=True, dpi=500)
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(1e-3),
    loss=[tf.keras.losses.MeanSquaredError(),
          tf.keras.losses.CategoricalCrossentropy(from_lofits=True)]
)

# 损失函数不同，所以定义两个损失函数

'''
4.2指标函数
同样对于指标：
'''
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(1e-3),
    loss=[tf.keras.losses.MeanSquaredError(),
          tf.keras.losses.CategoricalCrossentropy(from_logits=True)],
    metrics=[[tf.keras.metrics.MeanAbsolutePercentageError(),
              tf.keras.metrics.MeanAbsoluteError()]
             [tf.keras.metrics.CategoricalAccuracy()]]
)
# 由于我们为输出层命名，因此我们还可以通过dict指定每个输出的损失和指标
# 如果有两个以上的输出，我们建议使用显示名称和字典
# 可以使用以下参数对不同的特定于输出的损失赋予不同的权重(例如，在我们的示例中，我们可能希望通过将某些损失函数赋予跟更高的权重)

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(1e-3),
    loss={
        'score_output': tf.keras.losses.MeanSquaredError(),
        'class_output': tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    },
    metrics={
        'score_output': [tf.keras.metrics.MeanAbsolutePercentageError(),
                         tf.keras.metrics.MeanAbsoluteError()],
        'class_optput': [tf.keras.metrics.CategoricalAccuracy()]
    }
)

# 联合训练
# loss_weights:损失函数加权
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(1e-3),
    loss={
        'score_output': tf.keras.losses.MeanSquaredError(),
        'class_output': tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    },
    metrics={
        'score_output': [tf.keras.metrics.MeanAbsolutePercentageError(),
                         tf.keras.metrics.MeanAbsoluteError()],
        'class_optput': [tf.keras.metrics.CategoricalAccuracy()]
    },
    loss_weights={'score_output': 2., 'class_output': 1.}
)

imag_data = np.random.random_sample(size=(100, 32, 32, 3))
ts_data = np.random.random_sample(size=(100, 20, 10))
score_targets = np.random.random_sample(size=(100, 1))
class_targets = np.random.random_sample(size=(100, 5))

# fit on lists
model.fit([imag_data, ts_data], [score_targets, class_targets],
          batch_size=32,
          epochs=3)
# alternatively ,fit dicts
model.fit({'img_input': imag_data, 'ts_input': ts_data},
          {'score_output': score_targets, 'class_output': class_targets},
          batch_size=32,
          epochs=3)
