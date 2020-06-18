'''
自定义层：tf.keras.layers.Layer

tf.keras.Model和tf.keras.layers.Layer区别和联系
* 通过继承tf.keras.Model编写自己的模型类
* 通过继承tf.keras.layers.Layer编写自己的层
* tf.keras中的模型和层都是继承tf.Model实现的
* tf.keras.Model继承tf.keras.layers.Layer实现的

解释：tf.Model：定位为一个轻量级的状态容器，因为可以收集变量，所以这个类可以用来建模，配合
tf.GradientTape使用

三种自定义方法：
方法1：最基础方法

'''
import tensorflow as tf
from sklearn import datasets

iris = datasets.load_iris()
data = iris.data
target = iris.target
# class Linear(tf.keras.layers.Layer):
#     def __init__(self, units=1, input_dim=4):
#         super(Linear, self).__init__()
#         w_init = tf.random_normal_initializer()
#         self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units), dtype='float32'), trainable=True)
#         b_init = tf.zeros_initializer()
#         self.b = tf.Variable(initial_value=b_init(shape=(units,), dtype='float32'), trainable=True)
#     def call(self, inputs):
#         return tf.matmul(inputs, self.w) + self.b
#
#
# x = tf.constant(data)
# linear_layer = Linear(units=1, input_dim=4)
# y = linear_layer(x)
# print(y.shape)

# 方法2：
# class Linear(tf.keras.layers.Layer):
#     def __init__(self, units=1, input_dim=4):
#         super(Linear, self).__init__()
#         self.w = self.add_weight(shape=(input_dim, units),
#                                  initializer='random_normal',
#                                  trainable=True)
#         self.b = self.add_weight(shape=(units,),
#                                  initializer='zeros',
#                                  trainable=True)
#     def call(self, inputs):
#         return tf.matmul(inputs, self.w) + self.b
#
# x = tf.constant(data)
# linear_layer = Linear(units=1, input_dim=4)
# y = linear_layer(x)
# print(y.shape)

# 方法3:不知道输入维度
class Linear(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True,
                                 name='w')
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True,
                                 name='b')
        super(Linear, self).build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super(Linear, self).__init__()
        config.update({'units': self.units})
        return config
x = tf.constant(data)
linear_layer = Linear(units=1)
y = linear_layer(x)
print(y.shape)
print('weight', linear_layer.weights)  # 查看所有参数
print('non-trainable weight:', linear_layer.non_trainable_weights)  # 查看所有非训练的参数
print('trainable weight:', linear_layer.trainable_weights)  # 查看所有训练的参数

'''
自定义层的注意事项
注意一：这个问题是因为我们没有在自定义网络层时重写get_config导致的

解决方法：我们主要看传入的__init__接口时有哪些配置参数，然后再get_config内一一的将他们转为
字典键值并返回使用,下面举例
def get_config(self):
    config = super(Linear, self).get_config()
    config.update({'units':self.units})
    return config
get_config的作用：获取该层的参数配置，以便模型保存时使用

注意二：若模型保存(model.save)报错如下图时，则可能是自定义build中
创建初始矩阵时，name属性没写，会导致model.save报错
注意三：当我们自定义网络层并且有效保存模型后，希望使用tf.keras.models.load_model
进行模型加载时，可能会报如下错误：Unknown layer MyDense

解决方法：首先，建立一个字典，键是自定义网络层是设定该层的名字，其值为该自定义网络层的类名
该字典将用于加载模型时的使用！
然后，在tf.keras.models.load_model内传入custom_objects告知如何解析重建自定义网络层，其方法如下：
_custom_objects={
  "MyDense":MyDense,
}
tf.keras.models.load_model("keras_model_tf_version.h5", custom_objects=_custom_objects)
'''