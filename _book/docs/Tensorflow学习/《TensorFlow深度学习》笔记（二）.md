# 《TensorFlow深度学习》笔记（二）

## 反向传播算法 

### 激活函数

- ```python
  import numpy as np # 导入 numpy 库 
  def sigmoid(x): # 实现 sigmoid 函数     
      return 1 / (1 + np.exp(-x)) 
   
  def derivative(x):  # sigmoid 导数的计算     
      # sigmoid 函数的表达式由手动推导而得 
      return sigmoid(x)*(1-sigmoid(x))
  
  ##############################################################################
  def derivative(x):  # ReLU 函数的导数     
      d = np.array(x, copy=True)  # 用于保存梯度的张量     
      d[x < 0] = 0  # 元素为负的导数为 0     
      d[x >= 0] = 1  # 元素为正的导数为 1     
      return d 
   
  ##############################################################################
   # 其中 p 为 LeakyReLU 的负半段斜率，为超参数 def derivative(x, p): 
      dx = np.ones_like(x)  # 创建梯度张量，全部初始化为 1     
      dx[x < 0] = p  # 元素为负的导数为 p 
      return dx 
  ##############################################################################
  def sigmoid(x):  # sigmoid 函数实现     return 1 / (1 + np.exp(-x))  
  def tanh(x):  # tanh 函数实现     return 2*sigmoid(2*x) - 1 
  def derivative(x):  # tanh 导数实现 
      return 1-tanh(x)**2 
   
  ```

## Keras 高层接口 

### 常见网络层类 

- ```python
  # 在 tf.keras.layers 命名空间(下文使用 layers 指代 tf.keras.layers)中提供了大量常见网络层的类，如全连接层、激活函数层、池化层、卷积层、循环神经网络层等。对于这些网络 层类，只需要在创建时指定网络层的相关参数，并调用__call__方法即可完成前向计算。在 调用__call__方法时，Keras 会自动调用每个层的前向传播逻辑，这些逻辑一般实现在类的 call 函数中.
  # 以 Softmax 层为例:
  
  import tensorflow as tf 
  # 导入 keras 模型，不能使用 import keras，它导入的是标准的 Keras 库 
  from tensorflow import keras 
  from tensorflow.keras import layers # 导入常见网络层类 
  
  # 创建 Softmax 层，并调用__call__方法完成前向计算
  x = tf.constant([2.,1.,0.1])  # 创建输入张量 
  layer = layers.Softmax(axis=-1)  # 创建 Softmax 层 
  out = layer(x)  # 调用 softmax 前向计算，输出为 out
  ```

### 网络容器

- 通过 Keras 提供的网络容器 Sequential 将多个 网络层封装成一个大网络模型，只需要调用网络模型的实例一次即可完成数据从第一层到 最末层的顺序传播运算。 

- ```python
  # 例如，2 层的全连接层加上单独的激活函数层，可以通过 Sequential 容器封装为一个网络。 
  # 导入 Sequential 容器  
  from tensorflow.keras import layers, Sequential 
  network = Sequential([ # 封装为一个网络     
      layers.Dense(3, activation=None),  # 全连接层，此处不使用激活函数 
      layers.ReLU(), # 激活函数层     
      layers.Dense(2, activation=None), # 全连接层，此处不使用激活函数
      layers.ReLU() # 激活函数层 
  ]) 
  x = tf.random.normal([4,3]) 
  out = network(x) # 输入从第一层开始，逐层传播至输出层，并返回输出层的输出 
  
  ##############################################################################
  # Sequential 容器也可以通过 add()方法继续追加新的网络层，实现动态创建网络的功能： 
  layers_num = 2 # 堆叠 2 次 
  network = Sequential([]) # 先创建空的网络容器 
  for _ in range(layers_num): 
      network.add(layers.Dense(3)) # 添加全连接层 
      network.add(layers.ReLU())# 添加激活函数层 
  network.build(input_shape=(4, 4)) # 创建网络参数 
  network.summary()
  # 上述代码通过指定任意的 layers_num 参数即可创建对应层数的网络结构，在完成网络创建 时，网络层类并没有创建内部权值张量等成员变量，此时通过调用类的 build 方法并指定 输入大小，即可自动创建所有层的内部张量。通过 summary()函数可以方便打印出网络结 构和参数量
  
  ##############################################################################
  # Sequential 对象的 trainable_variables 和 variables 包含了所有层的待优化张量列表 和全部张量列表
  
   # 打印网络的待优化参数名与 shape 
  for p in network.trainable_variables: 
  print(p.name, p.shape) # 参数名和形状 
  ```

### 模型装配、训练与测试 

- 模型装配 ：

  - 在 Keras 中，有 2 个比较特殊的类：keras.Model 和 keras.layers.Layer 类。其中 Layer 类是网络层的母类，定义了网络层的一些常见功能，如添加权值、管理权值列表等。 Model 类是网络的母类，除了具有 Layer 类的功能，还添加了保存模型、加载模型、训练 与测试模型等便捷功能。Sequential 也是 Model 的子类，因此具有 Model 类的所有功能。

  -  ```python
    # MNIST 手写数字图片识
    # 创建 5 层的全连接网络 
    network = Sequential([layers.Dense(256, activation='relu'),                      layers.Dense(128, activation='relu'),                              	 layers.Dense(64, activation='relu'),                      
     	 layers.Dense(32, activation='relu'),                      				 layers.Dense(10)]) 
    network.build(input_shape=(4, 28*28)) 
    network.summary() 
    
    # 创建网络后，正常的流程是循环迭代数据集多个 Epoch，每次按批产生训练数据、前向计 算，然后通过损失函数计算误差值，并反向传播自动计算梯度、更新网络参数。这一部分 逻辑由于非常通用，在 Keras 中提供了 compile()和 fit()函数方便实现上述逻辑。首先通过 compile 函数指定网络使用的优化器对象、损失函数类型，评价指标等设定，这一步称为装配
    。
    # 导入优化器，损失函数模块 
    from tensorflow.keras import optimizers,losses  
    # 模型装配 
    # 采用 Adam 优化器，学习率为 0.01;采用交叉熵损失函数包含Softmax 
    network.compile(optimizer=optimizers.Adam(lr=0.01),         loss=losses.CategoricalCrossentropy(from_logits=True), 
            metrics=['accuracy'] # 设置测量指标为准确率 
    ) 
    
    ```

- 模型训练 

  - ```python
    # 模型装配完成后，即可通过 fit()函数送入待训练的数据集和验证用的数据集，这一步 称为模型训练。
    # 指定训练集为 train_db，验证集为 val_db,训练 5 个 epochs，每 2 个 epoch 验证一次 # 返回训练轨迹信息保存在 history 对象中 
    history = network.fit(train_db, epochs=5, validation_data=val_db, validation_freq=2) 
    # 运行上述代码即可实现网络的训练与验证的功能，fit 函数会返回训练过程的数据记录 history，其中 history.history 为字典对象，包含了训练过程中的 loss、测量指标等记录项， 我们可以直接查看这些训练数据
    history.history # 打印训练记录
    
    ```
  ```python
  # MNIST 手写数字图片识
  # 创建 5 层的全连接网络 
  network = Sequential([layers.Dense(256, activation='relu'),                      layers.Dense(128, activation='relu'),                              	 layers.Dense(64, activation='relu'),                      
   	 layers.Dense(32, activation='relu'),                      				 layers.Dense(10)]) 
  network.build(input_shape=(4, 28*28)) 
  network.summary() 
  
  #  创建网络后，正常的流程是循环迭代数据集多个 Epoch，每次按批产生训练数据、前  向计 算，然后通过损失函数计算误差值，并反向传播自动计算梯度、更新网络参数。这一  部分 逻辑由于非常通用，在 Keras 中提供了 compile()和 fit()函数方便实现上述逻辑。首先通过 compile 函数指定网络使用的优化器对象、损失函数类型，评价指标等设定，这一步称为装配。
  # 导入优化器，损失函数模块 
  from tensorflow.keras import optimizers,losses  
  # 模型装配 
  # 采用 Adam 优化器，学习率为 0.01;采用交叉熵损失函数包含Softmax 
  network.compile(optimizer=optimizers.Adam(lr=0.01),         loss=losses.CategoricalCrossentropy(from_logits=True), 
          metrics=['accuracy'] # 设置测量指标为准确率 
  ) 
  ```
- 模型测试 

  - Model 基类除了可以便捷地完成网络的装配与训练、验证，还可以非常方便的预测和 测试。

  - ```python
    # 通过 Model.predict(x)方法即可完成模型的预测
    # 加载一个 batch 的测试数据 
    x,y = next(iter(db_test)) print('predict x:', x.shape) # 打印当前 batch 的形状 
    out = network.predict(x) # 模型预测，预测结果保存在 out 中 
    print(out)
    ```

### 模型保存与加载 

- 张量方式

- ```python
  # Model.save_weights(path)方法即可将当前的 网络参数保存到 path 文件上，代码如下： 
  network.save_weights('weights.ckpt') # 保存模型的所有张量数据 
  # 上述代码将 network 模型保存到 weights.ckpt 文件上。
  
  # 例子：
  # 保存模型参数到文件上 
  network.save_weights('weights.ckpt') print('saved weights.') 
  del network # 删除网络对象 
  # 重新创建相同的网络结构 
  network = Sequential([layers.Dense(256, activation='relu'),                      layers.Dense(128, activation='relu'),                      
     layers.Dense(64, activation='relu'),                           
     layers.Dense(32, activation='relu'),                      
      layers.Dense(10)])
  network.compile(optimizer=optimizers.Adam(lr=0.01),         						loss=tf.losses.CategoricalCrossentropy(from_logits=True),        
          metrics=['accuracy'])  
  # 从参数文件中读取数据并写入当前网络 
  network.load_weights('weights.ckpt') 
  print('loaded weights!') 
  ```

- 网络方式 

  - 通过 Model.save(path) 函数将模型的结构及其参数保存到path路径
  - keras.models.load_model(path)  函数将模型的结构及其参数恢复

-  SavedModel 方式 

  - ```python
    # 保存模型结构与模型参数到文件 
    tf.saved_model.save(network, 'model-savedmodel') 
    print('saving savedmodel.') 
    del network # 删除网络对象
    print('load savedmodel from file.') # 从文件恢复网络结构与网络参数 
    network =  tf.saved_model.load('model-savedmodel') # 准确率计量器  
    acc_meter = metrics.CategoricalAccuracy()  for x,y in ds_val:  # 遍历测试集
        pred = network(x) # 前向计算     
        acc_meter.update_state(y_true=y, y_pred=pred) # 更新准确率统计
        # 打印准确率 
    print("Test Accuracy:%f" % acc_meter.result())
    ```

### 自定义网络层

- ```python
  # 通过 Sequential 容器方便地封装成一个网络模 型
  network = Sequential([
  MyDense(784, 256), # 使用自定义的层      
  MyDense(256, 128),                      
  MyDense(128, 64),                     
  MyDense(64, 32),                     
  MyDense(32, 10)])
  network.build(input_shape=(None, 28*28)) 
  network.summary() 
  
  # 来创建自定义网络类
  class MyModel(keras.Model): 
      # 自定义网络类，继承自 Model 基类     
      def __init__(self):         
          super(MyModel, self).__init__() 
          # 完成网络内需要的网络层的创建工作         
          self.fc1 = MyDense(28*28, 256)         
          self.fc2 = MyDense(256, 128)         
          self.fc3 = MyDense(128, 64) 
          self.fc4 = MyDense(64, 32) 
          self.fc5 = MyDense(32, 10) 
          
   # 实现自定义网络的前向运算逻
   def call(self, inputs, training=None): 
          # 自定义前向运算逻辑         
          x = self.fc1(inputs)          
          x = self.fc2(x)          
          x = self.fc3(x)          
          x = self.fc4(x)          
          x = self.fc5(x)  
          return x 
  
  ```

### 测量工具 

- Keras 的测量工具的使用方法一般有 4 个主要步骤：新建测量器，写入数据，读取统 计数据和清零测量器。 

- ```python
  # 统计误差值
  #  新建测量器 ,在 keras.metrics 模块中，提供了较多的常用测量器类，如统计平均值的 Mean 类，统 计准确率的 Accuracy 类，统计余弦相似度的 CosineSimilarity 类等
  # 新建平均测量器，适合 Loss 数据 
  loss_meter = metrics.Mean() 
  
  # 写入数据 
  # 记录采样的数据，通过 float()函数将张量转换为普通数值 
  loss_meter.update_state(float(loss)) 
  
  # 读取统计信息 
  # 打印统计期间的平均 loss 
  print(step, 'loss:', loss_meter.result()) 
  
  #  清除状态 
   if step % 100 == 0: 
          # 打印统计的平均 loss         
          print(step, 'loss:', loss_meter.result())  
          loss_meter.reset_states() # 打印完后，清零测量器 
          
   ####################################################################
  
  # 统计准确率
  acc_meter = metrics.Accuracy() # 创建准确率测量器 
  
  # 。需要注意的是，Accuracy 类的 update_state 函数的参数为预测值和真实值，而不是当前 Batch 的准确率。
   # [b, 784] => [b, 10]，网络输出值             
   out = network(x)  
   # [b, 10] => [b]，经过 argmax 后计算预测值             
   pred = tf.argmax(out, axis=1)              
   pred = tf.cast(pred, dtype=tf.int32) 
   # 根据预测值与真实值写入测量器 
   acc_meter.update_state(y, pred)
   # 读取统计结果        
   print(step, 'Evaluate Acc:', 
   acc_meter.result().numpy()) 
   acc_meter.reset_states() # 清零测量器 
  
  ```

##  过拟合 

-  Early stopping     正则化       Dropout      

- 数据增强（ 旋转 ， 翻转 ，裁剪 ）

  









