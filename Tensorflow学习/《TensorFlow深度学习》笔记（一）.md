#                   《TensorFlow深度学习》笔记（一）

## TensorFlow 进阶 

### 最值、均值、和 

- 通过 tf.reduce_max、tf.reduce_min、tf.reduce_mean、tf.reduce_sum 函数可以求解张量 在某个维度上的最大、最小、均值、和

### 张量比较 

- 为了计算分类任务的准确率等指标，一般需要将预测结果和真实标签比较，统计比较 结果中正确的数量来计算准确率，通过 tf.argmax 获取预测类别

### 经典数据集加载 

- 在 TensorFlow 中，keras.datasets 模块提供了常用经典数据集的自动下载、管理、加载 与转换功能，并且提供了 tf.data.Dataset 数据集对象

- 通过 datasets.xxx.load_data()函数即可实现经典数据集的自动加载，其中 xxx 代表具体 的数据集名称，如“CIFAR10”、“MNIST”

- ```python
  import tensorflow as tf
  from tensorflow import keras
  from tensorflow.keras import datasets  # 导入经典数据集加载模块
  
  # 加载 MNIST 数据集
  (x, y), (x_test, y_test) = datasets.mnist.load_data()
  print('x:', x.shape, 'y:', y.shape, 'x test:', x_test.shape, 'y test:',y_test)
  ```

- 数据加载进入内存后，需要转换成 Dataset 对象，才能利用 TensorFlow 提供的各种便捷功能。通过 Dataset.from_tensor_slices 可以将训练部分的数据图片 x 和标签 y 都转换成 Dataset 对象,  将数据转换成 Dataset 对象后，一般需要再添加一系列的数据集标准处理步骤，如随机打 散、预处理、按批装载等

- ```python
  train_db = tf.data.Dataset.from_tensor_slices((x, y)) # 构建 Dataset 对象 
  ----------------------------------------------------------
  
  train_db = train_db.shuffle(10000) # 随机打散样本，不会打乱样本与标签映射关系 
  
  --------------------------------------------------------
  
  # 为了利用显卡的并行计算能力，一般在网络的计算过程中会同时计算多个样本，我们 把这种训# 练方式叫做批训练，其中一个批中样本的数量叫做 Batch Size
  train_db = train_db.batch(128) # 设置批训练，batch size 为 128 
  
  --------------------------------------------------------
  
  # 预处理函数实现在 preprocess 函数中，传入函数名即可 
  train_db = train_db.map(preprocess)
  # 下面是自定义预处理函数
  def preprocess(x, y): # 自定义的预处理函数 
  # 调用此函数时会自动传入 x,y 对象，shape 为[b, 28, 28], [b]  
      # 标准化到 0~1 
      x = tf.cast(x, dtype=tf.float32) / 255. 
      x = tf.reshape(x, [-1, 28*28])     # 打平 
      y = tf.cast(y, dtype=tf.int32)    # 转成整型张量 
      y = tf.one_hot(y, depth=10)    # one-hot 编码 
      # 返回的 x,y 将替换传入的 x,y 参数，从而实现数据的预处理功能 
      return x,y  
  --------------------------------------------------------
  #  循环训练 
  # 当对 train_db 的所有样本完 成一次迭代后，for 循环终止退出。这样完成一个 Batch 的  
  # 数据训练，叫做一个 Step；通过 多个 step 来完成整个训练集的一次迭代，叫做一个 Epoch
  
  # 对于 Dataset 对象，在使用时可以通过 
     for step, (x,y) in enumerate(train_db): # 迭代数据集对象，带 step 参数 
  # 或 
      for x,y in train_db: # 迭代数据集对象 
  # 例如：训练20个Epoch
   for epoch in range(20): # 训练 Epoch 数 
          for step, (x,y) in enumerate(train_db): # 迭代 Step 数 
              # training... 
    
  --------------------------------------------------------    
  
  # 在训练的过程中，通过间隔数个 Step 后打印误差数据，可以有效监督模型的训练 进度，代码# 如下： 
          # 间隔 100 个 step 打印一次训练误差 
          if step % 100 == 0: 
              print(step, 'loss:', float(loss)) 
   
  ```

## 神经网络 

### 感知机

- 感知机模型的结构=它接受长度为𝑛的一维向量𝒙 = [𝑥1,𝑥2,…,𝑥𝑛]，每个 输入节点通过权值为𝑤𝑖,𝑖𝜖[1,𝑛]的连接汇集为变量𝑧，即：

 						𝑧 = 𝑤1 𝑥1 + 𝑤2 𝑥2 + ⋯+ 𝑤𝑛 𝑥𝑛 + 𝑏 

​	   其中𝑏称为感知机的偏置(Bias)，一维向量𝒘 = [𝑤1,𝑤2,…,𝑤𝑛]称为感知机的权值(Weight)，𝑧 称为	   感知机的净活性值(Net Activation)。 

​        上式写成向量形式： 
​								𝑧 = 𝒘T𝒙+ 𝑏 

### 全连接层(即：神经网络 的一层)

- 张量方式实现 

- ```python
  # 创建 W,b 张量 
  x = tf.random.normal([2,784]) 
  w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1)) 
  b1 = tf.Variable(tf.zeros([256])) 
  o1 = tf.matmul(x,w1) + b1  # 线性变换 
  o1 = tf.nn.relu(o1)  # 激活函数 
  ```

- 层方式实现 

- ```python
  x = tf.random.normal([4,28*28]) 
  from    tensorflow.keras import  layers # 导入层模块 
  # 创建全连接层，指定输出节点数和激活函数 
  fc = layers.Dense(512, activation=tf.nn.relu)  
  h1 = fc(x)  # 通过 fc 类实例完成一次全连接层的计算，返回输出张量 
  
  ########################################################################   fc.kernel # 获取 Dense 类的权值矩阵 
  fc.bias # 获取 Dense 类的偏置向量 
  fc.trainable_variables  # 返回待优化参数列表 
  fc.variables # 返回所有参数列表 
  ```

### 神经网络 

-  张量方式实现 

- ```python
  x = tf.random.normal([4,28*28]) 
  from    tensorflow.keras import  layers # 导入层模块 
  
  # 隐藏层 1 张量 
  w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1)) b1 = tf.Variable(tf.zeros([256])) 
  # 隐藏层 2 张量 
  w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1)) 
  b2 = tf.Variable(tf.zeros([128])) 
  # 隐藏层 3 张量 
  w3 = tf.Variable(tf.random.truncated_normal([128, 64], stddev=0.1)) b3 = tf.Variable(tf.zeros([64])) 
  # 输出层张量 
  w4 = tf.Variable(tf.random.truncated_normal([64, 10], stddev=0.1)) 
  b4 = tf.Variable(tf.zeros([10])) 
  
  # 在使用 TensorFlow 自动求导功能计算梯度时，需要将前向计算过程放置在 tf.GradientTape()环境中，从而利用 GradientTape 对象的 gradient()方法自动求解参数的梯 度，并利用 optimizers 对象更新参数。 
  with tf.GradientTape() as tape: # 梯度记录器 
       #  隐藏层 1 前向计算，[b, 28*28] => [b, 256]             
        h1 = x@w1 + tf.broadcast_to(b1, [x.shape[0], 256])             
        h1 = tf.nn.relu(h1) 
       # 隐藏层 2 前向计算，[b, 256] => [b, 128]             
        h2 = h1@w2 + b2             
        h2 = tf.nn.relu(h2) 
        # 隐藏层 3 前向计算，[b, 128] => [b, 64]              
        h3 = h2@w3 + b3             
        h3 = tf.nn.relu(h3) 
        # 输出层前向计算，[b, 64] => [b, 10]              
        h4 = h3@w4 + b4 
  ```

-  层方式实现 

- ```python
  #  对于这种数据依次向前传播的网络，也可以通过 Sequential 容器封装成一个网络大类对象
  #  ，调用大类的前向计算函数一次即可完成所有层的前向计算，使用起来更加方便，
  #  实现如下：
  
  import tensorflow as tf
  #   导入 Sequential 容器
  from tensorflow.keras import layers,Sequential
  x = tf.random.normal([4,28*28])
  #  通过 Sequential 容器封装为一个网络类
  model = Sequential([
  layers.Dense(256, activation=tf.nn.relu),  # 创建隐藏层 1
  layers.Dense(128, activation=tf.nn.relu) , # 创建隐藏层 2
  layers.Dense(64, activation=tf.nn.relu) , # 创建隐藏层 3
  layers.Dense(10, activation=None) ,   # 创建输出层
  ]);
  
  out = model(x) # 前向计算得到输出
  
  
  ```

###  优化目标 

- 我们把神经网络从输入到输出的计算过程叫做前向传播(Forward Propagation)
- 上述的最小化优化问题一般采用误差反向传播(Backward Propagation，简称 BP)算法来求解 网络参数𝜃的梯度信息，并利用梯度下降(Gradient Descent，简称 GD)算法

###  输出层设计 

- 我们将根据输出值的区间范围来分类讨论。常见的几种输出类型包括：  

  - 𝑜𝑖 ∈ 𝑅𝑑 输出属于整个实数空间，或者某段普通的实数空间，比如函数值趋势的预 测，年龄的预测问题等。 

  - 𝑜𝑖 ∈ [0,1] 输出值特别地落在[0,1]的区间，如图片生成，图片像素值一般用[0,1]区间 的值表示；或者二分类问题的概率，如硬币正反面的概率预测问题。 

  - 𝑜𝑖 ∈ [0, 1],  𝑜𝑖 𝑖 = 1 输出值落在[0,1]的区间，并且所有输出值之和为 1，常见的如 多分类问题，如 MNIST 手写数字图片识别，图片属于 10 个类别的概率之和应为 1。 

    - ```python
      z = tf.random.normal([2,10]) # 构造输出层的输出 
      y_onehot = tf.constant([1,3]) # 构造真实值 
      y_onehot = tf.one_hot(y_onehot, depth=10) # one-hot 编码 
      # 输出层未使用 Softmax 函数，故 from_logits 设置为 True 
      # 这样 categorical_crossentropy 函数在计算损失函数前，会先内部调用 Softmax 函数 
      loss = keras.losses.categorical_crossentropy(y_onehot,z,from_logits=True) 
      loss = tf.reduce_mean(loss) # 计算平均交叉熵损失 
      criteon = keras.losses.CategoricalCrossentropy(from_logits=True) 
      loss = criteon(y_onehot,z) # 计算损失 
      ```

  -  𝑜𝑖 ∈ [−1, 1] 输出值在[-1, 1]之间 

### 误差计算 

- ```python
  o = tf.random.normal([2,10]) # 构造网络输出 
  y_onehot = tf.constant([1,3]) # 构造真实值 
  y_onehot = tf.one_hot(y_onehot, depth=10) 
  loss = keras.losses.MSE(y_onehot, o) # 计算均方差 
  # 特别要注意的是，MSE 函数返回的是每个样本的均方差，需要在样本维度上再次平均来获 得平均样本的均方差，实现如下
  loss = tf.reduce_mean(loss) # 计算 batch 均方差 
  ```

  



