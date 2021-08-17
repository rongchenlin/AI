# Tensorflow常用函数

- [Tensorflow常用函数](#tensorflow常用函数)
  - [1. 基础](#1-基础)
  - [2. 数据处理](#2-数据处理)
  - [3. 网络搭建](#3-网络搭建)
  - [4.使用Keras](#4使用keras)
  - [5.自制数据（预处理），打包](#5自制数据（预处理），打包)

### Tensorflow常用函数
#### 1. 基础

```python
tf.int
tf.float32
tf.float64
```

```
tf.constant(张量内容，dtype=数据类型) # 创建张量
tf.convert_to_tensor(数据名，dtype=?) # numpy转tensor
tf.zeros(维度)
tf.ones(维度)
tf.fill(维度，指定值)  # 指定值的张量
```

 ```
tf.random.normal(维度，mean=均值，stddev=标准差)  # 正态分布的随机数
tf.random.truncated_normal(维度，mean=均值，stddev=标准差)  # 断式正态分布的随机数
 ```

```
tf.cast(张量名，dtype)  # 强制转换
tf.reduce_min(张量名)
tf.reduce_max(张量名)
te.reduce_mean(张量名,aixs=?)
te.reduce_sum(张量名,aixs=?)
```

```
axis :  
	在一个二维张量或数组中，可以通过调整 axis 等于0或1 控制执行维度。  axis=0代表跨行（经度，down)，而axis=1代表跨列（纬度，across) 如果不指定axis，则所有元素参与计算。
```

1. tf.Variable()  : 将变量标记为“可训练”，被标记的变量会在反向传播中记录梯度信息。神经网络训练中，常用该函数**标记待训练参数**。

   w = tf.Variable(tf.random.normal([2, 2], mean=0, stddev=1))

2. 四则运算：tf.add(张量1，张量2），tf.subtract，tf.multiply，tf.divide

3. 平方、次方与开方： tf.square，tf.pow，tf.sqrt

4. **矩阵乘：tf.matmul**

#### 2. 数据处理

1. data = **tf.data.Dataset.from_tensor_slices**((输入特征, 标签))

2. 梯度求解：

   ```
   # 这个with结构用于记录计算过程
   with tf.GradientTape( ) as tape:  
   	w = tf.Variable(tf.constant(3.0)) 
   	loss = tf.pow(w,2)
   grad = tape.gradient(loss,w)   # 计算梯度
   ```

3. **enumerate**：用于遍历元素

   ```
   for i, element in enumerate(seq): 
   	print(i, element)
   ```
   
4. ```
   np.random.seed(116) # 使用相同的seed，使输入特征/标签一一对应 
   np.random.shuffle(x_data) 
   np.random.seed(116) 
   np.random.shuffle(y_data) 
   tf.random.set_seed(116)
   ```

#### 3. 网络搭建

1. **tf.one_hot(待转换数据, depth=几分类)**：独热编码（one-hot encoding）：在分类问题中，**常用独热码做标签， 标记类别：1表示是，0表示非**。（比如，十分类的问题最终输出，但是只会把概率最大的输出为1，其他输出为0.

2.  tf.nn.softmax(x) 使输出符合概率分布

3. ```
   w.assign_sub (x)  
   tf.argmax (张量名,axis=操作轴) # 返回最大值的位置
   ```

4. ```
   tf.where(条件语句，真返回A，假返回B)  
   c=tf.where(tf.greater(a,b), a, b) # 若a>b，返回a对应位置的元素，否则 返回b对应位置的元素
   ```

5. **np.vstack(数组1，数组2)**：将两个数组按垂直方向叠加

6. ```
   np.mgrid[ 起始值 : 结束值 : 步长，起始值 : 结束值 : 步长 , … ]  # 类似于等差数组
   x, y = np.mgrid [1:3:1, 2:4:0.5]
   x.ravel( ) 将x变为一维数组
   np.c_[ ] 使返回的间隔数值点配对
   ```

7. 损失函数

   ```
   均方误差：loss=tf.reduce_mean(tf.square(y_-y))
   交叉熵：tf.losses.categorical_crossentropy(y_,y)
   ```

8. 正则化解决**过拟合**：正则化在损失函数中引入模型复杂度指标，**利用给W加权值**，弱化了训练 数据的噪声**（一般不正则化b）**

9. **优化器**：SGD、SGDM、Adagrad、RMSProp、Adam

#### 4.使用Keras

1. ```
   用Tensorflow API:tf.keras搭建网络八股
   六步法
   import
   train，test
   model=tf.keras.models.Sequential
   model.compile
   model.fit
   model.summary
   ```

2. model = tf.keras.models.Sequential ([ 网络结构 ]) #描述各层网络

3. 拉直层： tf.keras.layers.Flatten( )，将数据拉伸为一维数组

4. **全连接层**： tf.keras.layers.Dense(神经元个数, activation= "激活函数“ , kernel_regularizer=哪种正则化)

5. ```
   model.compile(optimizer = 优化器, loss = 损失函数 metrics = [“准确率”] )
   loss可选:'mse'、'sparse_categorical_crossentropy'
   Metrics可选:‘accuracy’、‘categorical_accuracy’ 、‘sparse_categorical_accuracy’（常用）
   ```

6. ```
   model.fit (训练集的输入特征, 训练集的标签, 			batch_size= ,	
   	epochs= ,
   	validation_data=(测试集的输入特征，测试集的标签), 	   validation_split=从训练集划分多少比例给测试集，       validation_freq = 多少次epoch测试一次)	
   ```

7. model.summary（）  : 返回参数

#### 5. 自制数据（预处理），打包

1. 数据增强

   ```python
   image_gen_train=ImageDataGenerator（
   	rescale=1./1.，#如为图像，分母为255时，可归至0~1
   	rotation_range=45，#随机45度旋转
   	width_shift_range=.15，#宽度偏移
   	height_shift_range=.15，#高度偏移
   	horizontal_flip=False，#水平翻转
   	zoom_range=0.5#将图像随机缩放闽量50%）
   image_gen_train.fit（x_train）
   ```

2. 断点续训

   ```python
   # 保存模型：使用回调函数callbacks
   tf.keras.callbacks.ModelCheckpoint( 
       filepath=路径文件名, 
       save_weights_only=True,  # 是否只保留模型参数
       save_best_only=True,   # 是否只保留最优解
   history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1, callbacks=[cp_callback])
   # 读取模型
   model.load_weights(路径文件名)
   ```

3. 参数提取：

   1. **model.trainable_variables** 模型中可训练的参数
   2. np.set_printoptions（threshold=np.inf)

4. acc/loss可视化

   ```
   history=model.fit(训练集数据, 训练集标签, batch_size=, epochs=, validation_split=用作测试数据的比例,validation_data=测试集, validation_freq=测试频率)
   # 参数设置
   history： 
   loss：训练集loss 
   val_loss：测试集 loss 
   sparse_categorical_accuracy：训练集准确率 
   val_sparse_categorical_accuracy：测试集准确率
   # 一个例子
   acc=history.history['sparse_categorical_accuracy']  
   valacc=history.history［'val_sparse_categorical accuracy']
   loss=history.history['loss']
   val loss=history. history['val_loss']
   ```

5. predict(输入特征, batch_size=整数)：返回前向传播计算结果

