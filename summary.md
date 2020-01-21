### 情感分析建模
#### 一、效果地址

#### 二、总结
##### 机器学习算法有三个部分组成：
##### 第一部分：input 数据
##### 数据来源是：AI Challenger 2018 细粒度情感分析赛道

##### 第二部分：cost function：用熵表示。
##### ce=tf.multiply(Y,-tf.log(y_model))
##### cost = -tf.reduce_sum(tf.multiply(ib_weight,ce)) #针对数据的不平衡，增加一个出错的惩罚权重。

##### 第三部分：优化方法：梯度下降方法。
##### train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

##### 另外增加采用了SGD方法，小范围内更新相关参数，和交叉验证方法。

##### 说明：代码使用的是tensorflow 1.0, 原因似乎是1.0可以更多的了解一些编码细节。2.0封装的更高一些。如果安装的是2.0，也可以通过如下命令使用1.
##### 在项目中对这些概念的理解更加深刻。
