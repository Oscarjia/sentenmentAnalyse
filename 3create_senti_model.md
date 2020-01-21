### 情感分析建模
#### 1、初始预备。
##### 使用传统的机器学习 逻辑回归来建立多分类模型，测试一下结果。准备好INPUT X 数据和 Y 数据。
```python
import pandas as pd
import jieba
import re
from util import seg_words, load_data_from_csv, getvectors, getonehotlabels
import config
import numpy as np
import tensorflow.compat.v1 as tf
from sklearn.metrics import  f1_score
from sklearn.externals import joblib
tf.disable_v2_behavior()
traindf = load_data_from_csv(config.train_data_path)
validdf = load_data_from_csv(config.validate_data_path)

content_train = traindf.iloc[:, 1]
valid_train = validdf.iloc[:, 1]
train_segs = seg_words(content_train)
valid_segs = seg_words(valid_train)

xs = getvectors(train_segs)
arr_train=np.arange(xs.shape[0])
np.random.shuffle(arr_train)  #数据集分布不均衡，所以这里也对数据进行打乱。
xs=xs[arr_train,:]
train_size, num_features = xs.shape
xs_valid=getvectors(valid_segs)
arr_valid=np.arange(xs_valid.shape[0])
np.random.shuffle(arr_valid)
xs_valid=xs_valid[arr_valid,:]
valid_size, _ = xs_valid.shape
ys_traindicts = getonehotlabels(traindf)
#ys_traindicts=ys_traindicts[arr_train,:]
ys_validdicts=getonehotlabels(validdf)
#ys_validdicts=ys_validdicts[arr_valid,:]
```
##### 2、训练模型
###### 实验不同的参数，判断模型的效果。发现当batch_size比较大为1000的时候，f1-score的效果比较差，取值100的时候比较好。这也说明数据集数据分布很不均衡。
###### 针对这种不均衡采取了一种 https://arxiv.org/pdf/1708.02002.pdf 的方法，增加了判定错误之后的惩罚ib_weight。感觉上这样做没有什么作用。[ib_wight=tf.multiply(Y, tf.pow(tf.subtract(1., y_model), gamma))]
###### 另外把这些重要的参数用joblib.dump保留到文件里面，用于做预测。joblib.dump(paramdict, config.model_path+'/paramdict_3')

```python
learning_rate = 0.01
training_epoch =1
num_labels = 4
batch_size = 100
gamma=4.
alpha=1.
X = tf.placeholder("float", shape=(None, num_features))
Y = tf.placeholder("float", shape=(None, num_labels))
w = tf.Variable(tf.zeros([num_features, num_labels]))
b = tf.Variable(tf.zeros([num_labels]))
y_model = tf.nn.softmax(tf.matmul(X, w) + b)
#ce=tf.multiply(Y,tf.log(y_model))
#for imbalanced data reason add ib_wight variable
model_true = tf.argmax(Y, 1)
model_predict = tf.argmax(y_model, 1)
correct_prediction = tf.equal(model_predict, model_true)
accuracy = tf.reduce_sum(tf.cast(correct_prediction, "float"))# for punishment
cost = -tf.reduce_sum(Y*tf.log(y_model))+gamma*accuracy
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    f1list = []
    #print(correct_prediction.eval())
    paramdict = dict()
    for column in config.columns[2:22]:
        print(column)
        # train set
        for step in range(training_epoch * train_size // batch_size):
            offset = (step * batch_size) % train_size
            batch_xs = xs[offset:(offset + batch_size), :]
            batch_lables = ys_traindicts[column][arr_train,:][offset:(offset + batch_size)]
            from sklearn.model_selection import KFold
            n_splits = 10
            kfold = KFold(n_splits=10, shuffle=True)
            kfold.random_state = step
            if len(batch_xs) <= n_splits:
                continue
            for train_idx, val_idx in kfold.split(batch_xs):
                train_x1 = batch_xs[train_idx]
                train_y1 = batch_lables[train_idx]
                test_x1 = batch_xs[val_idx]
                test_y1 = batch_lables[val_idx]
                # ,(vali_x,vali_y)
                err, _ = sess.run([cost, train_op], feed_dict={X: train_x1, Y: train_y1})
                err, _, acc, y_true, y_pred = sess.run([cost, train_op, accuracy, model_true, model_predict],
                                                       feed_dict={X: test_x1, Y: test_y1})
                model_f1_score = f1_score(y_true, y_pred, average='macro')
                model_f1_score_micro = f1_score(y_true, y_pred, average='micro')
                print('train step is {},err is {},acc is {},f1score macro is {},micro is {}'.format(step, err, acc, model_f1_score,model_f1_score_micro))
        # valid set
        for step in range(training_epoch * valid_size // batch_size):
            offset = (step * batch_size) % train_size
            batch_xs = xs_valid[offset:(offset + batch_size), :]
            batch_lables = ys_validdicts[column][arr_valid,:][offset:(offset + batch_size)]
            err, _, acc, y_true, y_pred = sess.run([cost, train_op, accuracy, model_true, model_predict],
                                                   feed_dict={X: batch_xs, Y: batch_lables})
            model_f1_score = f1_score(y_true, y_pred, average='macro')
            model_f1_score_micro = f1_score(y_true, y_pred, average='micro')
            f1list.append(model_f1_score)
            print('valid step is {},err is {},acc is {},f1score macro is {},micro is {}'.format(step, err, acc, model_f1_score,model_f1_score_micro))
            #f1list.append(model_f1_score)

        w_val = sess.run(w)
        #print('w ', w_val)
        b_val = sess.run(b)
        #print('b ', b_val)
        paramdict[column] = (w_val, b_val)
        # print('f1score ',f1score)
        #yv_true, yv_pred = sess.run([model_true, model_predict], feed_dict={X: xs_valid, Y: ys_validdicts[column]})
        #model_f1_scorev = f1_score(yv_true, yv_pred, average='macro'),
        #model_f1_scorev_micro = f1_score(yv_true, yv_pred, average='micro')
        #print('valid step is {},err is {},acc is {},f1score mcro is {},micro is {},'.format(step, err, acc,
        #                                                                                    model_f1_scorev,
        #                                                                                    model_f1_scorev_micro))
    print('final ly train f1 final macro score mean is {} '.format(np.mean(f1list)))
    joblib.dump(paramdict, config.model_path+'/paramdict_3')
```
