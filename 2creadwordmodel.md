### 对数据集语料建模
#### 1、对语料进行分词
##### 
```python
stop_words = []
cop = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")
def getsegs(content_train):
        contents_segs = list()
        for content in content_train:
                rcontent = cop.sub('',content)
                segs = [word for word in jieba.cut(rcontent) if word not in stop_words]
                contents_segs.append(" ".join(segs))
        return contents_segs
```
##### 2、fastText训练词向量
###### 把分词数组转换成txt文件，作为input给fastText模型
```python
import fasttext
# cbow model :
print('start to train word2vec model',datetime.datetime.now())
indexnum = 0
for path in filepaths:
        print('start to train file:', path)
        traindf = pd.read_csv(path)
        content_train = traindf.iloc[:, 1]
        contents_segs=getsegs(content_train)

        np.savetxt(r'data/train/tmp'+str(indexnum)+'.txt',contents_segs,fmt='%s',encoding='utf-8')
        model=fasttext.train_unsupervised('data/train/tmp'+str(indexnum)+'.txt',model='cbow')
        print('end to train file:', path)
        indexnum=indexnum+1
model.save_model("model_ft1.bin")
print('end to train word2vec model',datetime.datetime.now())

```
