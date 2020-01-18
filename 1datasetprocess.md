### 对数据集进行一些处理。
#### 1、数据分布初步分析
##### 可以看出数据集的分布式不均衡的。
```python
traindf=pd.read_csv('data/train/train.csv')
columns=traindf.columns.values.tolist()
for column in columns[2:]:
    print (column)
    print(traindf[column].value_counts())
out:
location_traffic_convenience
-2    81382
 1    21254
-1     1318
 0     1046
Name: location_traffic_convenience, dtype: int64
location_distance_from_business_district
-2    83680
 1    20201
-1      586
 0      533
Name: location_distance_from_business_district, dtype: int64
location_easy_to_find
-2    80605
 1    17947
-1     3976
 0     2472
 
```
##### 2、提取有效数据
###### 只提取中文、字母、数字
```python
import re
cop = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")
def seg_words(contents):
    contents_segs = list()
    for content in contents:
        rcontent = cop.sub('', content)
        segs = [word for word in jieba.cut(rcontent) if word not in stop_words]
        contents_segs.append(" ".join(segs))
    return contents_segs

```
