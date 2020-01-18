#### 对数据集进行一些处理。
##### 1、提前有效数据
#只提取中文和字母数字
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
