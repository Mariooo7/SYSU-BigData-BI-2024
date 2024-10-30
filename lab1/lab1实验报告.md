

# 实验报告_数据预处理

# 数据集：Spotify 2023年最著名歌曲

---

## 0. 引言

由于对Python操作比较熟悉，以及出于个人兴趣，对该实验我首先想到使用Python实现。

## 1. 实验结果与问题讨论

### 1.1 数据集用途

​	在进行预处理前，首先需要考虑的是数据预处理后的用途，包括后续分析需要哪些数据？数据的类型应该是什么？数据应该进行怎样的转化？以及如何处理缺失、异常值等

由实验手册，本数据集的用途如下：

- **音乐分析**

  > 需要与音乐特征相关的数据

- **平台对比**

  > 需要不同平台的数据

- **艺术家影响力**

  > 需要包含艺术家信息的数据

- **时间趋势**

  > 需要时间信息

- **跨平台存在**

  > 需要不同平台的数据

### 1.2 导入需要用到的包

本实验要用到的包有: pandas 和 scikit-learn

- 前者用于处理dataframe数据
- 后者我们引入MinMaxScaler用于对特定数据进行MinMax标准化

In[1]：


```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
```

### 1.3 读取数据集

使用pandas的`read_csv`函数读取文件

此处课上有提到通常使用的UTF-8解码会报错，我上网查了一下，发现使用'latin-1'就不会报错，解决了读取文件报错的问题 

In[2]:


```python
# 读取数据集，文件在当前目录下，文件地址即文件名
spotify = pd.read_csv('spotify-2023.csv', encoding = 'latin-1')
```

### 1.4 检查数据集基本信息

In[3]:

```python
# 这一步主要是为了在IDE里能快捷地直接查看csv文件
spotify
```

Out[3]略



使用`df.head()`检查数据集头部的5行，根据输出结果概览数据集的基本情况

In[4]:


```python
# 输出前5行数据
spotify.head()
```

Out[4]:

![image-20241010231703383](E:\2024秋\机器学习\大数据与商业智能-2024-王杉\Lab 1\lab1实验报告.assets\image-20241010231703383.png)



获取数据集中的列名，及变量名称

考虑每一列对应的数据的意义以及可能与哪些用途关联、变量之间是否存在冗余、重复、矛盾或可转化、结合的关系

In[5]:

```python
# 获取数据集中的列名
spotify.columns
```

Out[5]:


    Index(['track_name', 'artist(s)_name', 'artist_count', 'released_year',
           'released_month', 'released_day', 'in_spotify_playlists',
           'in_spotify_charts', 'streams', 'in_apple_playlists', 'in_apple_charts',
           'in_deezer_playlists', 'in_deezer_charts', 'in_shazam_charts', 'bpm',
           'key', 'mode', 'danceability_%', 'valence_%', 'energy_%',
           'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%'],
          dtype='object')



`df.info()`获取数据集的基本信息，包括非缺失值计数、行数列数、变量类型

In[6]:


```python
# 查看数据集各列基本信息
spotify.info()
```
Out[6]:

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 953 entries, 0 to 952
    Data columns (total 24 columns):
     #   Column                Non-Null Count  Dtype 
    ---  ------                --------------  ----- 
     0   track_name            953 non-null    object
     1   artist(s)_name        953 non-null    object
     2   artist_count          953 non-null    int64 
     3   released_year         953 non-null    int64 
     4   released_month        953 non-null    int64 
     5   released_day          953 non-null    int64 
     6   in_spotify_playlists  953 non-null    int64 
     7   in_spotify_charts     953 non-null    int64 
     8   streams               953 non-null    object
     9   in_apple_playlists    953 non-null    int64 
     10  in_apple_charts       953 non-null    int64 
     11  in_deezer_playlists   953 non-null    object
     12  in_deezer_charts      953 non-null    int64 
     13  in_shazam_charts      903 non-null    object
     14  bpm                   953 non-null    int64 
     15  key                   858 non-null    object
     16  mode                  953 non-null    object
     17  danceability_%        953 non-null    int64 
     18  valence_%             953 non-null    int64 
     19  energy_%              953 non-null    int64 
     20  acousticness_%        953 non-null    int64 
     21  instrumentalness_%    953 non-null    int64 
     22  liveness_%            953 non-null    int64 
     23  speechiness_%         953 non-null    int64 
    dtypes: int64(17), object(7)
    memory usage: 178.8+ KB

此时已经可以发现数据集中有列存在缺失值，以及有些列数据类型与期望不一致



`df.describe()` 用于获取描述性统计信息，默认返回数值变量的描述性统计信息，修改参数`include`还可以指定所需其他类型变量的描述性统计信息，包括独特值、频数等我们重点关注的属性

In[7]:

```python
# 查看数值变量的描述性统计信息
spotify.describe()
```

Out[7]:

![image-20241010233126914](.\lab1实验报告.assets\image-20241010233126914.png)

In[8]:


```python
# 查看数据类型为object的变量的描述性统计信息
spotify.describe(include = 'object')
```

Out[8]:

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>track_name</th>
      <th>artist(s)_name</th>
      <th>streams</th>
      <th>in_deezer_playlists</th>
      <th>in_shazam_charts</th>
      <th>key</th>
      <th>mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>953</td>
      <td>953</td>
      <td>953</td>
      <td>953</td>
      <td>903</td>
      <td>858</td>
      <td>953</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>943</td>
      <td>645</td>
      <td>949</td>
      <td>348</td>
      <td>198</td>
      <td>11</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Daylight</td>
      <td>Taylor Swift</td>
      <td>723894473</td>
      <td>0</td>
      <td>0</td>
      <td>C#</td>
      <td>Major</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>2</td>
      <td>34</td>
      <td>2</td>
      <td>24</td>
      <td>344</td>
      <td>120</td>
      <td>550</td>
    </tr>
  </tbody>
</table>
此处重点关注明显异常值以及音乐特征数据的独特值的个数，可能影响我们的预处理思路





检查有无重复行，如果有，需要删除

In[9]:


```python
# 检查有无重复行
spotify.duplicated().sum()
```

Out[9]:


    0

无重复行



`df.isnull().sum()`获取各列缺失值计数

In[10]:


```python
# 获取各列缺失值的汇总计数
missing_values = spotify.isnull().sum()
missing_values
```

Out[10]:


    track_name               0
    artist(s)_name           0
    artist_count             0
    released_year            0
    released_month           0
    released_day             0
    in_spotify_playlists     0
    in_spotify_charts        0
    streams                  0
    in_apple_playlists       0
    in_apple_charts          0
    in_deezer_playlists      0
    in_deezer_charts         0
    in_shazam_charts        50
    bpm                      0
    key                     95
    mode                     0
    danceability_%           0
    valence_%                0
    energy_%                 0
    acousticness_%           0
    instrumentalness_%       0
    liveness_%               0
    speechiness_%            0
    dtype: int64



### 1.5 探索结果与后续处理思路

**经过上述操作，分析输出结果有以下发现和思考**：

1. 'in_shazam_charts', 'key'列分别存在50个和95个**缺失值**，这两列都是重要数据，分别可以用于平台分析和音乐分析，需要考虑处理方法

   > - 前者的数据表示在榜单中的存在以及排名，由于观察到'in_apple_charts','in_deezer_charts'列中有最小值 0， 合理推测0用来表示未上榜，故该列缺失值考虑用0来代替；
   >
   > - 后者为音乐的调性，无法从其他数据中推测和估计，但与音乐属性特征相关的数据仅用于音乐分析，考虑对该列不缺失的行进行子数据集提取用于音乐分析，而后在用于其他分析的新数据集中删除其他存储了音乐属性特征数据的列。

2. 'artist_count'在后续分析中不会用到，是**冗余数据**

3. 在艺术家分析中考虑将共同创作一个作品的多个艺术家视为一个组合，而**不考虑对'artist_name'拆分**以及如何计算不同艺术家对同一作品的贡献比重

4. 每一行代表一个track即一首音乐，原数据集只有'track_name'作为每一行的标识，在后续分析中可能需要对其进行**编码**得到'track_id'

5. 'streams','in_deezer_playlists','in_shazam_charts'**应当是数值数据被读取为对象数据**，应转化数据类型

6. 有关发布时间的数据被分为三列'released_year','released_month','released_day'， 且被读取成int64数值型数据，但在后续时间趋势分析中，我们需要**日期数据**以进行时间序列分析，应转化数据类型

7. 通过观察描述性统计信息未发现明显异常值，例如：应该是整数的数据中存在浮点数，单位是百分比的列包含负数或有值大于100

8. 原数据集中**不存在重复行**

9. 'key'与'mode'都代表没有顺序、非数值的类别，很容易想到对其使用**独热编码**以便后续分析

10. 为方便后续用到'streams'数据的分析、比较、可视化等操作，考虑采用**Min-Max标准化**方法将其标准化



## 2. 数据处理

### 2.1 为数据集编码

'track_name'可能重复，但只要没有重复行，就认为每一行都是不同的track，因此不仅对'track_name'的唯一值赋予'track_id',而是考虑对原数据集中所有行按照索引赋予id，由于索引与行本来就唯一对应，因此不需要特别存储'track_id'与每条数据的对应关系。

In[11]:


```python
# 打印原数据集已有的索引，如果索引异常需要重设
spotify.index
```

Out[11]:


    RangeIndex(start=0, stop=953, step=1)

原索引无异常



In[12]:


```python
# 将索引存储到track_id列中作为每一行数据的独特编码
spotify['track_id'] = spotify.index
```

In[13]:

```python
# 查看改动后数据集前5行
spotify.head()
```

![image-20241010234142426](.\lab1实验报告.assets\image-20241010234142426.png)

可见最后一列已经成功加入id



### 2.2 获取用于音乐分析的数据集
- 将'streams'转化为数值型变量并进行Min-Max标准化
- 提取列'track_id', 'streams','bpm', 'key', 'mode', 'danceability_%', 'valence_%','energy_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%','speechiness_%'
- 删除'key'缺失数据的行
- 对'key', 'mode'列进行独热编码
- 导出处理完毕的用于音乐分析的数据集

In[14]:


```python
# spotify['streams'] = pd.to_numeric(spotify['streams'])
# 试转化数据类型为数值，但上述代码会发生报错，报错信息提示position 574处有无法转化为数值的字符串
# 删除该行，并重新转化
spotify.drop([574], inplace = True)
spotify['streams'] = pd.to_numeric(spotify['streams'])
```

In[15]:

```python
# 对新获得的'streams'数据进行Min-Max标准化
# 保存、训练和使用scaler标准化数据
scaler = MinMaxScaler()
spotify['streams'] = scaler.fit_transform(spotify[['streams']])

# 查看改动后数据集前5行
spotify.head()
```

Out[15]:

无异常，略



In[16]:


```python
# 提取音乐分析所需的列到新数据集spotify_features
spotify_features = spotify[['track_id', 'streams','bpm', 'key', 'mode', 'danceability_%', 'valence_%','energy_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%','speechiness_%']]
# 查看新数据集前5行
spotify_features.head()
```

Out[16]:

无异常，略



`df.dropna()`中指定`subset`可以按列删除有缺失值的行

In[17]:


```python
# 删除'key'缺失的行
spotify_features.dropna(subset = 'key')
```

![image-20241010234652292](.\lab1实验报告.assets\image-20241010234652292.png)

 

`pd.get_dummies()`指定列可以进行对数据的独热编码，默认用'_'连接原列名与每个独热编码对应的数据，并转化原单列数据为独特值个数列的数据

In[18]:


```python
# 将'key', 'mode'进行独热编码
spotify_features = pd.get_dummies(spotify_features, columns = ['key', 'mode'])
```


In[19]:
```python
# 查看改动后数据集的前5行
spotify_features.head()
```

![image-20241010234934964](.\lab1实验报告.assets\image-20241010234934964.png)

转化成功



`df.to_csv()`导出数据集为csv，` index= False`不导出索引

In[20]:


```python
# 存储处理完毕的用于音乐分析的数据集
spotify_features.to_csv('spotify_features.csv', index= False)
```



### 2.3 获取用于其他后续分析的数据集

- 删除只用于音乐分析的列和冗余的'artist_count'列
- 将'released_year','released_month','released_day'转化为日期数据
- 将'in_deezer_playlists','in_shazam_charts'数据类型转化为数值，并用0替代缺失值

In[21]:

```python
# 按列名删除不需要的列并存储到新数据集
spotify_else = spotify.drop(['artist_count', 'bpm', 'key', 'mode', 'danceability_%', 'valence_%','energy_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%','speechiness_%'], axis = 1)
# 查看提取数据集的前5行
spotify_else.head()
```

![image-20241010235200833](.\lab1实验报告.assets\image-20241010235200833.png)



`pd.to_datetime()`可以识别年月日数据并进行转化，但在该实验中遇到注释内描述的问题，后考虑使用连接字符串再转化解决问题

In[22]:


```python
# 获取日期数据，并存储为'date'列
# spotify_else['date'] = pd.to_datetime(spotify_else[['released_year','released_month', 'released_day']])
# 尝试上述函数后发现报错，to_datatime函数不能找到需要的年月日数据，故改用连接字符串格式的年月日数据再转化
spotify_else['date'] = pd.to_datetime(spotify_else['released_year'].astype(str) + '-' + spotify_else['released_month'].astype(str) + '-' + spotify_else['released_day'].astype(str))
# 删除原来的年月日列
spotify_else.drop(['released_year', 'released_month', 'released_day'], axis = 1, inplace = True)
# 查看改动后数据集前5行
spotify_else.head()
```

![image-20241010235223924](.\lab1实验报告.assets\image-20241010235223924.png)



In[23]:


```python
# 转化'in_deezer_playlists','in_shazam_charts'数据类型转化为数值
# 'in_deezer_playlists'无缺失值，直接转化
# spotify_else['in_deezer_playlists'] = pd.to_numeric(spotify_else['in_deezer_playlists'])
# 首次运行会报错，报错信息指出在position 48 的数据中有异常','
# 删除该列中的',', 再重复以上操作
spotify_else['in_deezer_playlists'] = spotify_else['in_deezer_playlists'].str.replace(',', '')
spotify_else['in_deezer_playlists'] = pd.to_numeric(spotify_else['in_deezer_playlists'])

# 'in_shazam_charts'有缺失值
# spotify_else['in_shazam_charts'] = pd.to_numeric(spotify_else['in_shazam_charts']).fillna(0).asype(int)
# 遇到上述相同类型报错
# 重复以上操作
spotify_else['in_shazam_charts'] = spotify_else['in_shazam_charts'].str.replace(',', '')
spotify_else['in_shazam_charts'] = pd.to_numeric(spotify_else['in_shazam_charts']).fillna(0).astype(int)
# 如果不附加.astype(int)会导致处理结果为浮点数
```

In[24]:

```python
# 查看处理完毕的数据集的前5行
spotify_else.head()
```

![image-20241010235442056](.\lab1实验报告.assets\image-20241010235442056.png)

In[25]:


```python
# 存储处理完毕的用于其他分析的数据集
spotify_else.to_csv('spotify_else.csv', index= False)
```



## 3. 实验心得

1. 该数据集比较干净，缺失值的处理一有章法可循，二可以通过按用途提取子数据集来避免数据损失
2. 该数据集的用途明确，方便了数据预处理，说明在数据预处理中后续分析用途即预处理目的至关重要
3. 大部分时间花在分析而不是操作上，思维比技术更重要
4. 数据处理中主观的成分有一定占比，尤其是在决定处理缺失值异常值以及考虑有关联的变量时



