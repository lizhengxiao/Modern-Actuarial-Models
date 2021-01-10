# 自然语言处理 {#nlp}

*陈美昆、蒋慧华、高光远*



<div class="figure" style="text-align: center">
<img src="./plots/7/history.png" alt="NLP历史（2000年以后）" width="70%" />
<p class="caption">(\#fig:history)NLP历史（2000年以后）</p>
</div>

在保险业，大量的书面证据，如保单合同或索赔通知，以及客户与企业对话助理互动的记录，为数据科学家和精算师提供了越来越多的可供分析的文本信息。

NLP在保险行业中的应用：

- 根据文字描述的索赔类型和严重程度对索赔进行分类

- 对电子邮件、保单、合同的分类

- 从文本数据中识别欺诈案例等

**传统方法的基本过程**

1. 文本预处理

2. **映射到实数集**: bag-of-words, bag-of-POS, pre-trained word embedding

3. 有监督机器学习算法: AdaBoost, random forest, XGBoost

**循环神经网络的基本过程**

1. 文本预处理

2. 循环神经网络: RNN, GRU, LSTM

**区别**

1. 传统方法非常依赖第二步特征工程的效果,神经网络不需要特征工程,它通过监督学习.进行自动特征工程.

2. 传统方法没有考虑文本的时间序列特征, 循环神经网络考虑了文本的时间序列特征.

## 预处理

1. 输入原始文本和格式

2. 将文本转化为小写

3. 分词（ tokenization ）

4. 删除停用词（ stopwords ）

5. 词性标注（POS tagging, 这步在 bag-of-POS需要, 如果只是 bag-of-words，此步不需要)

6. 词干提取或词形还原（Stemming or Lemmatization）

## Bag of words

Bag-of-words模型是信息检索领域常用的文档表示方法。在信息检索中，BOW模型假定对于一个文档，忽略它的单词顺序和语法、句法等要素，将其仅仅看作是若干个词汇的集合，文档中每个单词的出现都是独立的，不依赖于其它单词是否出现。也就是说，文档中任意一个位置出现的任何单词，都不受该文档语意影响而独立选择的。例如有如下两个文档：

1. Bob likes to play basketball, Jim likes too.

2. Bob also likes to play football games.

基于这两个文本文档，构造一个词典：

Dictionary = {1:"Bob", 2. "like", 3. "to", 4. "play", 5. "basketball", 6. "also", 7. "football", 8. "games", 9. "Jim", 10. "too"}

这个词典一共包含$10$个不同的单词，利用词典的索引号，上面两个文档每一个都可以用一个$10$维向量表示（用整数数字$0:n$（$n$为正整数）表示某个单词在文档中出现的次数）：

 1：$[1, 2, 1, 1, 1, 0, 0, 0, 1, 1]$

 2：$[1, 1, 1, 1 ,0, 1, 1, 1, 0, 0]$
     
向量中每个元素表示词典中相关元素在本文本样本中出现的次数。不过，在构造文档向量的过程中可以看到，我们并没有表达单词在原来句子中出现的次序（这是本Bag-of-words模型的缺点之一，不过瑕不掩瑜甚至在此处无关紧要）。    

我们使用`TfidfVectorizer`进行BOW,它不使用词出现的次数,而计算term frequency - inverse document frequency (tf-idf)。研究表明tf-idf更能反映文本的特征。

## Bag of part-of-speech

相较于BOW，bag of POS仅仅多了一步，即对每个词语进行词性标注，然后使用`TfidfVectorizer`。可知bag of POS可以达到降维的目的，但它散失了原文本的词语的信息，只考虑了词性。

## Word embeddings

词嵌入考虑把每个词语映射到用多维实数空间中，有很多预训练的映射可供选择，这些映射通常考虑了词语的先后顺序和同义词反义词等。常用的词嵌入模型包括：

- Neural probabilistic language model

- word2vec

- Global vectors for word representation

### Pre-trained word embeddings

我们使用`en_core_web_sm`进行词嵌入，它可以把任意的词映射到$\mathbb{R}^{96}$。另外，`en_core_web_mb`为更复杂的映射，可以把任意的词映射到$\mathbb{R}^{300}$。对于一个包含多个词语的文本，我们用所有词语的词嵌入平均值来表示这个文本

## 机器学习算法

通过bag-of-words, bag-of-POS，word embedding 我们把每个文本转变为一个向量。这样可以利用如下几种机器学习算法进行文本分类等监督学习。

- AdaBoost

- Random Forest

- XGBoost


## 神经网络

### 数据预处理

在神经网络中，我们不需要进行以上bag-of-words, bag-of-POS, word embedding等人工特征工程，我们只需要用实数把文本中**词语的顺序**表征出来即可，神经网络可以同步进行特征工程和有监督训练。

在python中可以使用Keras 中的`Tokenizer`模块把词语映射到非负整数上，此方法保持了保持了词语的顺序，是前面几种方法没有达到的。

可以看到，使用神经网络时，我们仅仅需要进行很少的特征工程，词语的意义将由神经网络在监督学习中学到。文本是时间序列数据，常用于时间序列分析的模型包括

- LSTM

- GRU

## Case study

<div class="figure" style="text-align: center">
<img src="./plots/7/files.png" alt="文档结构" width="70%"  />
<p class="caption">(\#fig:files)文档结构</p>
</div>

<div class="figure" style="text-align: center">
<img src="./plots/7/procedure.png" alt="nlp4class_exercise步骤" width="70%"  />
<p class="caption">(\#fig:procedure)nlp4class_exercise步骤</p>
</div>

- 任务描述：根据电影评论文本判断该评论是“好评”还是“差评”

- 数据来源： Internet Movie Database (IMDb)

- 数据量：案例中共有5000条含分类信息（pos/neg）的原始数据，从计算资源是运行时间考虑，我们从原始数据中随机抽取了1%，即500条数据进行测试，测试数据在toymdb文件夹下，文件结构和原始数据一致（需要注意的是， toymdb文件夹包含train和test两个文件，但并不是实际处理时的“训练集”和“测试集”，实际训练时读取所有数据并重新划分“训练集”和“测试集” ，所以实际上train和test中的文件没有区别，只是沿袭了原始数据的存储结构）。

- 数据结构：评论位置代表类别，一个txt存储一个评论数据。 

### 函数说明

### 可能遇到的问题

- 词性标注

```
  import nltk
  from nltk import pos_tag, word_tokenize
  出现LookupError
```

```
  解决方法：把'nltk_data.zip'里的文件全部拷贝至'/Users/huihuajiang/nltk_data/'         
  用以下命令可以查看你的 nltk_data 文件夹路径：             import nltk             
  print(nltk.data.path)
```

- 词嵌入

```
  import spacy
nlp = spacy.load(‘en_core_web_sm’) 
错误1：OSError: [E050] Can't find model 'en_core_web_sm'.
错误2：numpy.core.multiarray failed to import
```

```
  错误1解决方法1（可能不行）：
  命令行运行命令”python -m spacy download en_core_web_sm”
  错误1解决方法2：把模型下载到本地进行安装
  具体操作请参考 https://www.freesion.com/article/73801416523/
  错误2解决方法：重启一下终端
```

### 结果比较

*以下仅为表例*

$$
\begin{array}{|l|c|cc|c|}
\hline &  {\text { in-sample }} & {\text { out-of-sample }} & {\text { out-of-sample }}&  {\text { run times }}\\
& \text {both genders} & \text { female } & \text { male } & \text {both genders} \\
\hline \hline \text { LSTM3 }\left(T=10,\left(\tau_{0}, \tau_{1}, \tau_{2}, \tau_{3}\right)=(5,20,15,10)\right) & 4.7643 & 0.3402 & 1.1346 & 404 \mathrm{s}\\
\text { GRU3 }\left(T=10,\left(\tau_{0}, \tau_{1}, \tau_{2}, \tau_{3}\right)=(5,20,15,10)\right) & 4.6311 & 0.4646 & 1.2571 &  379 \mathrm{s} \\
\hline\hline \text { LSTM3 averaged over 100 different seeds} & - & 0.2451 & 1.2093 & 100 \cdot 404\mathrm{s}\\
\text { GRU3 averaged over 100 different seeds } & - & 0.2341& 1.2746 &  100 \cdot 379 \mathrm{s} \\
\hline \hline \text { LC model with SVD } & 6.2841 & 0.6045 &1.8152 &  - \\
\hline
\end{array}
$$


## 结论

我们用bag of words, bag of POS, word embeddings三种NLP模型对评论文本进行了向量化，并用ADA, RF, XGB三种机器学习方法对文档进行了分类，为此，我们引入了NLP管道来预处理文本数据 。最后，还采用RNN模型对文档进行了分类。

实验结果表明， 

- bag of words总体上变现更好， bag of POS的表现不佳。

- 与bag of words相比， word embedding模型的效率更高。

- Deep LSTM模型表现要比Single LSTM和更好。

- 与NLP模型相比，RNN模型性能更好，达到相同标准的时间更短。

- 如果用RNN的输入数据来拟合Ml模型，精度远不及RNN模型，可见RNN模型在该任务上能利用更少的信息实现更准确的分类。
