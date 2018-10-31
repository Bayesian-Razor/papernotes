#### Directional Skip-Gram: Explicitly Distinguishing Left and Right Context for Word Embeddings

[paper](http://aclweb.org/anthology/N18-2028)

### 背景

- `Skip-Gram Model`用w_i预测w_{i-t}到w_{i+t}的词，需要2|V|d个参数。但是它不区分具体的位置和上下文。
- 在此基础上，`Structured Skip-Gram Model`每个词会学习2t个词向量作为相对center word第i个位置的词向量。这样极大的增加了参数的个数。

### Directional Skip-Gram Mode

- 简化了`SSG`的模型，仅判断w_{i+t}是位于w_i的`left`还是`right`。
- 引入`delta`参数，作为用来判断是`left`还是`right`的参数。
- 目标函数分为两部分，一部分和原始的`Skip-Gram Model`一致，另一部分用`delta`作为参数，用来输出是左右关系。

### 其他

- 有对应的训练好的词向量，但是没有代码。
- delta有没有什么更好解释的意义？后面只有实验，缺少一些解释。
- 从实验结果上看，小语料上也有一些提升。