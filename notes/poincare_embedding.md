#### Poincare Embedding for Learning Hierarchical Representation

[paper](7213-poincare-embeddings-for-learning-hierarchical-representations.pdf)

[github](https://github.com/facebookresearch/poincare-embeddings)

####背景
常用的embedding方法是利用两个item的欧式距离来表示两者的相似性。且无法呈现出层次性。

####文章特点
这篇论文主要利用了<mark>hyperbolic space</mark>来进行embedding操作。使用Poincare ball model进行表达（Riemanniam manifold structure）。

相关的距离公式参考原文。

##### 模型学习
1. 通过惩罚无链接的节点和距离很远的相邻节点（negative sampling）进行梯度下降计算。
2. 初始化为uniform(-0.001, 0.001)，从而保证初始化时各item接近原点。
3. burn-in phase
4. reduced learning rate

比较神奇的是，目前我看到的训练集全是negative samples。