# SeqGAN

## 背景

其实NLP问题上应用GAN应该也算是老生常谈了。第一个就是离散值的问题，文字本身是离散的，而神经网络输出的结果就算是过了softmax也还是连续的，为了要得到词，就要使用采样等方法，但是这样就没办法求导反向传播了。为了解决这个问题，目前我见到的只要有两个思路，一个是不使用`softmax`转而使用`gumbel softmax`，另一个方法就是采用`Policy gradient`的方法，把目标函数写到一起去。`SeqGAN`就是后一种方法。

## 大体内容

seqGAN在常见的GAN的体系上，利用`CNN`的`discriminator`产生的结果作为reward，来对`RNN`的`generator`做`Policy gradient`。为了更好的评估位于中间的词生成的`Q-value`，采用`Monte Carlo search`来对采样出来的完整路径来估计值。

在开始训练模型的时候，会先根据`MLE`来训练`G`，然后根据`cross entropy`来训练的`D`，来提高训练的效率。`discriminator`会在每次完整的句子生成之后才去计算每个词的`reward`。不过，这里为了调参，使用参数控制了训练生成器、判别器的步骤以及判别器在生成数据上的训练的`epoch`。而且这些参数很敏感。

## 其他想法

整个idea很好。不过`policy gradient`本身`variance`就很高。只看训练曲线和实验效果，并没有好的那么多。而且对于超参很敏感。不过现在的GAN模型基本都有这个问题。而且这篇论文应该是16年底放上去的，感觉可以理解。


## 代码

[code](https://github.com/LantaoYu/SeqGAN)