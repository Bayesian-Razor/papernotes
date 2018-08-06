# Universal Language Model Fine-tuning for Text Classification

[paper](https://arxiv.org/abs/1801.06146)

-- 

总体来说，作者将CV上一些fine-tuning的方法应用到了NLP的`Text classification`任务上。就我自己感觉这些东西说是方法，说是一些`trick`更合适。

模型叫`ULMFiT`，整体分为三个步骤

- 使用一个大的性质良好的数据集来进行model pretraining。文中使用的是`Wikitext-103`。
- 在固定的任务上做fine-tuning。这里使用两个技巧。
    - 第一个是`Discriminative fine-tuning`。由于不同层会获取不同level的特征，越靠近输出的层越关注一些宏观的性质，越靠近输入的层越能学习到通用的特征，因此这里使用$\eta^{l-1} = \eta^l / 2.6$
    - 第二个是`Slanted triangular learning rates`。先使用较小的学习率，希望可以较小的学习到一个优化的方向，然后提高学习率，更快的优化，最后再慢慢降低学习率，慢慢的优化。
- 针对特征任务的分类器fine-tuning。这里也是用了四个技巧。
    - `concat pooling`。把rnn的每次的state，使用`maxpool`以及`meanpool`最后和输出的`h`拼到一起。（这里的这个技巧，其他地方用过，在分类任务上确实好用）
    - `gradual unfreezing`。从接近输出逐层解冻`layer`，然后做`fine tuning`。
    - `BPT3C`。将长文本切成等长的文本，然后每次后面的开始的时候，使用前面的`final state`，然后`pooling`是保持计算的。
    - `Bidirection LM`。
