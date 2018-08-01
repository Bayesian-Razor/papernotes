#  Neural Belief Tracker: Data-Driven Dialogue State Tracking

[paper](https://arxiv.org/pdf/1606.03777v2.pdf)

--
整体来说，作用于fram-based dialogue system，尝试用机器学习的方式替代基于keywords或人工工程的slot filling

工作集中在：

1. 用户输入的向量化表达，utterance的distributed representation
2. 意图识别、槽匹配，intent matching 
	1. 主要将 utterance representation， slot representation， system's request in previous step, 联合计算，进行binary decision

---

designed to detect the slot-value pairs that make up the user's goal at a given turn during the lfow of dialogue.


+ representation learning 
	+ user utterance embedding, **r**  
	+ current candidate slot-value pair, **c**
	+ system dialogue acts, (**tq**, **ts**, **tv**)
+ context modelling 
+ semantic decoding 
	+ intermediate interaction summary vectors 
		+ **dr**, **dc**, **d**
+ decision-making
	+ decide user expressed intent (candidate slot-value pair)

	
### maybe benefit
 Mrksiˇ c et al. ( ´ 2016), specialising word vectors to express semantic similarity (**《Counter-fitting Word Vectors to Linguistic Constraints. In Proceedings of HLT-NAACL.》**)
 
 unseen words semantically related to familiar slot values (i.e. inexpensive to cheap) will be recognised purely by their position in the original vector space (see also
Rocktaschel et al. ( ¨ 2016)).  (**《Reasoning about entailment with neural attention. In ICLR》**)