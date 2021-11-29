# Federated Bandit简介

## 一、Federated Learning

### 1. 简要说明

​	由于联邦Bandit问题是在多个MAB（Multi-armed Bandit）并行工作下引入联邦学习概念而产生的，故在介绍它之前先简单说明一下联邦学习的一些基本信息。如想了解更多可以考虑看一下我参考的几篇博客，链接附在最后。

​	联邦学习是一个诞生于实际生产中的方向，最早是2016年由Google提出的，主要想法是基于分布在多个设备上的数据集构建机器学习模型，同时防止数据泄露。我们知道，bandit在生产当中的应用一般是在推荐算法领域，这个环境同样也是联邦学习的应用场景之一，所以联邦bandit这个问题很自然地进入了我们的视野。本文基于2020年的论文《Federated Bandit: A Gossiping Approach》进行介绍，该论文涉及的联邦学习工作主要在于对于各Bandit数据隐私的处理。下面介绍文章所考虑的差分隐私概念。

### 2. 差分隐私

​	差分隐私顾名思义就是用来防范差分攻击的。而差分攻击又是什么呢？举个简单的例子，假设现在有一个婚恋数据库，2个单身8个已婚，只能查有多少人单身。刚开始的时候查询发现，2个人单身；现在张三跑去登记了自己婚姻状况，再一查，发现3个人单身。所以张三单身。这样的结果可以提供很多额外信息，比如攻击者可以通过这个结果确定该数据所录入的数据库。针对差分隐私的保护方法有很多，其中一种便是为返回数据加入拉普拉斯噪声，即把返回一个确定的值变为返回一个确定分布中的采样值。虽然对外界而言获取信息的确定性降低了，但对数据库而言安全性提高了。

​	同样地，在多MAB协同工作的场景下也涉及到各自数据隐私问题，MAB互相通信时也会考虑把自己发出的数据加噪而减少额外信息被他人获取。具体细节将在下一部分说明。

## 二、Federated Bandit

### 1. 问题预设

​	经典的bandit问题预设和基本算法大家都已有所了解，下面我将介绍联邦Bandit要解决的问题。

​	联邦bandit的基础是多MAB问题，而说到多个MAB，我们会想到Contextual Bandit。与上下文bandit的多个MAB串行工作的情况不同，联邦bandit场景下多个MAB并行工作且在一定条件下共享数据。每个MAB的臂数相同，同一个MAB臂与臂之间的奖励期望、不同MAB对应臂的奖励期望没有预设，但是每个MAB同一个臂平均奖励期望是有所预设的。那么，联邦bandit与传统bandit想要达成的目标类似，即通过这些MAB的共同努力找到并且大家都尽可能多拉动平均奖励期望最高的那个臂。

​	联邦bandit不像上下文bandit那样关心各臂奖励期望的构成，而是关注MAB之间的通信方式以及收到他人发来的信息后对自己的策略的更新方式。我所参考的这篇论文对MAB的分布场景基于一个Gossip的假设，即存在N个M臂的bandit，他们之间构成一个连通图（若边的权值均为1则结点间距离最大为N-1），每次迭代仅与一个邻居结点进行通信（gossip），通信内容为本轮所获第k臂奖励平均值。

​	文章介绍了两种算法，二者皆是在UCB算法的基础上作出了对当前新场景的适应性调整，且第二种Fed_UCB算法是第一种Gossip_UCB算法的进阶（加入了隐私保护），算法具体细节见下一部分。

### 2. 基本算法

#### （1）Gossip_UCB

​	文章给出了奖励X预测值的更新公式如下：

![公式（2）](https://github.com/Tongs2000/Bandits/blob/main/Federated%20Bandit/%E5%85%AC%E5%BC%8F%EF%BC%882%EF%BC%89.png?raw=true)

​	用于UCB算法步骤的函数C保留了经典UCB中的原始形式如下：

![公式（11）](https://github.com/Tongs2000/Bandits/blob/main/Federated%20Bandit/%E5%85%AC%E5%BC%8F%EF%BC%8811%EF%BC%89.png?raw=true)

​	输入MAB连通图、时间T以及函数C(t)（C函数的构造方法详见论文中的推导过程），我们可以基于下述Gossip_UCB算法进行迭代。其中，集合N中的元素为当前agent的邻接结点，X表示agent自己拉动臂获得的reward，v表示经过通信之后更新的reward估值，通信时传递的信息也是v。在通信的过程中，邻接结点间共享第k个臂迄今为止拉动次数，并以此估计该臂全局最多的拉动次数，若自身对该臂拉动的次数落后了（文章设定的界限是N次，N即结点数，这样设置可以使每个agent对于第k臂的知识具有局部一致性），那么就把这个臂计入集合A并多加拉动。

![Gossip_UCB](https://github.com/Tongs2000/Bandits/blob/main/Federated%20Bandit/Fed_UCB.png?raw=true)

#### （2）Fed_UCB

​	在Fed_UCB中，我们加入联邦学习的方法，先基于以下算法构造一个X的部分和的集合。每个部分和的求法见下述算法，简单来说就是把迄今为止得到的X观察值按顺序取……8个、4个、2个、1个分组并分别求和。

![算法2](https://github.com/Tongs2000/Bandits/blob/main/Federated%20Bandit/%E7%AE%97%E6%B3%952.png?raw=true)

​	得到部分和集合之后，我们基于以下算法对X进行重构，即添加拉普拉斯噪声。

![算法3](https://github.com/Tongs2000/Bandits/blob/main/Federated%20Bandit/%E7%AE%97%E6%B3%953.png?raw=true)

​	引入隐私保护后，C函数也必须得到修改。文章通过推导，得到新的C函数计算公式如下：

![公式（12）](https://github.com/Tongs2000/Bandits/blob/main/Federated%20Bandit/%E5%85%AC%E5%BC%8F%EF%BC%8812%EF%BC%89.png?raw=true)

​	完成前面的工作后，我们便可以修改Gossip_UCB得到Fed_UCB算法如下：

![Fed_UCB](https://github.com/Tongs2000/Bandits/blob/main/Federated%20Bandit/Fed_UCB.png?raw=true)

## 三、 参考

1. [Federated Bandit: A Gossiping Approach](https://arxiv.org/abs/2010.12763v1)

2. [差分隐私（一） Differential Privacy 简介](https://zhuanlan.zhihu.com/p/139114240)

3. [联邦学习笔记整理（一）](https://blog.csdn.net/weixin_43893151/article/details/105077890?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_title~default-0.no_search_link&spm=1001.2101.3001.4242.1)



