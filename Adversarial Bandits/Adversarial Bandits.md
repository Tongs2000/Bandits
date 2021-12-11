# Adversarial Bandits

## Adversarial Bandits简介

Stochastic Bandits的另一个变体被称为Adversarial Bandits，由Auer和Cesa-Bianchi（1998）首次提出。在这个变体中，在每个迭代中，代理人选择一个手臂，对手同时选择每个手臂的报酬结构。这是对强盗问题最有力的概括之一，因为它消除了所有的分布假设，对Adversarial Bandits的解决是对更具体的强盗问题的概括性解决。

在Adversarial bandits中, 我们不再假设reward是从一个固定的分布中采样获得, 相反, 它由一个称为adversary的环境来决定。换言之，任何算法都有可能连续取到最差的值，举例有：

> 赌场老板是adversary，赌客是agent。老板今天心情不好，赌客选什么，赌场老板就把什么奖励设置为最差的，赌客对此一无所知且无法干涉，这种情况下什么算法都白搭。

讨论Adversarial bandits是试图找出一个对所有bandits问题都适用的算法，因为adversary所返回的reward是随机的，因此算法也必须是随机的，连续取到最差的reward是无法避免的，只能从理论上去尽可能避免连续取到最差的reward。

## Adeversarial Bandits流程

已知的参数: arm 的数目K, 需要决策的轮次总数n

For t in range(n):

1. The adversary selects a gain vector for K arms ![[公式]](./1.svg)
2. The agent chooses ![[公式]](./2.svg)
3. The agent receives the reward ![[公式]](./3.svg)

同样, 我们定义regret

![[]](./4.svg)

将其与Stochastic Bandits的regret对比，易知

![[]](./6.png)

​												左：Adversarial bandit 右：Stochastic bandit 不等式

所以：Adversarial bandit的worst-case后悔以Stochastic bandit的worst-case后悔以为下界；将Adversarial bandit后悔界优化的算法也可以优化Stochastic bandit。

## Exp3算法流程

为Adversarial bandit设置的最标准算法是EXP3算法（**E**xponential-weight algorithm for**E**xploration and**E**xploitation）。

算法的主要思路是从均匀分布开始, agent 维护一个![[公式]](./7.svg) 上的分布, agent 对arm 的评估蕴含在这个分布选择各个arm的概率中, 而这个概率随着每轮接受到的reward而更新。（总的来看，比Stochastic Bandits更有强化学习输出动作概率采样的感觉，而且这个算法也用到了importance sampling思想)

![[]](./8.jpg)

![[]](./9.jpg)

![[]](./10.jpg)

Exp3算法有着各种变体和优化过的样子，上方3张分别是原论文中两种算法流程以及英文wiki上Mult-arm Bandits关于Exp3算法的流程。
