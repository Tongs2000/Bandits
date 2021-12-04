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

如果我们考虑对称的问题, 将reward 改为 loss，那么此时,regret 定义为

![[]](./5.svg)