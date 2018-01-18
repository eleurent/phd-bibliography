# Bibliography

# Table of contents

* [Reinforcement Learning](#reinforcement-learning)
  * [Value-based](#value-based)
  * [Policy-based](#policy-based)
    * [Policy gradient](#policy-gradient)
    * [Actor-critic](#actor-critic)
    * [Derivative-free](#derivative-free)
  * [Temporal abstraction](#temporal-abstraction)
  * [Partial observability](#partial-observability)
  * [Safety](#safety)
  * [Multi-agent](#multi-agent)
  * [Performances and acceleration](#performances-and-acceleration)
* [Learning from Demonstrations](#learning-from-demonstrations)
  * [Imitation Learning](#imitation-learning)
    * [IL applications](#il-applications)
  * [Inverse Reinforcement Learning](#inverse-reinforcement-learning)
    * [IRL applications](#irl-applications)
* [Tree Search](#tree-search)
* [Optimal Control](#optimal-control)
  * [Control Theory](#control-theory)
  * [Dynamic Programming](#dynamic-programming)
* [Motion Planning](#motion-planning)
  * [Search](#search)
  * [Sampling](#sampling)
  * [MP applications](#mp-applications)

![RL Diagram](https://rawgit.com/eleurent/phd-bibliography/master/reinforcement-learning.svg)

# Reinforcement Learning

## Value-based

* [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), Mnih V. et al, 2013. *DQN*. ([video](https://www.youtube.com/watch?v=iqXKQf2BOSE))
* [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461), van Hasselt H. Silver D. et al, 2015. *DDQN*.
* [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581), Wang Z. et al, 2015. *DDDQN*. ([video](https://www.youtube.com/watch?v=qJd3yaEN9Sw))
* [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952), Schaul T. et al, 2015. *PDDDQN*.
* [Continuous Deep Q-Learning with Model-based Acceleration](https://arxiv.org/abs/1603.00748), Gu S. et al, 2016. *NAF*.

## Policy-based

### Policy gradient

* [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf), Williams R., 1992. *REINFORCE*.
* [End-to-End Training of Deep Visuomotor Policies](https://arxiv.org/abs/1504.00702), Levine S. et al, 2015. *GPS*. ([video](https://www.youtube.com/watch?v=Q4bMcUk6pcw))
* [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477), Schulman J. et al, 2015. *TRPO*. ([video](https://www.youtube.com/watch?v=KJ15iGGJFvQ))
* [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), Schulman J. et al, 2017. *PPO*. ([video](https://www.youtube.com/watch?v=bqdjsmSoSgI))
* [Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/abs/1707.02286), Heess N. et al, 2017. *DPPO*. ([video](https://www.youtube.com/watch?v=hx_bgoTF7bs))

### Actor-critic

* [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf), Sutton R. et al, 1999. *AC*.
* [Natural Actor-Critic](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/NaturalActorCritic.pdf), Peters J. et al, 2005. *NAC*.
* [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783), Mnih V. et al 2016. *A3C*. ([video 1](https://www.youtube.com/watch?v=Ajjc08-iPx8) | [2](https://www.youtube.com/watch?v=0xo1Ldx3L5Q) | [3](https://www.youtube.com/watch?v=nMR5mjCFZCw))
* [Continuous Control With Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971), Lillicrap T. et al, 2016. *DDPG*. ([video 1](https://www.youtube.com/watch?v=lV5JhxsrSH8) | [2](https://www.youtube.com/watch?v=8CNck-hdys8) | [3](https://www.youtube.com/watch?v=xw73qehvSRQ) | [4](https://www.youtube.com/watch?v=vWxBmHRnQMI))

### Derivative-free

* [Learning Tetris Using the Noisy Cross-Entropy Method](http://iew3.technion.ac.il/CE/files/papers/Learning%20Tetris%20Using%20the%20Noisy%20Cross-Entropy%20Method.pdf), Szita I. Lörincz A., 2006. *CEM*. ([video](https://www.youtube.com/watch?v=UZnDYGk1j2c))
* [Completely Derandomized Self-Adaptation in Evolution Strategies](https://dl.acm.org/citation.cfm?id=1108843), Hansen N. Ostermeier A., 2001. *CMAES*.
* [Evolving Neural Networks through Augmenting Topologies](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf), Stanley K., 2002. *NEAT*. ([video](https://www.youtube.com/watch?v=5lJuEW-5vr8))

## Temporal abstraction

* [Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning](http://www-anw.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf), Sutton R. et al, 1999.
* [Intrinsically motivated learning of hierarchical collections of skills](http://www-anw.cs.umass.edu/pubs/2004/barto_sc_ICDL04.pdf), Barto A. et al, 2004.
* [Learning and Transfer of Modulated Locomotor Controllers](https://arxiv.org/abs/1610.05182), Heess N. et al, 2016. ([video](https://www.youtube.com/watch?v=sboPYvhpraQ&feature=youtu.be))
* [FeUdal Networks for Hierarchical Reinforcement Learning](https://arxiv.org/abs/1703.01161), Vezhnevets A. et al, 2017. *FuNs*.
* [On a Formal Model of Safe and Scalable Self-driving Cars](https://arxiv.org/abs/1708.06374), Shalev-Shwartz S. et al, 2017.

## Partial observability

* [Point-based Value Iteration: An anytime algorithm for POMDPs](https://www.ri.cmu.edu/pub_files/pub4/pineau_joelle_2003_3/pineau_joelle_2003_3.pdf), Pineau J. et al, 2003.
* [Point-Based Value Iteration for Continuous POMDPs](http://www.jmlr.org/papers/volume7/porta06a/porta06a.pdf), Porta J. et al, 2006.

## Safety

* [A Comprehensive Survey on Safe Reinforcement Learning](http://jmlr.org/papers/v16/garcia15a.html), García J., Fernández F., 2015.
* [Safe, Multi-Agent, Reinforcement Learning for Autonomous Driving](https://arxiv.org/abs/1610.03295), Shalev-Shwartz S. et al, 2016.

## Multi-agent

* [MAgent: A Many-Agent Reinforcement Learning Platform for Artificial Collective Intelligence](https://arxiv.org/abs/1712.00600), Zheng L. et al, 2017. ([video](https://www.youtube.com/watch?v=HCSm0kVolqI))

## Performances and acceleration

* [Near-optimal Regret Bounds for Reinforcement Learning](http://www.jmlr.org/papers/volume11/jaksch10a/jaksch10a.pdf), Jaksch T., ‎2010. *UCRL2*.
* [Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.84.6983&rep=rep1&type=pdf), Sutton R., 1990. *Dyna*.

# Tree Search

* [Mastering the game of Go with deep neural networks and tree search](https://www.nature.com/articles/nature16961), Silver D. et al, 2016.
* [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815), Silver D. et al, 2017.

# Learning from Demonstrations

## Imitation Learning

* [Learning from Demonstrations for Real World Reinforcement Learning](https://pdfs.semanticscholar.org/a7fb/199f85943b3fb6b5f7e9f1680b2e2a445cce.pdf), Hester T. et al, 2017. *DQfD*. ([videos](https://www.youtube.com/watch?v=JR6wmLaYuu4&list=PLdjpGm3xcO-0aqVf--sBZHxCKg-RZfa5T))
* [Value Iteration Networks](https://arxiv.org/abs/1602.02867), Tamar A. et al , 2016. *VIN*. ([video](https://www.youtube.com/watch?v=RcRkog93ZRU))
* [Value Prediction Network](https://arxiv.org/abs/1707.03497), Oh J. et al, 2017. *VPN*.

### IL applications

* [ALVINN, an autonomous land vehicle in a neural network](https://papers.nips.cc/paper/95-alvinn-an-autonomous-land-vehicle-in-a-neural-network), Pomerleau D., 1989.
* [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316), Bojarski M. et al (NVIDIA), 2016. ([video](https://www.youtube.com/watch?v=qhUvQiKec2U))
* [End-to-End Deep Learning for Steering Autonomous Vehicles Considering Temporal Dependencies](https://arxiv.org/abs/1710.03804), Eraqi H. et al, 2017.

## Inverse Reinforcement Learning

* [Apprenticeship learning via inverse reinforcement learning](http://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf), Abbeel P. Ng A. 2004. *Projection*.
* [Bayesian inverse reinforcement learning](https://www.aaai.org/Papers/IJCAI/2007/IJCAI07-416.pdf), Ramachandran D. Amir E., 2007. *BIRL*.
* [Maximum Entropy Inverse Reinforcement Learning](https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf), Ziebart B. et al, 2008. *MEIRL*.
* [Maximum Entropy Deep Inverse Reinforcement Learning](https://arxiv.org/abs/1507.04888), Wulfmeier M., 2015. *MEDIRL*.
* [Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization](https://arxiv.org/abs/1603.00448), Finn C. et al, 2016. *GCL*. ([video](https://www.youtube.com/watch?v=hXxaepw0zAw))
* [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476), Ho J., Ermon S., 2016. *GAIL*.
* [Bridging the Gap Between Imitation Learning and Inverse Reinforcement Learning](http://ieeexplore.ieee.org/document/7464854/), Piot B. et al, 2017.

### IRL applications

* [Apprenticeship Learning for Motion Planning, with Application to Parking Lot Navigation](http://ieeexplore.ieee.org/document/4651222/), Abbeel P. et al, 2008.
* [Planning-based Prediction for Pedestrians](http://ieeexplore.ieee.org/abstract/document/5354147/), Ziebart B. et al, 2009. ([video](https://www.youtube.com/watch?v=XOZ69Bg4JKg))
* [Watch This: Scalable Cost-Function Learning for Path Planning in Urban Environments](https://arxiv.org/abs/1607.02329), Wulfmeier M., 2016. ([video](https://www.youtube.com/watch?v=Sdfir_1T-UQ))
* [Learning Driving Styles for Autonomous Vehicles from Demonstration](http://ieeexplore.ieee.org/document/7139555/), Kuderer M. et al, 2015.

# Optimal Control

## Control theory

* (book) [Model Predictive Control](http://een.iust.ac.ir/profs/Shamaghdari/MPC/Resources/), Camacho E., 1995.
* (book) [Predictive Control With Constraints](https://books.google.fr/books/about/Predictive_Control.html?id=HV_Y58c7KiwC&redir_esc=y), Maciejowski J. M., 2002.
* (book) [Constrained Control and Estimation](http://www.springer.com/gp/book/9781852335489),  Goodwin G., 2005.
* [A Generalized Path Integral Control Approach to Reinforcement Learning](http://www.jmlr.org/papers/volume11/theodorou10a/theodorou10a.pdf), Theodorou E. et al, 2010. *PI²*
* [Path Integral Policy Improvement with Covariance Matrix Adaptation](https://arxiv.org/abs/1206.4621), Stulp F., Sigaud O., 2010. *PI²-CMA*.
* [A generalized iterative LQG method for locally-optimal feedback control of constrained nonlinear stochastic systems](http://maeresearch.ucsd.edu/skelton/publications/weiwei_ilqg_CDC43.pdf), Todorov E., 2005. *iLQG*.
* [Synthesis and stabilization of complex behaviors through online trajectory optimization](https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf), Tassa Y., 2012. *iLQG+*.

### Dynamic programming

* (book) [Dynamic Programming](https://press.princeton.edu/titles/9234.html), Bellman R., 1957.
* (book) [Dynamic Programming and Optimal Control, Volumes 1 and 2](http://web.mit.edu/dimitrib/www/dpchapter.html), Bertsekas D., 1995.
* (book) [Markov Decision Processes - Discrete Stochastic Dynamic Programming](http://eu.wiley.com/WileyCDA/WileyTitle/productCd-1118625870.html), Puterman M., 1995.

# Motion Planning

## Search

* [3D perception and planning for self-driving and cooperative automobiles](http://www.mrt.kit.edu/z/publ/download/2012/StillerZiegler2012SSD.pdf), Stiller C., Ziegler J., 2012.
* [Motion Planning under Uncertainty for On-Road Autonomous Driving](https://www.ri.cmu.edu/pub_files/2014/6/ICRA14_0863_Final.pdf), Xu W. et al, 2014.
* [Monte Carlo Tree Search for Simulated Car Racing](http://julian.togelius.com/Fischer2015Monte.pdf), Fischer J. et al, 2015. ([video](https://www.youtube.com/watch?v=GbUMssvolvU))

## Sampling

* [Sampling-based Algorithms for Optimal Motion Planning](https://arxiv.org/abs/1105.1186), Karaman S., Frazzoli E., 2011. *RRT*\*. ([video](https://www.youtube.com/watch?v=p3nZHnOWhrg))
* [LQG-MP: Optimized Path Planning for Robots with Motion Uncertainty and Imperfect State Information](https://people.eecs.berkeley.edu/~pabbeel/papers/vandenBergAbbeelGoldberg_RSS2010.pdf), van den Berg J. et al, 2010.
* [Motion Planning under Uncertainty using Differential Dynamic Programming in Belief Space](http://rll.berkeley.edu/~sachin/papers/Berg-ISRR2011.pdf), van den Berg J. et al, 2011.
* [Rapidly-exploring Random Belief Trees for Motion Planning Under Uncertainty](https://groups.csail.mit.edu/rrg/papers/abry_icra11.pdf), Bry A., Roy N., 2011.

## MP applications

* [A Review of Motion Planning Techniques for Automated Vehicles](http://ieeexplore.ieee.org/document/7339478/), González D. et al, 2016.
* [A Survey of Motion Planning and Control Techniques for Self-driving Urban Vehicles](https://arxiv.org/abs/1604.07446), Paden B. et al, 2016.
* [Autonomous driving in urban environments: Boss and the Urban Challenge](https://www.ri.cmu.edu/publications/autonomous-driving-in-urban-environments-boss-and-the-urban-challenge/), Urmson C. et al, 2008.
* [The MIT-Cornell collision and why it happened](http://onlinelibrary.wiley.com/doi/10.1002/rob.20266/pdf), Fletcher L. et al, 2008.
* [Making bertha drive-an autonomous journey on a historic route](http://ieeexplore.ieee.org/document/6803933/), Ziegler J. et al, 2014.