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
  * [Performances and acceleration](#performances-and-acceleration)
  * [Multi-agent](#multi-agent)
* [Learning from Demonstrations](#learning-from-demonstrations)
  * [Imitation Learning](#imitation-learning)
    * [IL applications](#il-applications)
  * [Inverse Reinforcement Learning](#inverse-reinforcement-learning)
    * [IRL applications](#irl-applications)
* [Optimal Control](#optimal-control)
  * [Control Theory](#control-theory)
  * [Dynamic Programming](#dynamic-programming)

![RL Diagram](https://rawgit.com/eleurent/phd-bibliography/master/reinforcement-learning.svg)

## Reinforcement Learning

### Value-based

* [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), Mnih V. et al, *DQN*, 2013.
* [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461), van Hasselt H. Silver D. et al, *DDQN*, 2015.
* [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581), Wang Z. et al, *DDDQN*, 2015.
* [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952), Schaul T. et al, 2015.
* [Continuous Deep Q-Learning with Model-based Acceleration](https://arxiv.org/abs/1603.00748), Gu S. et al, 2016.

### Policy-based

#### Policy gradient

* [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf), Williams R., *REINFORCE*, 1992.
* [End-to-End Training of Deep Visuomotor Policies](https://arxiv.org/abs/1504.00702), Levine S. et al, *GPS*, 2015.
* [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477), Schulman J. et al, *TRPO*, 2015.
* [Proximal Policy Optimization Algorithms ](https://arxiv.org/abs/1707.06347), Schulman J. et al, *PPO*, 2017.

#### Actor-critic

* [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf), Sutton R. et al, *Actor-critic*, 1999.
* [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783), Mnih V. et al, *A3C* 2016.
* [Continuous Control With Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971), Lillicrap T. et al, *DDPG*, 2016.

#### Derivative-free

* [Learning Tetris Using the Noisy Cross-Entropy Method](http://iew3.technion.ac.il/CE/files/papers/Learning%20Tetris%20Using%20the%20Noisy%20Cross-Entropy%20Method.pdf), Szita I. Lörincz A., *CEM*, 2006.
* [Completely Derandomized Self-Adaptation in Evolution Strategies](https://dl.acm.org/citation.cfm?id=1108843), Hansen N. Ostermeier A., *CMAES*, 2001.
* [Evolving Neural Networks through Augmenting Topologies](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf), Stanley K., *NEAT*, 2002.

### Temporal abstraction

* [Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning](http://www-anw.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf), Sutton R. et al, 1999.
* [Learning and Transfer of Modulated Locomotor Controllers](https://arxiv.org/abs/1610.05182), Heess N. et al, 2016.
* [On a Formal Model of Safe and Scalable Self-driving Cars](https://arxiv.org/abs/1708.06374), Shalev-Shwartz S. et al, 2017.

### Partial observability

* [Point-based Value Iteration: An anytime algorithm for POMDPs](https://www.ri.cmu.edu/pub_files/pub4/pineau_joelle_2003_3/pineau_joelle_2003_3.pdf), Pineau J. et al, 2003.
* [Point-Based Value Iteration for Continuous POMDPs](http://www.jmlr.org/papers/volume7/porta06a/porta06a.pdf), Porta J. et al, 2006.

### Performances and acceleration

* [Near-optimal Regret Bounds for Reinforcement Learning](http://www.jmlr.org/papers/volume11/jaksch10a/jaksch10a.pdf), Jaksch T., *UCRL2*, ‎2010.
* [Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.84.6983&rep=rep1&type=pdf), Sutton R., *Dyna*, 1990.

### Multi-agent

* [MAgent: A Many-Agent Reinforcement Learning Platform for Artificial Collective Intelligence](https://arxiv.org/abs/1712.00600), Zheng L. et al, 2017.
* [Safe, Multi-Agent, Reinforcement Learning for Autonomous Driving](https://arxiv.org/abs/1610.03295), Shalev-Shwartz S. et al, 2016.


## Learning from Demonstrations

### Imitation Learning

* [Learning from Demonstrations for Real World Reinforcement Learning](https://pdfs.semanticscholar.org/a7fb/199f85943b3fb6b5f7e9f1680b2e2a445cce.pdf), Hester T. et al, *DQfD*, 2017.
* [Value Iteration Networks](https://arxiv.org/abs/1602.02867), Tamar A. et al , 2016.
* [Value Prediction Network](https://arxiv.org/abs/1707.03497), Oh J. et al, 2017

### IL applications

* [ALVINN, an autonomous land vehicle in a neural network](https://papers.nips.cc/paper/95-alvinn-an-autonomous-land-vehicle-in-a-neural-network), Pomerleau D., 1989.
* [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316), Bojarski M. et al (NVIDIA), 2016.
* [End-to-End Deep Learning for Steering Autonomous Vehicles Considering Temporal Dependencies](https://arxiv.org/abs/1710.03804), Eraqi H. et al, 2017.

### Inverse Reinforcement Learning

* [Apprenticeship learning via inverse reinforcement learning](http://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf), Abbeel P. Ng A., *Projection* 2004.
* [Bayesian inverse reinforcement learning](https://www.aaai.org/Papers/IJCAI/2007/IJCAI07-416.pdf), Ramachandran D. Amir E., *BIRL*, 2007.
* [Maximum Entropy Inverse Reinforcement Learning](https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf), Ziebart B. et al, *MEIRL*, 2008.
* [Maximum Entropy Deep Inverse Reinforcement Learning](https://arxiv.org/abs/1507.04888), Wulfmeier M., *MEDIRL*, 2015.
* [Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization](https://arxiv.org/abs/1603.00448), Finn C. et al, *GCL*, 2016.
* [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476), Ho J., Ermon S., *GAIL*, 2016.
* [Bridging the Gap Between Imitation Learning and Inverse Reinforcement Learning](http://ieeexplore.ieee.org/document/7464854/), Piot B. et al, 2017.

#### IRL applications

* [Apprenticeship Learning for Motion Planning, with Application to Parking Lot Navigation](http://ieeexplore.ieee.org/document/4651222/), Abbeel P. et al, 2008.
* [Planning-based Prediction for Pedestrians](http://ieeexplore.ieee.org/abstract/document/5354147/), Ziebart B. et al, 2009.
* [Watch This: Scalable Cost-Function Learning for Path Planning in Urban Environments](https://arxiv.org/abs/1607.02329), Wulfmeier M., 2016.
* [Learning Driving Styles for Autonomous Vehicles from Demonstration](http://ieeexplore.ieee.org/document/7139555/), Kuderer M. et al, 2015.

## Optimal Control

### Control theory

* (book) [Model Predictive Control](http://een.iust.ac.ir/profs/Shamaghdari/MPC/Resources/), Camacho E., 1995.
* (book) [Predictive Control With Constraints](https://books.google.fr/books/about/Predictive_Control.html?id=HV_Y58c7KiwC&redir_esc=y), Maciejowski J. M., 2002.
* (book) [Constrained Control and Estimation](http://www.springer.com/gp/book/9781852335489),  Goodwin G., 2005.
* [A Generalized Path Integral Control Approach to Reinforcement Learning](http://www.jmlr.org/papers/volume11/theodorou10a/theodorou10a.pdf), Theodorou E. et al, *PI²*, 2010.
* [Path Integral Policy Improvement with Covariance Matrix Adaptation](https://arxiv.org/abs/1206.4621), Stulp F., Sigaud O., *PI²-CMA*, 2010.
* [A generalized iterative LQG method for locally-optimal feedback control of constrained nonlinear stochastic systems](http://maeresearch.ucsd.edu/skelton/publications/weiwei_ilqg_CDC43.pdf), Todorov E., *iLQG*, 2005.
* [Synthesis and stabilization of complex behaviors through online trajectory optimization](https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf), Tassa Y., *iLQG+*, 2012.

### Dynamic programming

* (book) [Dynamic Programming](https://press.princeton.edu/titles/9234.html), Bellman R., 1957.
* (book) [Dynamic Programming and Optimal Control, Volumes 1 and 2](http://web.mit.edu/dimitrib/www/dpchapter.html), Bertsekas D., 1995.
* (book) [Markov Decision Processes - Discrete Stochastic Dynamic Programming](http://eu.wiley.com/WileyCDA/WileyTitle/productCd-1118625870.html), Puterman M., 1995.
