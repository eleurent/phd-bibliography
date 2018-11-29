# Bibliography

# Table of contents

* [Optimal Control](#optimal-control)
  * [Dynamic Programming](#dynamic-programming)
  * [Approximate Planning](#approximate-planning)
  * [Control Theory](#control-theory)
  * [Model Predictive Control](#model-predictive-control)
* [Safe Control](#safe-control)
  * [Robust Control](robust-control)
  * [Risk-Averse Control](risk-averse-control)
  * [Constrained Control](constrained-control)
  * [Uncertain Dynamical Systems](uncertain-dynamical-systems)
* [Sequential Learning](#sequential-learning)
  * [Multi-Armed Bandit](#multi-armed-bandit)
  * [Reinforcement Learning](#reinforcement-learning)
    * [Value-based](#value-based)
    * [Policy-based](#policy-based)
      * [Policy Gradient](#policy-gradient)
      * [Actor-critic](#actor-critic)
      * [Derivative-free](#derivative-free)
    * [Model-based](#model-based)
    * [Temporal Abstraction](#temporal-abstraction)
    * [Partial Observability](#partial-observability)
    * [Transfer](#transfer)
    * [Multi-agent](#multi-agent)
    * [Representation](#representation)
* [Learning from Demonstrations](#learning-from-demonstrations)
  * [Imitation Learning](#imitation-learning)
    * [Applications to Autonomous Driving](#applications-to-autonomous-driving)
  * [Inverse Reinforcement Learning](#inverse-reinforcement-learning)
    * [Applications to Autonomous Driving](#applications-to-autonomous-driving)
* [Motion Planning](#motion-planning)
  * [Search](#search)
  * [Sampling](#sampling)
  * [Optimization](#optimization)
  * [Reactive](#reactive)
  * [Architecture and applications](#architecture-and-applications)

![RL Diagram](https://rawgit.com/eleurent/phd-bibliography/master/reinforcement-learning.svg)

# Optimal Control

## Dynamic Programming

* (book) [Dynamic Programming](https://press.princeton.edu/titles/9234.html), Bellman R. (1957).
* (book) [Dynamic Programming and Optimal Control, Volumes 1 and 2](http://web.mit.edu/dimitrib/www/dpchapter.html), Bertsekas D. (1995).
* (book) [Markov Decision Processes - Discrete Stochastic Dynamic Programming](http://eu.wiley.com/WileyCDA/WileyTitle/productCd-1118625870.html), Puterman M. (1995).

## Approximate Planning

* **`ExpectiMinimax`** [Optimal strategy in games with chance nodes](http://www.inf.u-szeged.hu/actacybernetica/edb/vol18n2/pdf/Melko_2007_ActaCybernetica.pdf), Melkó E., Nagy B. (2007).
* **`Sparse sampling`** [A sparse sampling algorithm for near-optimal planning in large Markov decision processes](https://www.cis.upenn.edu/~mkearns/papers/sparsesampling-journal.pdf), Kearns M. et al. (2002).
* **`MCTS`** [Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search](https://hal.inria.fr/inria-00116992/document), Rémi Coulom, *SequeL* (2006).
* **`UCT`** [Bandit based Monte-Carlo Planning](http://ggp.stanford.edu/readings/uct.pdf), Kocsis L., Szepesvári C. (2006).
* **`OPD`** [Optimistic Planning for Deterministic Systems](https://hal.inria.fr/hal-00830182), Hren J., Munos R. (2008).
* **`OLOP`** [Open Loop Optimistic Planning](http://sbubeck.com/COLT10_BM.pdf), Bubeck S., Munos R. (2010).
* **`AlphaGo`** [Mastering the game of Go with deep neural networks and tree search](https://www.nature.com/articles/nature16961), Silver D. et al. (2016).
* **`AlphaGo Zero`** [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270), Silver D. et al. (2017).
* **`AlphaZero`** [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815), Silver D. et al. (2017).
* **`TrailBlazer`** [Blazing the trails before beating the path: Sample-efficient Monte-Carlo planning](https://papers.nips.cc/paper/6253-blazing-the-trails-before-beating-the-path-sample-efficient-monte-carlo-planning.pdf), Grill J. B., Valko M., Munos R. (2017).
* **`MCTSnets`** [Learning to search with MCTSnets](https://arxiv.org/abs/1802.04697), Guez A. et al. (2018).
* **`ADI`** [Solving the Rubik's Cube Without Human Knowledge](https://arxiv.org/abs/1805.07470), McAleer S. et al. (2018).

## Control Theory

* (book) [Constrained Control and Estimation](http://www.springer.com/gp/book/9781852335489),  Goodwin G. (2005).
* **`PI²`** [A Generalized Path Integral Control Approach to Reinforcement Learning](http://www.jmlr.org/papers/volume11/theodorou10a/theodorou10a.pdf), Theodorou E. et al. (2010).
* **`PI²-CMA`** [Path Integral Policy Improvement with Covariance Matrix Adaptation](https://arxiv.org/abs/1206.4621), Stulp F., Sigaud O. (2010).
* **`iLQG`** [A generalized iterative LQG method for locally-optimal feedback control of constrained nonlinear stochastic systems](http://maeresearch.ucsd.edu/skelton/publications/weiwei_ilqg_CDC43.pdf), Todorov E. (2005).
* **`iLQG+`** [Synthesis and stabilization of complex behaviors through online trajectory optimization](https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf), Tassa Y. (2012).

## Model Predictive Control

* (book) [Model Predictive Control](http://een.iust.ac.ir/profs/Shamaghdari/MPC/Resources/), Camacho E. (1995).
* (book) [Predictive Control With Constraints](https://books.google.fr/books/about/Predictive_Control.html?id=HV_Y58c7KiwC&redir_esc=y), Maciejowski J. M. (2002).
* [Linear Model Predictive Control for Lane Keeping and Obstacle Avoidance on Low Curvature Roads](http://ieeexplore.ieee.org/document/6728261/), Turri V. et al. (2013).
* **`MPCC`** [Optimization-based autonomous racing of 1:43 scale RC cars](https://arxiv.org/abs/1711.07300), Liniger A. et al. (2014). ([video 1](https://www.youtube.com/watch?v=mXaElWYQKC4) | [2](https://www.youtube.com/watch?v=JoHfJ6LEKVo))
* **`MIQP`** [Optimal trajectory planning for autonomous driving integrating logical constraints: An MIQP perspective](https://hal.archives-ouvertes.fr/hal-01342358v1/document), Qian X., Altché F., Bender P., Stiller C. de La Fortelle A. (2016).



# Safe Control

## Robust Control

* [Minimax analysis of stochastic problems](https://www2.isye.gatech.edu/~anton/MinimaxSP.pdf), Shapiro A., Kleywegt A. (2002).
* **`Robust DP`** [Robust Dynamic Programming](https://www.researchgate.net/publication/220442530/download), Iyengar G. (2005).
* [Robust Planning and Optimization](https://www.researchgate.net/profile/Francisco_Perez-Galarce/post/can_anyone_recommend_a_report_or_article_on_two_stage_robust_optimization/attachment/59d62578c49f478072e9a500/AS%3A272164542976002%401441900491330/download/2011+-+Robust+planning+and+optimization.pdf), Laumanns M. (2011). (lecture notes)
* [Robust Markov Decision Processes](https://pubsonline.informs.org/doi/pdf/10.1287/moor.1120.0566), Wiesemann W., Kuhn D., Rustem B. (2012).
* **`Coarse-Id`** [On the Sample Complexity of the Linear Quadratic Regulator](https://arxiv.org/abs/1710.01688), Dean S., Mania H., Matni N., Recht B., Tu S. (2017).
* **`Tube-MPPI`** [Robust Sampling Based Model Predictive Control with Sparse Objective Information](http://www.roboticsproceedings.org/rss14/p42.pdf), Williams G. et al. (2018). ([video](https://www.youtube.com/watch?v=32v-e3dptjo))

## Risk-Averse Control

* [A Comprehensive Survey on Safe Reinforcement Learning](http://jmlr.org/papers/v16/garcia15a.html), García J., Fernández F. (2015).

## Constrained Control

* **`ICS`** [Will the Driver Seat Ever Be Empty?](https://hal.inria.fr/hal-00965176), Fraichard T. (2014).
* **`RSS`** [On a Formal Model of Safe and Scalable Self-driving Cars](https://arxiv.org/abs/1708.06374), Shalev-Shwartz S. et al. (2017).
* **`BFTQ`** [Safe Transfer across Reinforcement Learning Tasks](), Carrara N. et al. (2018).

## Uncertain Dynamical Systems

* [Simulation of Controlled Uncertain Nonlinear Systems](https://www.sciencedirect.com/science/article/pii/009630039400112H), Tibken B., Hofer E. (1995).
* [Trajectory computation of dynamic uncertain systems](https://ieeexplore.ieee.org/iel5/8969/28479/01272787.pdf), Adrot O., Flaus J-M. (2002).
* [Simulation of Uncertain Dynamic Systems Described By Interval Models: a Survey](https://www.sciencedirect.com/science/article/pii/S1474667016362206), Puig V. et al. (2005).
* [Design of interval observers for uncertain dynamical systems](https://hal.inria.fr/hal-01276439/file/Interval_Survey.pdf), Efimov D., Raïssi T. (2016).




# Sequential Learning

## Multi-Armed Bandit

* **`kl-UCB`** [The KL-UCB Algorithm for Bounded Stochastic Bandits and Beyond](https://arxiv.org/abs/1102.2490), Garivier A., Cappé O. (2011).
* **`KL-UCB`** [Kullback-Leibler Upper Confidence Bounds for Optimal Sequential Allocation](https://projecteuclid.org/euclid.aos/1375362558), Cappé O. et al. (2013).
* **`LUCB`** [PAC Subset Selection in Stochastic Multi-armed Bandits](https://www.cse.iitb.ac.in/~shivaram/papers/ktas_icml_2012.pdf), Kalyanakrishnan S. et al. (2012).
* **`Track-and-Stop`** [Optimal Best Arm Identification with Fixed Confidence](https://arxiv.org/abs/1602.04589), Garivier A., Kaufmann E. (2016).
* **`M-LUCB/M-Racing`** [Maximin Action Identification: A New Bandit Framework for Games](https://arxiv.org/abs/1602.04676), Garivier A., Kaufmann E., Koolen W. (2016).
* **`LUCB-micro`** [Structured Best Arm Identification with Fixed Confidence](https://arxiv.org/abs/1706.05198), Huang R. et al. (2017).




# Reinforcement Learning

* [Reinforcement learning: A survey](https://www.jair.org/media/301/live-301-1562-jair.pdf), Kaelbling L. et al. (1996).

## Value-based

* **`DQN`** [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), Mnih V. et al. (2013). ([video](https://www.youtube.com/watch?v=iqXKQf2BOSE))
* **`DDQN`** [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461), van Hasselt H., Silver D. et al. (2015).
* **`DDDQN`** [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581), Wang Z. et al. (2015). ([video](https://www.youtube.com/watch?v=qJd3yaEN9Sw))
* **`PDDDQN`** [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952), Schaul T. et al. (2015).
* **`NAF`** [Continuous Deep Q-Learning with Model-based Acceleration](https://arxiv.org/abs/1603.00748), Gu S. et al. (2016).
* **`Rainbow`** [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298), Hessel M. et al. (2017).
* **`Ape-X DQfD`** [Observe and Look Further: Achieving Consistent Performance on Atari](https://arxiv.org/abs/1805.11593), Pohlen T. et al. (2018). ([videos](https://www.youtube.com/watch?v=-0xOdnoxAFo&index=4&list=PLnZpNNVLsMmOfqMwJLcpLpXKLr3yKZ8Ak))

## Policy-based

### Policy gradient

* **`REINFORCE`** [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf), Williams R. (1992).
* **`Natural Gradient`** [A Natural Policy Gradient](https://papers.nips.cc/paper/2073-a-natural-policy-gradient.pdf), Kakade S. (2002).
* [Policy Gradient Methods for Robotics](http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/IROS2006-Peters_%5b0%5d.pdf), Peters J.,  Schaal S. (2006).
* **`TRPO`** [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477), Schulman J. et al. (2015). ([video](https://www.youtube.com/watch?v=KJ15iGGJFvQ))
* **`PPO`** [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), Schulman J. et al. (2017). ([video](https://www.youtube.com/watch?v=bqdjsmSoSgI))
* **`DPPO`** [Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/abs/1707.02286), Heess N. et al. (2017). ([video](https://www.youtube.com/watch?v=hx_bgoTF7bs))

### Actor-critic

* **`AC`** [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf), Sutton R. et al. (1999).
* **`NAC`** [Natural Actor-Critic](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/NaturalActorCritic.pdf), Peters J. et al. (2005).
* **`DPG`** [Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf), Silver D. et al. (2014).
* **`DDPG`** [Continuous Control With Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971), Lillicrap T. et al. (2015). ([video 1](https://www.youtube.com/watch?v=lV5JhxsrSH8) | [2](https://www.youtube.com/watch?v=8CNck-hdys8) | [3](https://www.youtube.com/watch?v=xw73qehvSRQ) | [4](https://www.youtube.com/watch?v=vWxBmHRnQMI))
* **`A3C`** [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783), Mnih V. et al 2016. ([video 1](https://www.youtube.com/watch?v=Ajjc08-iPx8) | [2](https://www.youtube.com/watch?v=0xo1Ldx3L5Q) | [3](https://www.youtube.com/watch?v=nMR5mjCFZCw))
* **`SAC`** [Soft Actor-Critic : Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290), Haarnoja T. et al. (2018). ([video](https://vimeo.com/252185258))

### Derivative-free

* **`CEM`** [Learning Tetris Using the Noisy Cross-Entropy Method](http://iew3.technion.ac.il/CE/files/papers/Learning%20Tetris%20Using%20the%20Noisy%20Cross-Entropy%20Method.pdf), Szita I., Lörincz A. (2006). ([video](https://www.youtube.com/watch?v=UZnDYGk1j2c))
* **`CMAES`** [Completely Derandomized Self-Adaptation in Evolution Strategies](https://dl.acm.org/citation.cfm?id=1108843), Hansen N., Ostermeier A. (2001).
* **`NEAT`** [Evolving Neural Networks through Augmenting Topologies](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf), Stanley K. (2002). ([video](https://www.youtube.com/watch?v=5lJuEW-5vr8))

## Model-based

* **`Dyna`** [Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.84.6983&rep=rep1&type=pdf), Sutton R. (1990).
* **`UCRL2`** [Near-optimal Regret Bounds for Reinforcement Learning](http://www.jmlr.org/papers/volume11/jaksch10a/jaksch10a.pdf), Jaksch T. (2010).
* **`PILCO`** [PILCO: A Model-Based and Data-Efficient Approach to Policy Search](http://mlg.eng.cam.ac.uk/pub/pdf/DeiRas11.pdf), Deisenroth M., Rasmussen C. (2011). ([talk](https://www.youtube.com/watch?v=f7y60SEZfXc))
* **`DBN`** [Probabilistic MDP-behavior planning for cars](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6082928), Brechtel S. et al. (2011).
* **`GPS`** [End-to-End Training of Deep Visuomotor Policies](https://arxiv.org/abs/1504.00702), Levine S. et al. (2015). ([video](https://www.youtube.com/watch?v=Q4bMcUk6pcw))
* **`DeepMPC`** [DeepMPC: Learning Deep Latent Features for Model Predictive Control](https://www.cs.stanford.edu/people/asaxena/papers/deepmpc_rss2015.pdf), Lenz I. et al. (2015). ([video](https://www.youtube.com/watch?v=BwA90MmkvPU))
* **`SVG`** [Learning Continuous Control Policies by Stochastic Value Gradients](https://arxiv.org/abs/1510.09142), Heess N. et al. (2015). ([video](https://www.youtube.com/watch?v=PYdL7bcn_cM))
* [Optimal control with learned local models: Application to dexterous manipulation](https://homes.cs.washington.edu/~todorov/papers/KumarICRA16.pdf), Kumar V. et al. (2016). ([video](https://www.youtube.com/watch?v=bD5z1I1TU3w))
* **`BPTT`** [Long-term Planning by Short-term Prediction](https://arxiv.org/abs/1602.01580), Shalev-Shwartz S. et al. (2016). ([video 1](https://www.youtube.com/watch?v=Nqmv1anUaF4) | [2](https://www.youtube.com/watch?v=UgGZ9lMvey8))
* [Deep visual foresight for planning robot motion](https://arxiv.org/abs/1610.00696), Finn C., Levine S. (2016). ([video](https://www.youtube.com/watch?v=6k7GHG4IUCY))
* **`VIN`** [Value Iteration Networks](https://arxiv.org/abs/1602.02867), Tamar A. et al  (2016). ([video](https://www.youtube.com/watch?v=RcRkog93ZRU))
* **`VPN`** [Value Prediction Network](https://arxiv.org/abs/1707.03497), Oh J. et al. (2017).
* [An LSTM Network for Highway Trajectory Prediction](https://arxiv.org/abs/1801.07962), Altché F., de La Fortelle A. (2017).
* **`DistGBP`** [Model-Based Planning with Discrete and Continuous Actions](https://arxiv.org/abs/1705.07177), Henaff M. et al. (2017). ([video 1](https://www.youtube.com/watch?v=9Xh2TRQ_4nM) | [2](https://www.youtube.com/watch?v=XLdme0TTjiw))
* [Prediction and Control with Temporal Segment Models](https://arxiv.org/abs/1703.04070), Mishra N. et al. (2017).
* **`Predictron`** [The Predictron: End-To-End Learning and Planning](https://arxiv.org/abs/1612.08810), Silver D. et al. (2017). ([video](https://www.youtube.com/watch?v=BeaLdaN2C3Q))
* **`MPPI`** [Information Theoretic MPC for Model-Based Reinforcement Learning](https://ieeexplore.ieee.org/document/7989202/), Williams G. et al. (2017). ([video](https://www.youtube.com/watch?v=f2at-cqaJMM))
* [Learning Real-World Robot Policies by Dreaming](https://arxiv.org/abs/1805.07813), Piergiovanni A. et al. (2018).

## Temporal Abstraction

* [Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning](http://www-anw.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf), Sutton R. et al. (1999).
* [Intrinsically motivated learning of hierarchical collections of skills](http://www-anw.cs.umass.edu/pubs/2004/barto_sc_ICDL04.pdf), Barto A. et al. (2004).
* [Learning and Transfer of Modulated Locomotor Controllers](https://arxiv.org/abs/1610.05182), Heess N. et al. (2016). ([video](https://www.youtube.com/watch?v=sboPYvhpraQ&feature=youtu.be))
* [Safe, Multi-Agent, Reinforcement Learning for Autonomous Driving](https://arxiv.org/abs/1610.03295), Shalev-Shwartz S. et al. (2016).
* **`FuNs`** [FeUdal Networks for Hierarchical Reinforcement Learning](https://arxiv.org/abs/1703.01161), Vezhnevets A. et al. (2017).
* [Combining Neural Networks and Tree Search for Task and Motion Planning in Challenging Environments](https://arxiv.org/abs/1703.07887), Paxton C. et al. (2017). ([video](https://www.youtube.com/watch?v=MM2U_SGMtk8))

## Partial Observability

* **`PBVI`** [Point-based Value Iteration: An anytime algorithm for POMDPs](https://www.ri.cmu.edu/pub_files/pub4/pineau_joelle_2003_3/pineau_joelle_2003_3.pdf), Pineau J. et al. (2003).
* **`cPBVI`** [Point-Based Value Iteration for Continuous POMDPs](http://www.jmlr.org/papers/volume7/porta06a/porta06a.pdf), Porta J. et al. (2006).
* **`POMCP`** [Monte-Carlo Planning in Large POMDPs](https://papers.nips.cc/paper/4031-monte-carlo-planning-in-large-pomdps), Silver D., Veness J. (2010).
* [A POMDP Approach to Robot Motion Planning under Uncertainty](http://users.isr.ist.utl.pt/~mtjspaan/POMDPPractioners/pomdp2010_submission_5.pdf), Du Y. et al. (2010).
* [Solving Continuous POMDPs: Value Iteration with Incremental Learning of an Efficient Space Representation](http://proceedings.mlr.press/v28/brechtel13.pdf), Brechtel S. et al. (2013).
* [Probabilistic Decision-Making under Uncertainty for Autonomous Driving using Continuous POMDPs](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6957722), Brechtel S. et al. (2014).
* **`MOMDP`** [Intention-Aware Motion Planning](http://ares.lids.mit.edu/fm/documents/intentionawaremotionplanning.pdf), Bandyopadhyay T. et al. (2013).
* [The value of inferring the internal state of traffic participants for autonomous freeway driving](https://arxiv.org/abs/1702.00858), Sunberg Z. et al. (2017).

## Transfer

* [Virtual to Real Reinforcement Learning for Autonomous Driving](https://arxiv.org/abs/1704.03952), Pan X. et al. (2017). ([video](https://www.youtube.com/watch?v=Bce2ZSlMuqY))
* [Sim-to-Real: Learning Agile Locomotion For Quadruped Robots](https://arxiv.org/abs/1804.10332), Tan J. et al. (2018). ([video](https://www.youtube.com/watch?v=lUZUr7jxoqM))
* **`ME-TRPO`** [Model-Ensemble Trust-Region Policy Optimization](https://arxiv.org/abs/1802.10592), Kurutach T. et al. (2018). ([video](https://www.youtube.com/watch?v=tpS8qj7yhoU))
* [Kickstarting Deep Reinforcement Learning](https://arxiv.org/abs/1803.03835), Schmitt S. et al. (2018).
* [Learning Dexterous In-Hand Manipulation](https://blog.openai.com/learning-dexterity/), OpenAI (2018). ([video](https://www.youtube.com/watch?v=DKe8FumoD4E))


## Multi-agent

* [Autonomous Agents Modelling Other Agents: A Comprehensive Survey and Open Problems](https://arxiv.org/abs/1709.08071), Albrecht S., Stone P. (2017).
* **`MILP`** [Time-optimal coordination of mobile robots along specified paths](https://arxiv.org/abs/1603.04610), Altché F. et al. (2016). ([video](https://www.youtube.com/watch?v=RiW2OFsdHOY))
* **`MIQP`** [An Algorithm for Supervised Driving of Cooperative Semi-Autonomous Vehicles](https://arxiv.org/abs/1706.08046), Altché F. et al. (2017). ([video](https://www.youtube.com/watch?v=JJZKfHMUeCI))
* **`SA-CADRL`** [Socially Aware Motion Planning with Deep Reinforcement Learning](https://arxiv.org/abs/1703.08862), Chen Y. et al. (2017). ([video](https://www.youtube.com/watch?v=CK1szio7PyA))
* [Multipolicy decision-making for autonomous driving via changepoint-based behavior prediction: Theory and experiment](https://link.springer.com/article/10.1007/s10514-017-9619-z), Galceran E. et al. (2017).
* [Online decision-making for scalable autonomous systems](https://www.ijcai.org/proceedings/2017/664), Wray K. et al. (2017).
* **`MAgent`** [MAgent: A Many-Agent Reinforcement Learning Platform for Artificial Collective Intelligence](https://arxiv.org/abs/1712.00600), Zheng L. et al. (2017). ([video](https://www.youtube.com/watch?v=HCSm0kVolqI))
* [Cooperative Motion Planning for Non-Holonomic Agents with Value Iteration Networks](https://arxiv.org/abs/1709.05273), Rehder E. et al. (2017).
* **`COMA`** [Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1709.05273), Foerster J. et al. (2017).
* **`FTW`** [Human-level performance in first-person multiplayer games with population-based deep reinforcement learning](https://arxiv.org/abs/1807.01281), Jaderberg M. et al. (2018). ([video](https://www.youtube.com/watch?v=dltN4MxV1RI))

## Representation

* **`DeepDriving`** [DeepDriving: Learning Affordance for Direct Perception in Autonomous Driving](http://deepdriving.cs.princeton.edu/paper.pdf), Chen C. et al. (2015). ([video](https://www.youtube.com/watch?v=5hFvoXV9gII))
* [On the Sample Complexity of End-to-end Training vs. Semantic Abstraction Training](https://arxiv.org/abs/1604.06915), Shalev-Shwartz S. et al. (2016).
* **`VAE-MDN-RNN`** [World Models](https://worldmodels.github.io/), Ha D., Schmidhuber J. (2018).
* **`MERLIN`** [Unsupervised Predictive Memory in a Goal-Directed Agent](https://arxiv.org/abs/1803.10760), Wayne G. et al. (2018). ([video 1](https://www.youtube.com/watch?v=YFx-D4eEs5A) | [2](https://www.youtube.com/watch?v=IiR_NOomcpk) | [3](https://www.youtube.com/watch?v=dQMKJtLScmk) | [4](https://www.youtube.com/watch?v=xrYDlTXyC6Q) | [5](https://www.youtube.com/watch?v=04H28-qA3f8) [6](https://www.youtube.com/watch?v=3iA19h0Vvq0))

## Other

* [Is the Bellman residual a bad proxy?](https://arxiv.org/abs/1606.07636), Geist M., Piot B., Pietquin O. (2016).
* [Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560), Henderson P. et al. (2017).
* [Automatic Bridge Bidding Using Deep Reinforcement Learning](https://arxiv.org/abs/1607.03290), Yeh C. and Lin H. (2016).
* [Shared Autonomy via Deep Reinforcement Learning](https://arxiv.org/abs/1802.01744), Reddy S. et al. (2018). ([videos](https://sites.google.com/view/deep-assist))
* [Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review](https://arxiv.org/abs/1805.00909), Levine S. (2018).


# Learning from Demonstrations

## Imitation Learning

* **`DQfD`** [Learning from Demonstrations for Real World Reinforcement Learning](https://pdfs.semanticscholar.org/a7fb/199f85943b3fb6b5f7e9f1680b2e2a445cce.pdf), Hester T. et al. (2017). ([videos](https://www.youtube.com/watch?v=JR6wmLaYuu4&list=PLdjpGm3xcO-0aqVf--sBZHxCKg-RZfa5T))
* **`UPN`** [Universal Planning Networks](https://arxiv.org/abs/1804.00645), Srinivas A. et al. (2018). ([videos](https://sites.google.com/view/upn-public/home))
* **`QMDP-RCNN`** [Reinforcement Learning via Recurrent Convolutional Neural Networks](https://arxiv.org/abs/1701.02392), Shankar T. et al. (2016). ([talk](https://www.youtube.com/watch?v=gpwA3QNTPOQ))
* **`GAIL`** [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476), Ho J., Ermon S. (2016).
* [From perception to decision: A data-driven approach to end-to-end motion planning for autonomous ground robots](https://arxiv.org/abs/1609.07910), Pfeiffer M. et al. (2017). ([video](https://www.youtube.com/watch?v=ZedKmXzwdgI))
* **`Branched`** [End-to-end Driving via Conditional Imitation Learning](https://arxiv.org/abs/1710.02410), Codevilla F. et al. (2017). ([video](https://www.youtube.com/watch?v=cFtnflNe5fM) | [talk](https://www.youtube.com/watch?v=KunVjVHN3-U))
* **`DeepMimic`** [DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills](https://xbpeng.github.io/projects/DeepMimic/index.html), Peng X. B. et al. (2018). ([video](https://www.youtube.com/watch?v=vppFvq2quQ0&feature=youtu.be))

### Applications to Autonomous Driving

* [ALVINN, an autonomous land vehicle in a neural network](https://papers.nips.cc/paper/95-alvinn-an-autonomous-land-vehicle-in-a-neural-network), Pomerleau D. (1989).
* [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316), Bojarski M. et al. (2016). ([video](https://www.youtube.com/watch?v=qhUvQiKec2U))
* [End-to-end Learning of Driving Models from Large-scale Video Datasets](https://arxiv.org/abs/1612.01079), Xu H., Gao Y. et al. (2016). ([video](https://www.youtube.com/watch?v=jxlNfUzbGAY))
* [End-to-End Deep Learning for Steering Autonomous Vehicles Considering Temporal Dependencies](https://arxiv.org/abs/1710.03804), Eraqi H. et al. (2017).
* [Driving Like a Human: Imitation Learning for Path Planning using Convolutional Neural Networks](https://www.semanticscholar.org/paper/Driving-Like-a-Human%3A-Imitation-Learning-for-Path-Rehder-Quehl/a1150417083918c3f5f88b7ddad8841f2ce88188), Rehder E. et al. (2017).
* [Imitating Driver Behavior with Generative Adversarial Networks](https://arxiv.org/abs/1701.06699), Kuefler A. et al. (2017).
* **`PS-GAIL`** [Multi-Agent Imitation Learning for Driving Simulation](https://arxiv.org/abs/1803.01044), Bhattacharyya R. et al. (2018). ([video](https://github.com/sisl/ngsim_env/blob/master/media/single_multi_model_2_seed_1.gif))

## Inverse Reinforcement Learning

* **`Projection`** [Apprenticeship learning via inverse reinforcement learning](http://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf), Abbeel P., Ng A. (2004).
* **`MMP`** [Maximum margin planning](https://www.ri.cmu.edu/pub_files/pub4/ratliff_nathan_2006_1/ratliff_nathan_2006_1.pdf), Ratliff N. et al. (2006).
* **`BIRL`** [Bayesian inverse reinforcement learning](https://www.aaai.org/Papers/IJCAI/2007/IJCAI07-416.pdf), Ramachandran D., Amir E. (2007).
* **`MEIRL`** [Maximum Entropy Inverse Reinforcement Learning](https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf), Ziebart B. et al. (2008).
* **`CIOC`** [Continuous Inverse Optimal Control with Locally Optimal Examples](http://graphics.stanford.edu/projects/cioc/), Levine S., Koltun V. (2012). ([video](http://graphics.stanford.edu/projects/cioc/cioc.mp4))
* **`MEDIRL`** [Maximum Entropy Deep Inverse Reinforcement Learning](https://arxiv.org/abs/1507.04888), Wulfmeier M. (2015).
* **`GCL`** [Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization](https://arxiv.org/abs/1603.00448), Finn C. et al. (2016). ([video](https://www.youtube.com/watch?v=hXxaepw0zAw))
* **`RIRL`** [Repeated Inverse Reinforcement Learning](https://arxiv.org/abs/1705.05427), Amin K. et al. (2017).
* [Bridging the Gap Between Imitation Learning and Inverse Reinforcement Learning](http://ieeexplore.ieee.org/document/7464854/), Piot B. et al. (2017).

### Applications to Autonomous Driving

* [Apprenticeship Learning for Motion Planning, with Application to Parking Lot Navigation](http://ieeexplore.ieee.org/document/4651222/), Abbeel P. et al. (2008).
* [Navigate like a cabbie: Probabilistic reasoning from observed context-aware behavior](http://www.cs.cmu.edu/~bziebart/publications/navigate-bziebart.pdf), Ziebart B. et al. (2008).
* [Planning-based Prediction for Pedestrians](http://ieeexplore.ieee.org/abstract/document/5354147/), Ziebart B. et al. (2009). ([video](https://www.youtube.com/watch?v=XOZ69Bg4JKg))
* [Learning Driving Styles for Autonomous Vehicles from Demonstration](http://ieeexplore.ieee.org/document/7139555/), Kuderer M. et al. (2015).
* [Learning to Drive using Inverse Reinforcement Learning and Deep Q-Networks](https://arxiv.org/abs/1612.03653), Sharifzadeh S. et al. (2016).
* [Watch This: Scalable Cost-Function Learning for Path Planning in Urban Environments](https://arxiv.org/abs/1607.02329), Wulfmeier M. (2016). ([video](https://www.youtube.com/watch?v=Sdfir_1T-UQ))
* [Planning for Autonomous Cars that Leverage Effects on Human Actions](https://robotics.eecs.berkeley.edu/~sastry/pubs/Pdfs%20of%202016/SadighPlanning2016.pdf), Sadigh D. et al. (2016).
* [A Learning-Based Framework for Handling Dilemmas in Urban Automated Driving](http://ieeexplore.ieee.org/document/7989172/), Lee S., Seo S. (2017).



# Motion Planning

## Search

* **`Dijkstra`** [A Note on Two Problems in Connexion with Graphs](http://www-m3.ma.tum.de/foswiki/pub/MN0506/WebHome/dijkstra.pdf), Dijkstra E. W. (1959).
* **`A*`** [ A Formal Basis for the Heuristic Determination of Minimum Cost Paths ](http://ieeexplore.ieee.org/document/4082128/), Hart P. et al. (1968).
* [Planning Long Dynamically-Feasible Maneuvers For Autonomous Vehicles](https://www.cs.cmu.edu/~maxim/files/planlongdynfeasmotions_rss08.pdf), Likhachev M., Ferguson D. (2008).
* [Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame](https://www.researchgate.net/profile/Moritz_Werling/publication/224156269_Optimal_Trajectory_Generation_for_Dynamic_Street_Scenarios_in_a_Frenet_Frame/links/54f749df0cf210398e9277af.pdf), Werling M., Kammel S. (2010). ([video](https://www.youtube.com/watch?v=Cj6tAQe7UCY))
* [3D perception and planning for self-driving and cooperative automobiles](http://www.mrt.kit.edu/z/publ/download/2012/StillerZiegler2012SSD.pdf), Stiller C., Ziegler J. (2012).
* [Motion Planning under Uncertainty for On-Road Autonomous Driving](https://www.ri.cmu.edu/pub_files/2014/6/ICRA14_0863_Final.pdf), Xu W. et al. (2014).
* [Monte Carlo Tree Search for Simulated Car Racing](http://julian.togelius.com/Fischer2015Monte.pdf), Fischer J. et al. (2015). ([video](https://www.youtube.com/watch?v=GbUMssvolvU))

## Sampling

* **`RRT*`** [Sampling-based Algorithms for Optimal Motion Planning](https://arxiv.org/abs/1105.1186), Karaman S., Frazzoli E. (2011). ([video](https://www.youtube.com/watch?v=p3nZHnOWhrg))
* **`LQG-MP`** [LQG-MP: Optimized Path Planning for Robots with Motion Uncertainty and Imperfect State Information](https://people.eecs.berkeley.edu/~pabbeel/papers/vandenBergAbbeelGoldberg_RSS2010.pdf), van den Berg J. et al. (2010).
* [Motion Planning under Uncertainty using Differential Dynamic Programming in Belief Space](http://rll.berkeley.edu/~sachin/papers/Berg-ISRR2011.pdf), van den Berg J. et al. (2011).
* [Rapidly-exploring Random Belief Trees for Motion Planning Under Uncertainty](https://groups.csail.mit.edu/rrg/papers/abry_icra11.pdf), Bry A., Roy N. (2011).
* **`PRM-RL`** [PRM-RL: Long-range Robotic Navigation Tasks by Combining Reinforcement Learning and Sampling-based Planning](https://arxiv.org/abs/1710.03937), Faust A. et al. (2017).

## Optimization

* [Trajectory planning for Bertha - A local, continuous method](https://pdfs.semanticscholar.org/bdca/7fe83f8444bb4e75402a417053519758d36b.pdf), Ziegler J. et al. (2014).
* [Learning Attractor Landscapes for Learning Motor Primitives](https://papers.nips.cc/paper/2140-learning-attractor-landscapes-for-learning-motor-primitives.pdf), Ijspeert A. et al. (2002).

## Reactive

* **`PF`** [Real-time obstacle avoidance for manipulators and mobile robots](http://ieeexplore.ieee.org/document/1087247/), Khatib O. (1986).
* **`VFH`** [The Vector Field Histogram - Fast Obstacle Avoidance For Mobile Robots](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=88137), Borenstein J. (1991).
* **`VFH+`** [VFH+: Reliable Obstacle Avoidance for Fast Mobile Robots](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.438.3464&rep=rep1&type=pdf), Ulrich I., Borenstein J. (1998).
* **`Velocity Obstacles`** [Motion planning in dynamic environments using velocity obstacles](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.56.6352&rep=rep1&type=pdf), Fiorini P., Shillert Z. (1998).

## Architecture and applications

* [A Review of Motion Planning Techniques for Automated Vehicles](http://ieeexplore.ieee.org/document/7339478/), González D. et al. (2016).
* [A Survey of Motion Planning and Control Techniques for Self-driving Urban Vehicles](https://arxiv.org/abs/1604.07446), Paden B. et al. (2016).
* [Autonomous driving in urban environments: Boss and the Urban Challenge](https://www.ri.cmu.edu/publications/autonomous-driving-in-urban-environments-boss-and-the-urban-challenge/), Urmson C. et al. (2008).
* [The MIT-Cornell collision and why it happened](http://onlinelibrary.wiley.com/doi/10.1002/rob.20266/pdf), Fletcher L. et al. (2008).
* [Making bertha drive-an autonomous journey on a historic route](http://ieeexplore.ieee.org/document/6803933/), Ziegler J. et al. (2014).