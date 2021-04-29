# Bibliography

# Table of contents

* [Optimal Control](#optimal-control-dart)
  * [Dynamic Programming](#dynamic-programming)
  * [Tree-Based Planning](#tree-based-planning)
  * [Control Theory](#control-theory)
  * [Model Predictive Control](#model-predictive-control)
* [Safe Control](#safe-control-lock)
  * [Robust Control](#robust-control)
  * [Risk-Averse Control](#risk-averse-control)
  * [Value-Constrained Control](#value-constrained-control)
  * [State-Constrained Control and Stability](#state-constrained-control-and-stability)
  * [Uncertain Dynamical Systems](#uncertain-dynamical-systems)
* [Game Theory](#game-theory-spades)
* [Sequential Learning](#sequential-learning-shoe)
  * [Multi-Armed Bandit](#multi-armed-bandit-slot_machine)
    * [Best Arm Identification](#best-arm-identification-muscle)
    * [Black-box Optimization](#black-box-optimization-black_large_square)
  * [Reinforcement Learning](#reinforcement-learning-robot)
    * [Theory](#theory-books)
    * [Value-based](#value-based-chart_with_upwards_trend)
    * [Policy-based](#policy-based-muscle)
      * [Policy Gradient](#policy-gradient)
      * [Actor-critic](#actor-critic)
      * [Derivative-free](#derivative-free)
    * [Model-based](#model-based-world_map)
    * [Exploration](#exploration-tent)
    * [Hierarchy and Temporal Abstraction](#hierarchy-and-temporal-abstraction-clock2)
    * [Partial Observability](#partial-observability-eye)
    * [Transfer](#transfer-earth_americas)
    * [Multi-agent](#multi-agent-two_men_holding_hands)
    * [Representation Learning](#representation-learning)
* [Learning from Demonstrations](#learning-from-demonstrations-mortar_board)
  * [Imitation Learning](#imitation-learning)
    * [Applications to Autonomous Driving](#applications-to-autonomous-driving-car)
  * [Inverse Reinforcement Learning](#inverse-reinforcement-learning)
    * [Applications to Autonomous Driving](#applications-to-autonomous-driving-taxi)
* [Motion Planning](#motion-planning-running_man)
  * [Search](#search)
  * [Sampling](#sampling)
  * [Optimization](#optimization)
  * [Reactive](#reactive)
  * [Architecture and applications](#architecture-and-applications)

![RL Diagram](https://rawgit.com/eleurent/phd-bibliography/master/reinforcement-learning.svg)

# Optimal Control :dart:

## Dynamic Programming

* (book) [Dynamic Programming](https://press.princeton.edu/titles/9234.html), Bellman R. (1957).
* (book) [Dynamic Programming and Optimal Control, Volumes 1 and 2](http://web.mit.edu/dimitrib/www/dpchapter.html), Bertsekas D. (1995).
* (book) [Markov Decision Processes - Discrete Stochastic Dynamic Programming](http://eu.wiley.com/WileyCDA/WileyTitle/productCd-1118625870.html), Puterman M. (1995).
* [An Upper Bound on the Loss from Approximate Optimal-Value Functions](https://www.cis.upenn.edu/~mkearns/teaching/cis620/papers/SinghYee.pdf), Singh S., Yee R. (1994).
* [Stochastic optimization of sailing trajectories in an upwind regatta](https://link.springer.com/article/10.1057%2Fjors.2014.40), Dalang R. et al. (2015).

## Tree-Based Planning

* **`ExpectiMinimax`** [Optimal strategy in games with chance nodes](http://www.inf.u-szeged.hu/actacybernetica/edb/vol18n2/pdf/Melko_2007_ActaCybernetica.pdf), Melkó E., Nagy B. (2007).
* **`Sparse sampling`** [A sparse sampling algorithm for near-optimal planning in large Markov decision processes](https://www.cis.upenn.edu/~mkearns/papers/sparsesampling-journal.pdf), Kearns M. et al. (2002).
* **`MCTS`** [Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search](https://hal.inria.fr/inria-00116992/document), Rémi Coulom, *SequeL* (2006).
* **`UCT`** [Bandit based Monte-Carlo Planning](http://ggp.stanford.edu/readings/uct.pdf), Kocsis L., Szepesvári C. (2006).
* [Bandit Algorithms for Tree Search](https://hal.inria.fr/inria-00136198v2), Coquelin P-A., Munos R. (2007).
* **`OPD`** [Optimistic Planning for Deterministic Systems](https://hal.inria.fr/hal-00830182), Hren J., Munos R. (2008).
* **`OLOP`** [Open Loop Optimistic Planning](http://sbubeck.com/COLT10_BM.pdf), Bubeck S., Munos R. (2010).
* **`SOOP`** [Optimistic Planning for Continuous-Action Deterministic Systems](http://researchers.lille.inria.fr/munos/papers/files/adprl13-soop.pdf), Buşoniu L. et al. (2011).
* **`OPSS`** [Optimistic planning for sparsely stochastic systems](https://www.dcsc.tudelft.nl/~bdeschutter/pub/rep/11_007.pdf), L. Buşoniu, R. Munos, B. De Schutter, and R. Babuska (2011).
* **`HOOT`** [Sample-Based Planning for Continuous ActionMarkov Decision Processes](https://www.aaai.org/ocs/index.php/ICAPS/ICAPS11/paper/viewFile/2679/3175), Mansley C., Weinstein A., Littman M. (2011).
* **`HOLOP`** [Bandit-Based Planning and Learning inContinuous-Action Markov Decision Processes](https://pdfs.semanticscholar.org/a445/d8cc503781c481c3f3c4ee1758b862b3e869.pdf), Weinstein A., Littman M. (2012).
* **`BRUE`** [Simple Regret Optimization in Online Planning for Markov Decision Processes](https://www.jair.org/index.php/jair/article/view/10905/26003), Feldman Z. and Domshlak C. (2014).
* **`LGP`** [Logic-Geometric Programming: An Optimization-Based Approach to Combined Task and Motion Planning](https://ipvs.informatik.uni-stuttgart.de/mlr/papers/15-toussaint-IJCAI.pdf), Toussaint M. (2015). [🎞️](https://www.youtube.com/watch?v=B2s85xfo2uE)
* **`AlphaGo`** [Mastering the game of Go with deep neural networks and tree search](https://www.nature.com/articles/nature16961), Silver D. et al. (2016).
* **`AlphaGo Zero`** [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270), Silver D. et al. (2017).
* **`AlphaZero`** [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815), Silver D. et al. (2017).
* **`TrailBlazer`** [Blazing the trails before beating the path: Sample-efficient Monte-Carlo planning](https://papers.nips.cc/paper/6253-blazing-the-trails-before-beating-the-path-sample-efficient-monte-carlo-planning.pdf), Grill J. B., Valko M., Munos R. (2017).
* **`MCTSnets`** [Learning to search with MCTSnets](https://arxiv.org/abs/1802.04697), Guez A. et al. (2018).
* **`ADI`** [Solving the Rubik's Cube Without Human Knowledge](https://arxiv.org/abs/1805.07470), McAleer S. et al. (2018).
* **`OPC/SOPC`** [Continuous-action planning for discounted inﬁnite-horizon nonlinear optimal control with Lipschitz values](http://busoniu.net/files/papers/aut18.pdf), Buşoniu L., Pall E., Munos R. (2018).
* [Real-time tree search with pessimistic scenarios: Winning the NeurIPS 2018 Pommerman Competition](http://proceedings.mlr.press/v101/osogami19a.html), Osogami T., Takahashi T. (2019)

## Control Theory

* (book) [The Mathematical Theory of Optimal Processes](https://books.google.fr/books?id=kwzq0F4cBVAC&printsec=frontcover&redir_esc=y#v=onepage&q&f=false), L. S. Pontryagin, Boltyanskii V. G., Gamkrelidze R. V., and Mishchenko E. F. (1962).
* (book) [Constrained Control and Estimation](http://www.springer.com/gp/book/9781852335489),  Goodwin G. (2005).
* **`PI²`** [A Generalized Path Integral Control Approach to Reinforcement Learning](http://www.jmlr.org/papers/volume11/theodorou10a/theodorou10a.pdf), Theodorou E. et al. (2010).
* **`PI²-CMA`** [Path Integral Policy Improvement with Covariance Matrix Adaptation](https://arxiv.org/abs/1206.4621), Stulp F., Sigaud O. (2010).
* **`iLQG`** [A generalized iterative LQG method for locally-optimal feedback control of constrained nonlinear stochastic systems](http://maeresearch.ucsd.edu/skelton/publications/weiwei_ilqg_CDC43.pdf), Todorov E. (2005). [:octocat:](https://github.com/neka-nat/ilqr-gym)
* **`iLQG+`** [Synthesis and stabilization of complex behaviors through online trajectory optimization](https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf), Tassa Y. (2012).

## Model Predictive Control

* (book) [Model Predictive Control](http://een.iust.ac.ir/profs/Shamaghdari/MPC/Resources/), Camacho E. (1995).
* (book) [Predictive Control With Constraints](https://books.google.fr/books/about/Predictive_Control.html?id=HV_Y58c7KiwC&redir_esc=y), Maciejowski J. M. (2002).
* [Linear Model Predictive Control for Lane Keeping and Obstacle Avoidance on Low Curvature Roads](http://ieeexplore.ieee.org/document/6728261/), Turri V. et al. (2013).
* **`MPCC`** [Optimization-based autonomous racing of 1:43 scale RC cars](https://arxiv.org/abs/1711.07300), Liniger A. et al. (2014). [🎞️](https://www.youtube.com/watch?v=mXaElWYQKC4) | [🎞️](https://www.youtube.com/watch?v=JoHfJ6LEKVo)
* **`MIQP`** [Optimal trajectory planning for autonomous driving integrating logical constraints: An MIQP perspective](https://hal.archives-ouvertes.fr/hal-01342358v1/document), Qian X., Altché F., Bender P., Stiller C. de La Fortelle A. (2016).



# Safe Control :lock:

## Robust Control

* [Minimax analysis of stochastic problems](https://www2.isye.gatech.edu/~anton/MinimaxSP.pdf), Shapiro A., Kleywegt A. (2002).
* **`Robust DP`** [Robust Dynamic Programming](https://www.researchgate.net/publication/220442530/download), Iyengar G. (2005).
* [Robust Planning and Optimization](https://www.researchgate.net/profile/Francisco_Perez-Galarce/post/can_anyone_recommend_a_report_or_article_on_two_stage_robust_optimization/attachment/59d62578c49f478072e9a500/AS%3A272164542976002%401441900491330/download/2011+-+Robust+planning+and+optimization.pdf), Laumanns M. (2011). (lecture notes)
* [Robust Markov Decision Processes](https://pubsonline.informs.org/doi/pdf/10.1287/moor.1120.0566), Wiesemann W., Kuhn D., Rustem B. (2012).
* [Safe and Robust Learning Control with Gaussian Processes](http://www.dynsyslab.org/wp-content/papercite-data/pdf/berkenkamp-ecc15.pdf), Berkenkamp F., Schoellig A. (2015). [🎞️](https://www.youtube.com/watch?v=YqhLnCm0KXY)
* **`Tube-MPPI`** [Robust Sampling Based Model Predictive Control with Sparse Objective Information](http://www.roboticsproceedings.org/rss14/p42.pdf), Williams G. et al. (2018). [🎞️](https://www.youtube.com/watch?v=32v-e3dptjo)

## Risk-Averse Control

* [A Comprehensive Survey on Safe Reinforcement Learning](http://jmlr.org/papers/v16/garcia15a.html), García J., Fernández F. (2015).
* **`RA-QMDP`** [Risk-averse Behavior Planning for Autonomous Driving under Uncertainty](https://arxiv.org/abs/1812.01254), Naghshvar M. et al. (2018).
* **`StoROO`** [X-Armed Bandits: Optimizing Quantiles and Other Risks](https://arxiv.org/abs/1904.08205), Torossian L., Garivier A., Picheny V. (2019).
* [Worst Cases Policy Gradients](https://arxiv.org/abs/1911.03618), Tang Y. C. et al. (2019).

## Value-Constrained Control

* **`ICS`** [Will the Driver Seat Ever Be Empty?](https://hal.inria.fr/hal-00965176), Fraichard T. (2014).
* **`SafeOPT`** [Safe Controller Optimization for Quadrotors with Gaussian Processes](https://arxiv.org/abs/1509.01066), Berkenkamp F., Schoellig A., Krause A. (2015). [🎞️](https://www.youtube.com/watch?v=GiqNQdzc5TI) [:octocat:](https://github.com/befelix/SafeOpt)
* **`SafeMDP`** [Safe Exploration in Finite Markov Decision Processes with Gaussian Processes](https://arxiv.org/abs/1606.04753), Turchetta M., Berkenkamp F., Krause A. (2016). [:octocat:](https://github.com/befelix/SafeMDP)
* **`RSS`** [On a Formal Model of Safe and Scalable Self-driving Cars](https://arxiv.org/abs/1708.06374), Shalev-Shwartz S. et al. (2017).
* **`CPO`** [Constrained Policy Optimization](https://arxiv.org/abs/1705.10528), Achiam J., Held D., Tamar A., Abbeel P. (2017). [:octocat:](https://github.com/jachiam/cpo)
* **`RCPO`** [Reward Constrained Policy Optimization](https://arxiv.org/abs/1805.11074), Tessler C., Mankowitz D., Mannor S. (2018).
* **`BFTQ`** [A Fitted-Q Algorithm for Budgeted MDPs](https://hal.archives-ouvertes.fr/hal-01867353), Carrara N. et al. (2018).
* **`SafeMPC`** [Learning-based Model Predictive Control for Safe Exploration](https://arxiv.org/abs/1803.08287), Koller T, Berkenkamp F., Turchetta M. Krause A. (2018).
* **`CCE`** [Constrained Cross-Entropy Method for Safe Reinforcement Learning](https://papers.nips.cc/paper/7974-constrained-cross-entropy-method-for-safe-reinforcement-learning), Wen M., Topcu U. (2018). [:octocat:](https://github.com/liuzuxin/safe-mbrl)
* **`LTL-RL`** [Reinforcement Learning with Probabilistic Guarantees for Autonomous Driving](https://arxiv.org/abs/1904.07189), Bouton M. et al. (2019).
* [Safe Reinforcement Learning with Scene Decomposition for Navigating Complex Urban Environments](https://arxiv.org/abs/1904.11483v1), Bouton M. et al. (2019). [:octocat:](https://github.com/sisl/AutomotivePOMDPs.jl)
* [Batch Policy Learning under Constraints](https://arxiv.org/abs/1903.08738), Le H., Voloshin C., Yue Y. (2019).
* [Safely Learning to Control the Constrained Linear Quadratic Regulator](https://ieeexplore.ieee.org/abstract/document/8814865), Dean S. et al (2019).
* [Learning to Walk in the Real World with Minimal Human Effort](https://arxiv.org/abs/2002.08550), Ha S. et al. (2020) [🎞️](https://youtu.be/cwyiq6dCgOc)
* [Responsive Safety in Reinforcement Learning by PID Lagrangian Methods](https://arxiv.org/abs/2007.03964), Stooke A., Achiam J., Abbeel P. (2020). [:octocat:](https://github.com/astooke/rlpyt/tree/master/rlpyt/projects/safe)

## State-Constrained Control and Stability

* **`HJI-reachability`** [Safe learning for control: Combining disturbance estimation, reachability analysis and reinforcement learning with systematic exploration](http://kth.diva-portal.org/smash/get/diva2:1140173/FULLTEXT01.pdf), Heidenreich C. (2017).
* **`MPC-HJI`** [On Infusing Reachability-Based Safety Assurance within Probabilistic Planning Frameworks for Human-Robot Vehicle Interactions](https://stanfordasl.github.io/wp-content/papercite-data/pdf/Leung.Schmerling.Chen.ea.ISER18.pdf), Leung K. et al. (2018).
* [A General Safety Framework for Learning-Based Control in Uncertain Robotic Systems](https://arxiv.org/abs/1705.01292), Fisac J. et al (2017). [🎞️](https://www.youtube.com/watch?v=WAAxyeSk2bw&feature=youtu.be)
* [Safe Model-based Reinforcement Learning with Stability Guarantees](https://arxiv.org/abs/1705.08551), Berkenkamp F. et al. (2017).
* **`Lyapunov-Net`** [Safe Interactive Model-Based Learning](https://arxiv.org/abs/1911.06556), Gallieri M. et al. (2019).
* [Enforcing robust control guarantees within neural network policies](https://arxiv.org/abs/2011.08105), Donti P. et al. (2021). [:octocat:](https://github.com/locuslab/robust-nn-control)

## Uncertain Dynamical Systems

* [Simulation of Controlled Uncertain Nonlinear Systems](https://www.sciencedirect.com/science/article/pii/009630039400112H), Tibken B., Hofer E. (1995).
* [Trajectory computation of dynamic uncertain systems](https://ieeexplore.ieee.org/iel5/8969/28479/01272787.pdf), Adrot O., Flaus J-M. (2002).
* [Simulation of Uncertain Dynamic Systems Described By Interval Models: a Survey](https://www.sciencedirect.com/science/article/pii/S1474667016362206), Puig V. et al. (2005).
* [Design of interval observers for uncertain dynamical systems](https://hal.inria.fr/hal-01276439/file/Interval_Survey.pdf), Efimov D., Raïssi T. (2016).




# Game Theory :spades:

* [Hierarchical Game-Theoretic Planning for Autonomous Vehicles](https://arxiv.org/abs/1810.05766), Fisac J. et al. (2018).
* [Efficient Iterative Linear-Quadratic Approximations for Nonlinear Multi-Player General-Sum Differential Games](https://arxiv.org/abs/1909.04694), Fridovich-Keil D. et al. (2019). [🎞️](https://www.youtube.com/watch?v=KPEPk-QrkQ8&feature=youtu.be)




# Sequential Learning :shoe:

## Multi-Armed Bandit :slot_machine:

* **`TS`** [On the Likelihood that One Unknown Probability Exceeds Another in View of the Evidence of Two Samples](https://www.jstor.org/stable/pdf/2332286.pdf), Thompson W. (1933).
* [Exploration and Exploitation in Organizational Learning](https://www3.nd.edu/~ggoertz/abmir/march1991.pdf), March J. (1991).
* **`UCB1 / UCB2`** [Finite-time Analysis of the Multiarmed Bandit Problem](https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf), Auer P., Cesa-Bianchi N., Fischer P. (2002).
* **`Empirical Bernstein / UCB-V`** [Exploration-exploitation tradeoff using variance estimates in multi-armed bandits](https://hal.inria.fr/hal-00711069/),  Audibert J-Y, Munos R., Szepesvari C. (2009).
* [Empirical Bernstein Bounds and Sample Variance Penalization](https://arxiv.org/abs/0907.3740), Maurer A., Ponti M. (2009).
* [An Empirical Evaluation of Thompson Sampling](https://papers.nips.cc/paper/4321-an-empirical-evaluation-of-thompson-sampling), Chapelle O., Li L. (2011).
* **`kl-UCB`** [The KL-UCB Algorithm for Bounded Stochastic Bandits and Beyond](https://arxiv.org/abs/1102.2490), Garivier A., Cappé O. (2011).
* **`KL-UCB`** [Kullback-Leibler Upper Confidence Bounds for Optimal Sequential Allocation](https://projecteuclid.org/euclid.aos/1375362558), Cappé O. et al. (2013).
* **`IDS`** [Information Directed Sampling and Bandits with Heteroscedastic Noise](https://arxiv.org/abs/1801.09667) Kirschner J., Krause A. (2018).

#### Contextual

* **`LinUCB`** [A Contextual-Bandit Approach to Personalized News Article Recommendation](https://arxiv.org/abs/1003.0146), Li L. et al. (2010).
* **`OFUL`** [Improved Algorithms for Linear Stochastic Bandits](https://papers.nips.cc/paper/4417-improved-algorithms-for-linear-stochastic-bandits), Abbasi-yadkori Y., Pal D., Szepesvári C. (2011).
* [Contextual Bandits with Linear Payoff Functions](http://proceedings.mlr.press/v15/chu11a.html), Chu W. et al. (2011).
* [Self-normalization techniques for streaming confident regression](https://hal.archives-ouvertes.fr/hal-01349727v2), Maillard O.-A. (2017).
* [Learning from Delayed Outcomes via Proxies with Applications to Recommender Systems](https://arxiv.org/abs/1807.09387) Mann T. et al. (2018). (prediction setting)
* [Weighted Linear Bandits for Non-Stationary Environments](https://arxiv.org/abs/1909.09146), Russac Y. et al. (2019).
* [Linear bandits with Stochastic Delayed Feedback](http://proceedings.mlr.press/v119/vernade20a.html), Vernade C. et al. (2020).


### Best Arm Identification :muscle:

* **`Successive Elimination`** [Action Elimination and Stopping Conditions for the Multi-Armed Bandit and Reinforcement Learning Problems](http://jmlr.csail.mit.edu/papers/volume7/evendar06a/evendar06a.pdf), Even-Dar E. et al. (2006).
* **`LUCB`** [PAC Subset Selection in Stochastic Multi-armed Bandits](https://www.cse.iitb.ac.in/~shivaram/papers/ktas_icml_2012.pdf), Kalyanakrishnan S. et al. (2012).
* **`UGapE`** [Best Arm Identification: A Unified Approach to Fixed Budget and Fixed Confidence](https://hal.archives-ouvertes.fr/hal-00747005), Gabillon V., Ghavamzadeh M., Lazaric A. (2012).
* **`Sequential Halving`** [Almost Optimal Exploration in Multi-Armed Bandits](http://proceedings.mlr.press/v28/karnin13.pdf), Karnin Z. et al (2013).
* **`M-LUCB / M-Racing`** [Maximin Action Identification: A New Bandit Framework for Games](https://arxiv.org/abs/1602.04676), Garivier A., Kaufmann E., Koolen W. (2016).
* **`Track-and-Stop`** [Optimal Best Arm Identification with Fixed Confidence](https://arxiv.org/abs/1602.04589), Garivier A., Kaufmann E. (2016).
* **`LUCB-micro`** [Structured Best Arm Identification with Fixed Confidence](https://arxiv.org/abs/1706.05198), Huang R. et al. (2017).

### Black-box Optimization :black_large_square:

* **`GP-UCB`** [Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design](https://arxiv.org/abs/0912.3995), Srinivas N., Krause A., Kakade S., Seeger M. (2009).
* **`HOO`** [X–Armed Bandits](https://arxiv.org/abs/1001.4475), Bubeck S., Munos R., Stoltz G., Szepesvari C. (2009).
* **`DOO/SOO`** [Optimistic Optimization of a Deterministic Function without the Knowledge of its Smoothness](https://papers.nips.cc/paper/4304-optimistic-optimization-of-a-deterministic-function-without-the-knowledge-of-its-smoothness), Munos R. (2011).
* **`StoOO`** [From Bandits to Monte-Carlo Tree Search: The Optimistic Principle Applied to Optimization and Planning](https://hal.archives-ouvertes.fr/hal-00747575v4/), Munos R. (2014).
* **`StoSOO`** [Stochastic Simultaneous Optimistic Optimization](http://proceedings.mlr.press/v28/valko13.pdf), Valko M., Carpentier A., Munos R. (2013). 
* **`POO`** [Black-box optimization of noisy functions with unknown smoothness](https://hal.inria.fr/hal-01222915v4/), Grill J-B., Valko M., Munos R. (2015).
* **`EI-GP`** [Bayesian Optimization in AlphaGo](https://arxiv.org/abs/1812.06855), Chen Y. et al. (2018)



# Reinforcement Learning :robot:

* [Reinforcement learning: A survey](https://www.jair.org/media/301/live-301-1562-jair.pdf), Kaelbling L. et al. (1996).

## Theory :books:

* [Expected mistake bound model for on-line reinforcement learning](https://pdfs.semanticscholar.org/13b8/1dd08aab636c3761c5eb4337dbe43aedaf31.pdf), Fiechter C-N. (1997).
* **`UCRL2`** [Near-optimal Regret Bounds for Reinforcement Learning](http://www.jmlr.org/papers/volume11/jaksch10a/jaksch10a.pdf), Jaksch T. (2010). ![Setting](https://img.shields.io/badge/setting-average-green)![Setting](https://img.shields.io/badge/communicating-orange)![Bound](https://img.shields.io/badge/DS(AT)^0.5-orange)
* **`PSRL`** [Why is Posterior Sampling Better than Optimism for Reinforcement Learning?](https://arxiv.org/abs/1607.00215), Osband I., Van Roy B. (2016). ![Setting](https://img.shields.io/badge/setting-episodic-green) ![Bayesian](https://img.shields.io/badge/bayesian-green)
* **`UCBVI`** [Minimax Regret Bounds for Reinforcement Learning](http://proceedings.mlr.press/v70/azar17a.html), Azar M., Osband I., Munos R. (2017). ![Setting](https://img.shields.io/badge/setting-episodic-green)![Bound](https://img.shields.io/badge/H(SAT)^0.5-orange)
* **`Q-Learning-UCB`** [Is Q-Learning Provably Efficient?](https://papers.nips.cc/paper/7735-is-q-learning-provably-efficient), Jin C., Allen-Zhu Z., Bubeck S., Jordan M. (2018). ![Setting](https://img.shields.io/badge/setting-episodic-green)
* **`LSVI-UCB`** [Provably Efficient Reinforcement Learning with Linear Function Approximation](https://arxiv.org/abs/1907.05388), Jin C., Yang Z., Wang Z., Jordan M. (2019). ![Setting](https://img.shields.io/badge/setting-episodic-green) ![Spaces](https://img.shields.io/badge/approximation-linear-green)
* [Lipschitz Continuity in Model-based Reinforcement Learning](https://arxiv.org/abs/1804.07193), Asadi K. et al (2018).
* [On Function Approximation in Reinforcement Learning: Optimism in the Face of Large State Spaces](https://arxiv.org/abs/2011.04622), Yang Z., Jin C., Wang Z., Wang M., Jordan M. (2021) ![Setting](https://img.shields.io/badge/setting-episodic-green) ![Spaces](https://img.shields.io/badge/approximation-kernel/nn-green) ![Bound](https://img.shields.io/badge/delta.H^2(T)^0.5-orange)

### Generative Model

*  **`QVI`** [On the Sample Complexity of Reinforcement Learning with a Generative Model](https://arxiv.org/abs/1206.6461), Azar M., Munos R., Kappen B. (2012).
* [Model-Based Reinforcement Learning with a Generative Model is Minimax Optimal](https://arxiv.org/abs/1906.03804), Agarwal A. et al. (2019).

### Policy Gradient

* [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation), Sutton R. et al (2000).
* [Approximately Optimal Approximate Reinforcement Learning](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf), Kakade S., Langford J. (2002).
* [On the Theory of Policy Gradient Methods: Optimality, Approximation, and Distribution Shift](https://arxiv.org/abs/1908.00261), Agarwal A. et al. (2019)
* [PC-PG: Policy Cover Directed Exploration for Provable Policy Gradient Learning](https://arxiv.org/abs/2007.08459),  Agarwal A. et al. (2020) 

### Linear Systems

* [PAC Adaptive Control of Linear Systems](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.49.339&rep=rep1&type=pdf), Fiechter C.-N. (1997)
* **`OFU-LQ`** [Regret Bounds for the Adaptive Control of Linear Quadratic Systems](http://proceedings.mlr.press/v19/abbasi-yadkori11a/abbasi-yadkori11a.pdf), Abbasi-Yadkori Y., Szepesvari C. (2011).
* **`TS-LQ`** [Improved Regret Bounds for Thompson Sampling in Linear Quadratic Control Problems](http://proceedings.mlr.press/v80/abeille18a.html), Abeille M., Lazaric A. (2018).
* [Exploration-Exploitation with Thompson Sampling in Linear Systems](https://tel.archives-ouvertes.fr/tel-01816069/), Abeille M. (2017). (phd thesis)
* **`Coarse-Id`** [On the Sample Complexity of the Linear Quadratic Regulator](https://arxiv.org/abs/1710.01688), Dean S., Mania H., Matni N., Recht B., Tu S. (2017).
* [Regret Bounds for Robust Adaptive Control of the Linear Quadratic Regulator](http://papers.nips.cc/paper/7673-regret-bounds-for-robust-adaptive-control-of-the-linear-quadratic-regulator), Dean S. et al (2018).
* [Robust exploration in linear quadratic reinforcement learning](https://papers.nips.cc/paper/9668-robust-exploration-in-linear-quadratic-reinforcement-learning), Umenberger J. et al (2019).
* [Online Control with Adversarial Disturbances](https://arxiv.org/abs/1902.08721), Agarwal N. et al (2019).  ![Noise](https://img.shields.io/badge/noise-adversarial-red)![Costs](https://img.shields.io/badge/costs-convex-green)
* [Logarithmic Regret for Online Control](https://arxiv.org/abs/1909.05062), Agarwal N. et al (2019).  ![Noise](https://img.shields.io/badge/noise-adversarial-red)![Costs](https://img.shields.io/badge/costs-convex-green)

## Value-based :chart_with_upwards_trend:

* **`NFQ`** [Neural fitted Q iteration - First experiences with a data efficient neural Reinforcement Learning method](http://ml.informatik.uni-freiburg.de/former/_media/publications/rieecml05.pdf), Riedmiller M. (2005).
* **`DQN`** [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), Mnih V. et al. (2013). [🎞️](https://www.youtube.com/watch?v=iqXKQf2BOSE)
* **`DDQN`** [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461), van Hasselt H., Silver D. et al. (2015).
* **`DDDQN`** [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581), Wang Z. et al. (2015). [🎞️](https://www.youtube.com/watch?v=qJd3yaEN9Sw)
* **`PDDDQN`** [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952), Schaul T. et al. (2015).
* **`NAF`** [Continuous Deep Q-Learning with Model-based Acceleration](https://arxiv.org/abs/1603.00748), Gu S. et al. (2016).
* **`Rainbow`** [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298), Hessel M. et al. (2017).
* **`Ape-X DQfD`** [Observe and Look Further: Achieving Consistent Performance on Atari](https://arxiv.org/abs/1805.11593), Pohlen T. et al. (2018). [🎞️](https://www.youtube.com/watch?v=-0xOdnoxAFo&index=4&list=PLnZpNNVLsMmOfqMwJLcpLpXKLr3yKZ8Ak)

## Policy-based :muscle:

### Policy gradient

* **`REINFORCE`** [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf), Williams R. (1992).
* **`Natural Gradient`** [A Natural Policy Gradient](https://papers.nips.cc/paper/2073-a-natural-policy-gradient.pdf), Kakade S. (2002).
* [Policy Gradient Methods for Robotics](http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/IROS2006-Peters_%5b0%5d.pdf), Peters J.,  Schaal S. (2006).
* **`TRPO`** [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477), Schulman J. et al. (2015). [🎞️](https://www.youtube.com/watch?v=KJ15iGGJFvQ)
* **`PPO`** [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), Schulman J. et al. (2017). [🎞️](https://www.youtube.com/watch?v=bqdjsmSoSgI)
* **`DPPO`** [Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/abs/1707.02286), Heess N. et al. (2017). [🎞️](https://www.youtube.com/watch?v=hx_bgoTF7bs)

### Actor-critic

* **`AC`** [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf), Sutton R. et al. (1999).
* **`NAC`** [Natural Actor-Critic](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/NaturalActorCritic.pdf), Peters J. et al. (2005).
* **`DPG`** [Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf), Silver D. et al. (2014).
* **`DDPG`** [Continuous Control With Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971), Lillicrap T. et al. (2015). [🎞️ 1](https://www.youtube.com/watch?v=lV5JhxsrSH8) | [2](https://www.youtube.com/watch?v=8CNck-hdys8) | [3](https://www.youtube.com/watch?v=xw73qehvSRQ) | [4](https://www.youtube.com/watch?v=vWxBmHRnQMI)
* **`MACE`** [Terrain-Adaptive Locomotion Skills Using Deep Reinforcement Learning](https://www.cs.ubc.ca/~van/papers/2016-TOG-deepRL/index.html), Peng X., Berseth G., van de Panne M. (2016). [🎞️](https://www.youtube.com/watch?v=KPfzRSBzNX4) | [🎞️](https://www.youtube.com/watch?v=A0BmHoujP9k)
* **`A3C`** [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783), Mnih V. et al 2016. [🎞️ 1](https://www.youtube.com/watch?v=Ajjc08-iPx8) | [2](https://www.youtube.com/watch?v=0xo1Ldx3L5Q) | [3](https://www.youtube.com/watch?v=nMR5mjCFZCw)
* **`SAC`** [Soft Actor-Critic : Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290), Haarnoja T. et al. (2018). [🎞️](https://vimeo.com/252185258)
* **`MPO`** [Maximum a Posteriori Policy Optimisation](https://arxiv.org/abs/1806.06920), Abdolmaleki A. et al (2021).

### Derivative-free

* **`CEM`** [Learning Tetris Using the Noisy Cross-Entropy Method](http://iew3.technion.ac.il/CE/files/papers/Learning%20Tetris%20Using%20the%20Noisy%20Cross-Entropy%20Method.pdf), Szita I., Lörincz A. (2006). [🎞️](https://www.youtube.com/watch?v=UZnDYGk1j2c)
* **`CMAES`** [Completely Derandomized Self-Adaptation in Evolution Strategies](https://dl.acm.org/citation.cfm?id=1108843), Hansen N., Ostermeier A. (2001).
* **`NEAT`** [Evolving Neural Networks through Augmenting Topologies](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf), Stanley K. (2002). [🎞️](https://www.youtube.com/watch?v=5lJuEW-5vr8)
* **`iCEM`** [Sample-efficient Cross-Entropy Method for Real-time Planning](https://arxiv.org/abs/2008.06389), Pinneri C. et al. (2020).

## Model-based :world_map:

* **`Dyna`** [Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.84.6983&rep=rep1&type=pdf), Sutton R. (1990).
* **`PILCO`** [PILCO: A Model-Based and Data-Efficient Approach to Policy Search](http://mlg.eng.cam.ac.uk/pub/pdf/DeiRas11.pdf), Deisenroth M., Rasmussen C. (2011). ([talk](https://www.youtube.com/watch?v=f7y60SEZfXc))
* **`DBN`** [Probabilistic MDP-behavior planning for cars](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6082928), Brechtel S. et al. (2011).
* **`GPS`** [End-to-End Training of Deep Visuomotor Policies](https://arxiv.org/abs/1504.00702), Levine S. et al. (2015). [🎞️](https://www.youtube.com/watch?v=Q4bMcUk6pcw)
* **`DeepMPC`** [DeepMPC: Learning Deep Latent Features for Model Predictive Control](https://www.cs.stanford.edu/people/asaxena/papers/deepmpc_rss2015.pdf), Lenz I. et al. (2015). [🎞️](https://www.youtube.com/watch?v=BwA90MmkvPU)
* **`SVG`** [Learning Continuous Control Policies by Stochastic Value Gradients](https://arxiv.org/abs/1510.09142), Heess N. et al. (2015). [🎞️](https://www.youtube.com/watch?v=PYdL7bcn_cM)
* **`FARNN`** [Nonlinear Systems Identification Using Deep Dynamic Neural Networks](https://arxiv.org/abs/1610.01439), Ogunmolu O. et al. (2016). [:octocat:](https://github.com/lakehanne/FARNN)
* [Optimal control with learned local models: Application to dexterous manipulation](https://homes.cs.washington.edu/~todorov/papers/KumarICRA16.pdf), Kumar V. et al. (2016). [🎞️](https://www.youtube.com/watch?v=bD5z1I1TU3w)
* **`BPTT`** [Long-term Planning by Short-term Prediction](https://arxiv.org/abs/1602.01580), Shalev-Shwartz S. et al. (2016). [🎞️ 1](https://www.youtube.com/watch?v=Nqmv1anUaF4) | [2](https://www.youtube.com/watch?v=UgGZ9lMvey8)
* [Deep visual foresight for planning robot motion](https://arxiv.org/abs/1610.00696), Finn C., Levine S. (2016). [🎞️](https://www.youtube.com/watch?v=6k7GHG4IUCY)
* **`VIN`** [Value Iteration Networks](https://arxiv.org/abs/1602.02867), Tamar A. et al  (2016). [🎞️](https://www.youtube.com/watch?v=RcRkog93ZRU)
* **`VPN`** [Value Prediction Network](https://arxiv.org/abs/1707.03497), Oh J. et al. (2017).
* **`DistGBP`** [Model-Based Planning with Discrete and Continuous Actions](https://arxiv.org/abs/1705.07177), Henaff M. et al. (2017). [🎞️ 1](https://www.youtube.com/watch?v=9Xh2TRQ_4nM) | [2](https://www.youtube.com/watch?v=XLdme0TTjiw)
* [Prediction and Control with Temporal Segment Models](https://arxiv.org/abs/1703.04070), Mishra N. et al. (2017).
* **`Predictron`** [The Predictron: End-To-End Learning and Planning](https://arxiv.org/abs/1612.08810), Silver D. et al. (2017). [🎞️](https://www.youtube.com/watch?v=BeaLdaN2C3Q)
* **`MPPI`** [Information Theoretic MPC for Model-Based Reinforcement Learning](https://ieeexplore.ieee.org/document/7989202/), Williams G. et al. (2017). [:octocat:](https://github.com/ferreirafabio/mppi_pendulum) [🎞️](https://www.youtube.com/watch?v=f2at-cqaJMM)
* [Learning Real-World Robot Policies by Dreaming](https://arxiv.org/abs/1805.07813), Piergiovanni A. et al. (2018).
* [Coupled Longitudinal and Lateral Control of a Vehicle using Deep Learning](https://arxiv.org/abs/1810.09365), Devineau G., Polack P., Alchté F., Moutarde F. (2018) [🎞️](https://www.youtube.com/watch?v=yyWy1uavlXs)
* **`PlaNet`** [Learning Latent Dynamics for Planning from Pixels](https://planetrl.github.io/), Hafner et al. (2018).  [🎞️](https://www.youtube.com/watch?v=tZk1eof_VNA)
* **`NeuralLander`** [Neural Lander: Stable Drone Landing Control using Learned Dynamics](https://arxiv.org/abs/1811.08027), Shi G. et al. (2018). [🎞️](https://www.youtube.com/watch?v=FLLsG0S78ik)
* **`DBN+POMCP`** [Towards Human-Like Prediction and Decision-Making for Automated Vehicles in Highway Scenarios ](https://tel.archives-ouvertes.fr/tel-02184362), Sierra Gonzalez D. (2019).
* [Planning with Goal-Conditioned Policies](https://sites.google.com/view/goal-planning), Nasiriany S. et al. (2019). [🎞️](https://sites.google.com/view/goal-planning#h.p_0m-H0QfKVj4n)
* **`MuZero`** [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265), Schrittwiese J. et al. (2019). [:octocat:](https://github.com/werner-duvaud/muzero-general)
* **`BADGR`** [BADGR: An Autonomous Self-Supervised Learning-Based Navigation System](https://sites.google.com/view/badgr), Kahn G., Abbeel P., Levine S. (2020). [🎞️](https://www.youtube.com/watch?v=EMV0zEXbcc4) [:octocat:](https://github.com/gkahn13/badgr)
* **`H-UCRL`** [Efficient Model-Based Reinforcement Learning through Optimistic Policy Search and Planning](https://proceedings.neurips.cc//paper_files/paper/2020/hash/a36b598abb934e4528412e5a2127b931-Abstract.html), Curi S., Berkenkamp F., Krause A. (2020). [:octocat:](https://github.com/sebascuri/hucrl)


## Exploration :tent:

* [Combating Reinforcement Learning's Sisyphean Curse with Intrinsic Fear](https://arxiv.org/abs/1611.01211), Lipton Z. et al. (2016).
* **`Pseudo-count`** [Unifying Count-Based Exploration and Intrinsic Motivation](https://arxiv.org/abs/1606.01868), Bellemare M. et al (2016). [🎞️](https://www.youtube.com/watch?v=0yI2wJ6F8r0) 
* **`HER`** [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495), Andrychowicz M. et al. (2017). [🎞️](https://www.youtube.com/watch?v=Dz_HuzgMxzo)
* **`VHER`** [Visual Hindsight Experience Replay](https://arxiv.org/abs/1901.11529), Sahni H. et al. (2019).
* **`RND`** [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894), Burda Y. et al. (OpenAI) (2018).  [🎞️](https://openai.com/blog/reinforcement-learning-with-prediction-based-rewards/)
* **`Go-Explore`** [Go-Explore: a New Approach for Hard-Exploration Problems](https://arxiv.org/abs/1901.10995), Ecoffet A. et al. (Uber) (2018). [🎞️](https://www.youtube.com/watch?v=gnGyUPd_4Eo)
* **`C51-IDS`** [Information-Directed Exploration for Deep Reinforcement Learning](https://arxiv.org/abs/1812.07544), Nikolov N., Kirschner J., Berkenkamp F., Krause A. (2019). [:octocat:](https://github.com/nikonikolov/rltf)
* **`Plan2Explore`** [Planning to Explore via Self-Supervised World Models](https://ramanans1.github.io/plan2explore/), Sekar R. et al. (2020). [🎞️](https://www.youtube.com/watch?v=GftqnPWsCWw&feature=emb_title) [:octocat:](https://github.com/ramanans1/plan2explore)
* **`RIDE`** [RIDE: Rewarding Impact-Driven Exploration for Procedurally-Generated Environments](https://openreview.net/pdf?id=rkg-TJBFPB), Raileanu R., Rocktäschel T., (2020). [:octocat:](https://github.com/facebookresearch/impact-driven-exploration)

## Hierarchy and Temporal Abstraction :clock2:

* [Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning](http://www-anw.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf), Sutton R. et al. (1999).
* [Intrinsically motivated learning of hierarchical collections of skills](http://www-anw.cs.umass.edu/pubs/2004/barto_sc_ICDL04.pdf), Barto A. et al. (2004).
* **`OC`** [The Option-Critic Architecture](https://arxiv.org/abs/1609.05140), Bacon P-L., Harb J., Precup D. (2016).
* [Learning and Transfer of Modulated Locomotor Controllers](https://arxiv.org/abs/1610.05182), Heess N. et al. (2016). [🎞️](https://www.youtube.com/watch?v=sboPYvhpraQ&feature=youtu.be)
* [Safe, Multi-Agent, Reinforcement Learning for Autonomous Driving](https://arxiv.org/abs/1610.03295), Shalev-Shwartz S. et al. (2016).
* **`FuNs`** [FeUdal Networks for Hierarchical Reinforcement Learning](https://arxiv.org/abs/1703.01161), Vezhnevets A. et al. (2017).
* [Combining Neural Networks and Tree Search for Task and Motion Planning in Challenging Environments](https://arxiv.org/abs/1703.07887), Paxton C. et al. (2017). [🎞️](https://www.youtube.com/watch?v=MM2U_SGMtk8)
* **`DeepLoco`** [DeepLoco: Dynamic Locomotion Skills Using Hierarchical Deep Reinforcement Learning ](https://www.cs.ubc.ca/~van/papers/2017-TOG-deepLoco/), Peng X. et al. (2017). [🎞️](https://www.youtube.com/watch?v=hd1yvLWm6oA) | [🎞️](https://www.youtube.com/watch?v=x-HrYko_MRU)
* [Hierarchical Policy Design for Sample-Efficient Learning of Robot Table Tennis Through Self-Play](https://arxiv.org/abs/1811.12927), Mahjourian R. et al (2018). [🎞️](https://sites.google.com/view/robottabletennis)
* **`DAC`** [DAC: The Double Actor-Critic Architecture for Learning Options](https://arxiv.org/abs/1904.12691), Zhang S., Whiteson S. (2019). 
* [Multi-Agent Manipulation via Locomotion using Hierarchical Sim2Real](https://arxiv.org/abs/1908.05224), Nachum O. et al (2019). [🎞️](https://sites.google.com/view/manipulation-via-locomotion)
* [SoftCon: Simulation and Control of Soft-Bodied Animals with Biomimetic Actuators](http://mrl.snu.ac.kr/publications/ProjectSoftCon/SoftCon.html), Min S. et al. (2020). [🎞️](https://www.youtube.com/watch?v=I2ylkhPSkT4) [:octocat:](https://github.com/seiing/SoftCon)
* **`H-REIL`** [Reinforcement Learning based Control of Imitative Policies for Near-Accident Driving](http://iliad.stanford.edu/pdfs/publications/cao2020reinforcement.pdf), Cao Z. et al. (2020). [🎞️ 1](https://www.youtube.com/watch?v=CY24zlC_HdI&feature=youtu.be), [2](https://www.youtube.com/watch?v=envT7b5YRts&feature=youtu.be)

## Partial Observability :eye:

* **`PBVI`** [Point-based Value Iteration: An anytime algorithm for POMDPs](https://www.ri.cmu.edu/pub_files/pub4/pineau_joelle_2003_3/pineau_joelle_2003_3.pdf), Pineau J. et al. (2003).
* **`cPBVI`** [Point-Based Value Iteration for Continuous POMDPs](http://www.jmlr.org/papers/volume7/porta06a/porta06a.pdf), Porta J. et al. (2006).
* **`POMCP`** [Monte-Carlo Planning in Large POMDPs](https://papers.nips.cc/paper/4031-monte-carlo-planning-in-large-pomdps), Silver D., Veness J. (2010).
* [A POMDP Approach to Robot Motion Planning under Uncertainty](http://users.isr.ist.utl.pt/~mtjspaan/POMDPPractioners/pomdp2010_submission_5.pdf), Du Y. et al. (2010).
* [Probabilistic Online POMDP Decision Making for Lane Changes in Fully Automated Driving](https://users.cs.duke.edu/~pdinesh/sources/06728533.pdf), Ulbrich S., Maurer M. (2013).
* [Solving Continuous POMDPs: Value Iteration with Incremental Learning of an Efficient Space Representation](http://proceedings.mlr.press/v28/brechtel13.pdf), Brechtel S. et al. (2013).
* [Probabilistic Decision-Making under Uncertainty for Autonomous Driving using Continuous POMDPs](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6957722), Brechtel S. et al. (2014).
* **`MOMDP`** [Intention-Aware Motion Planning](http://ares.lids.mit.edu/fm/documents/intentionawaremotionplanning.pdf), Bandyopadhyay T. et al. (2013).
* **`DNC`** [Hybrid computing using a neural network with dynamic external memory](https://www.nature.com/articles/nature20101), Graves A. et al (2016). [🎞️](https://www.youtube.com/watch?v=B9U8sI7TcMY)
* [The value of inferring the internal state of traffic participants for autonomous freeway driving](https://arxiv.org/abs/1702.00858), Sunberg Z. et al. (2017).
* [Belief State Planning for Autonomously Navigating Urban Intersections](https://arxiv.org/abs/1704.04322), Bouton M., Cosgun A., Kochenderfer M. (2017).
* [Scalable Decision Making with Sensor Occlusions for Autonomous Driving](https://ieeexplore.ieee.org/document/8460914), Bouton M. et al. (2018).
* [Probabilistic Decision-Making at Road Intersections: Formulation and Quantitative Evaluation](https://hal.inria.fr/hal-01940392), Barbier M., Laugier C., Simonin O., Ibanez J. (2018).
* [Beauty and the Beast: Optimal Methods Meet Learning for Drone Racing](https://arxiv.org/abs/1810.06224), Kaufmann E. et al. (2018). [🎞️](https://www.youtube.com/watch?v=UuQvijZcUSc)
*  **`social perception`** [Behavior Planning of Autonomous Cars with Social Perception](https://arxiv.org/abs/1905.00988), Sun L. et al (2019).

## Transfer :earth_americas:

* **`IT&E`** [Robots that can adapt like animals](https://arxiv.org/abs/1407.3501), Cully A., Clune J., Tarapore D., Mouret J-B. (2014). [🎞️](https://www.youtube.com/watch?v=T-c17RKh3uE)
* **`MAML`** [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400), Finn C., Abbeel P., Levine S. (2017). [🎞️](https://sites.google.com/view/maml)
* [Virtual to Real Reinforcement Learning for Autonomous Driving](https://arxiv.org/abs/1704.03952), Pan X. et al. (2017). [🎞️](https://www.youtube.com/watch?v=Bce2ZSlMuqY)
* [Sim-to-Real: Learning Agile Locomotion For Quadruped Robots](https://arxiv.org/abs/1804.10332), Tan J. et al. (2018). [🎞️](https://www.youtube.com/watch?v=lUZUr7jxoqM)
* **`ME-TRPO`** [Model-Ensemble Trust-Region Policy Optimization](https://arxiv.org/abs/1802.10592), Kurutach T. et al. (2018). [🎞️](https://www.youtube.com/watch?v=tpS8qj7yhoU)
* [Kickstarting Deep Reinforcement Learning](https://arxiv.org/abs/1803.03835), Schmitt S. et al. (2018).
* [Learning Dexterous In-Hand Manipulation](https://blog.openai.com/learning-dexterity/), OpenAI (2018). [🎞️](https://www.youtube.com/watch?v=DKe8FumoD4E)
* **`GrBAL / ReBAL`** [Learning to Adapt in Dynamic, Real-World Environments Through Meta-Reinforcement Learning](https://arxiv.org/abs/1803.11347), Nagabandi A. et al. (2018). [🎞️](https://sites.google.com/berkeley.edu/metaadaptivecontrol)
* [Learning agile and dynamic motor skills for legged robots](https://robotics.sciencemag.org/content/4/26/eaau5872), Hwangbo J. et al. (ETH Zurich / Intel ISL) (2019). [🎞️](https://www.youtube.com/watch?v=ITfBKjBH46E)
* [Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning](https://arxiv.org/abs/1901.07517), Lee J., Hwangbo J., Hutter M. (ETH Zurich RSL) (2019)
* **`IT&E`** [Learning and adapting quadruped gaits with the "Intelligent Trial & Error" algorithm](https://hal.inria.fr/hal-02084619), Dalin E., Desreumaux P., Mouret J-B. (2019). [🎞️](https://www.youtube.com/watch?v=v90CWJ_HsnM)
* **`FAMLE`** [Fast Online Adaptation in Robotics through Meta-Learning Embeddings of Simulated Priors](https://arxiv.org/abs/2003.04663), Kaushik R., Anne T., Mouret J-B. (2020). [🎞️](https://www.youtube.com/watch?v=QIY1Sm7wHhE)
* [Robust Deep Reinforcement Learning against Adversarial Perturbations on Observations](https://arxiv.org/abs/2003.08938), Zhang H. et al (2020). [:octocat:](https://github.com/chenhongge/StateAdvDRL)
* [Learning quadrupedal locomotion over challenging terrain](https://robotics.sciencemag.org/content/5/47/eabc5986), Lee J. et al. (2020). [🎞️](https://www.youtube.com/watch?v=9j2a1oAHDL8)

## Multi-agent :two_men_holding_hands:

* **`Minimax-Q`** [Markov games as a framework for multi-agent reinforcement learning](https://www.cs.rutgers.edu/~mlittman/papers/ml94-final.pdf), M. Littman (1994).
* [Autonomous Agents Modelling Other Agents: A Comprehensive Survey and Open Problems](https://arxiv.org/abs/1709.08071), Albrecht S., Stone P. (2017).
* **`MILP`** [Time-optimal coordination of mobile robots along specified paths](https://arxiv.org/abs/1603.04610), Altché F. et al. (2016). [🎞️](https://www.youtube.com/watch?v=RiW2OFsdHOY)
* **`MIQP`** [An Algorithm for Supervised Driving of Cooperative Semi-Autonomous Vehicles](https://arxiv.org/abs/1706.08046), Altché F. et al. (2017). [🎞️](https://www.youtube.com/watch?v=JJZKfHMUeCI)
* **`SA-CADRL`** [Socially Aware Motion Planning with Deep Reinforcement Learning](https://arxiv.org/abs/1703.08862), Chen Y. et al. (2017). [🎞️](https://www.youtube.com/watch?v=CK1szio7PyA)
* [Multipolicy decision-making for autonomous driving via changepoint-based behavior prediction: Theory and experiment](https://link.springer.com/article/10.1007/s10514-017-9619-z), Galceran E. et al. (2017).
* [Online decision-making for scalable autonomous systems](https://www.ijcai.org/proceedings/2017/664), Wray K. et al. (2017).
* **`MAgent`** [MAgent: A Many-Agent Reinforcement Learning Platform for Artificial Collective Intelligence](https://arxiv.org/abs/1712.00600), Zheng L. et al. (2017). [🎞️](https://www.youtube.com/watch?v=HCSm0kVolqI)
* [Cooperative Motion Planning for Non-Holonomic Agents with Value Iteration Networks](https://arxiv.org/abs/1709.05273), Rehder E. et al. (2017).
* **`MPPO`** [Towards Optimally Decentralized Multi-Robot Collision Avoidance via Deep Reinforcement Learning](https://arxiv.org/abs/1709.10082), Long P. et al. (2017). [🎞️](https://sites.google.com/view/drlmaca)
* **`COMA`** [Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1709.05273), Foerster J. et al. (2017).
* **`MADDPG`** [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275), Lowe R. et al (2017). [:octocat:](https://github.com/openai/maddpg) 
* **`FTW`** [Human-level performance in first-person multiplayer games with population-based deep reinforcement learning](https://arxiv.org/abs/1807.01281), Jaderberg M. et al. (2018). [🎞️](https://www.youtube.com/watch?v=dltN4MxV1RI)
* [Towards Learning Multi-agent Negotiations via Self-Play](https://arxiv.org/abs/2001.10208), Tang Y. C. (2020).

## Representation Learning

* [Variable Resolution Discretization in Optimal Control](https://rd.springer.com/content/pdf/10.1023%2FA%3A1017992615625.pdf), Munos R., Moore A. (2002). [🎞️](http://researchers.lille.inria.fr/~munos/variable/index.html)
* **`DeepDriving`** [DeepDriving: Learning Affordance for Direct Perception in Autonomous Driving](http://deepdriving.cs.princeton.edu/paper.pdf), Chen C. et al. (2015). [🎞️](https://www.youtube.com/watch?v=5hFvoXV9gII)
* [On the Sample Complexity of End-to-end Training vs. Semantic Abstraction Training](https://arxiv.org/abs/1604.06915), Shalev-Shwartz S. et al. (2016).
* [Learning sparse representations in reinforcement learning with sparse coding](https://arxiv.org/abs/1707.08316), Le L., Kumaraswamy M., White M. (2017).
* [World Models](https://arxiv.org/abs/1803.10122), Ha D., Schmidhuber J. (2018). [🎞️](https://worldmodels.github.io/) [:octocat:](https://github.com/ctallec/world-models)
* [Learning to Drive in a Day](https://arxiv.org/abs/1807.00412), Kendall A. et al. (2018). [🎞️](https://www.youtube.com/watch?v=eRwTbRtnT1I)
* **`MERLIN`** [Unsupervised Predictive Memory in a Goal-Directed Agent](https://arxiv.org/abs/1803.10760), Wayne G. et al. (2018). [🎞️ 1](https://www.youtube.com/watch?v=YFx-D4eEs5A) | [2](https://www.youtube.com/watch?v=IiR_NOomcpk) | [3](https://www.youtube.com/watch?v=dQMKJtLScmk) | [4](https://www.youtube.com/watch?v=xrYDlTXyC6Q) | [5](https://www.youtube.com/watch?v=04H28-qA3f8) | [6](https://www.youtube.com/watch?v=3iA19h0Vvq0)
* [Variational End-to-End Navigation and Localization](https://arxiv.org/abs/1811.10119), Amini A. et al. (2018). [🎞️](https://www.youtube.com/watch?v=aXI4a_Nvcew)
* [Making Sense of Vision and Touch: Self-Supervised Learning of Multimodal Representations for Contact-Rich Tasks](https://arxiv.org/pdf/1810.10191.pdf), Lee M. et al. (2018). [🎞️](https://www.youtube.com/watch?v=TjwDJ_R2204)
* [Deep Neuroevolution of Recurrent and Discrete World Models](http://sebastianrisi.com/wp-content/uploads/risi_gecco19.pdf), Risi S., Stanley K.O. (2019). [🎞️](https://www.youtube.com/watch?v=a-tcsnZe-yE) [:octocat:](https://github.com/sebastianrisi/ga-world-models)
* **`FERM`** [A Framework for Efficient Robotic Manipulation](https://sites.google.com/view/efficient-robotic-manipulation), Zhan A., Zhao R. et al. (2021). [:octocat:](https://github.com/PhilipZRH/ferm)

## Other

* [Is the Bellman residual a bad proxy?](https://arxiv.org/abs/1606.07636), Geist M., Piot B., Pietquin O. (2016).
* [Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560), Henderson P. et al. (2017).
* [Automatic Bridge Bidding Using Deep Reinforcement Learning](https://arxiv.org/abs/1607.03290), Yeh C. and Lin H. (2016).
* [Shared Autonomy via Deep Reinforcement Learning](https://arxiv.org/abs/1802.01744), Reddy S. et al. (2018). [🎞️](https://sites.google.com/view/deep-assist)
* [Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review](https://arxiv.org/abs/1805.00909), Levine S. (2018).
* [The Value Function Polytope in Reinforcement Learning](https://arxiv.org/abs/1901.11524), Dadashi R. et al. (2019).
* [On Value Functions and the Agent-Environment Boundary](https://arxiv.org/abs/1905.13341), Jiang N. (2019).
* [How to Train Your Robot with Deep Reinforcement Learning; Lessons We've Learned](https://arxiv.org/abs/2102.02915), Ibartz J. et al (2021).


# Learning from Demonstrations :mortar_board:

## Imitation Learning

* **`DAgger`** [A Reduction of Imitation Learning and Structured Predictionto No-Regret Online Learning](https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf), Ross S., Gordon G., Bagnell J. A. (2011).
* **`QMDP-RCNN`** [Reinforcement Learning via Recurrent Convolutional Neural Networks](https://arxiv.org/abs/1701.02392), Shankar T. et al. (2016). ([talk](https://www.youtube.com/watch?v=gpwA3QNTPOQ))
* **`DQfD`** [Learning from Demonstrations for Real World Reinforcement Learning](https://pdfs.semanticscholar.org/a7fb/199f85943b3fb6b5f7e9f1680b2e2a445cce.pdf), Hester T. et al. (2017). [🎞️](https://www.youtube.com/watch?v=JR6wmLaYuu4&list=PLdjpGm3xcO-0aqVf--sBZHxCKg-RZfa5T)
* [Find Your Own Way: Weakly-Supervised Segmentation of Path Proposals for Urban Autonomy](https://arxiv.org/abs/1610.01238), Barnes D., Maddern W., Posner I. (2016). [🎞️](https://www.youtube.com/watch?v=rbZ8ck_1nZk)
* **`GAIL`** [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476), Ho J., Ermon S. (2016).
* [From perception to decision: A data-driven approach to end-to-end motion planning for autonomous ground robots](https://arxiv.org/abs/1609.07910), Pfeiffer M. et al. (2017). [🎞️](https://www.youtube.com/watch?v=ZedKmXzwdgI)
* **`Branched`** [End-to-end Driving via Conditional Imitation Learning](https://arxiv.org/abs/1710.02410), Codevilla F. et al. (2017). [🎞️](https://www.youtube.com/watch?v=cFtnflNe5fM) | [talk](https://www.youtube.com/watch?v=KunVjVHN3-U)
* **`UPN`** [Universal Planning Networks](https://arxiv.org/abs/1804.00645), Srinivas A. et al. (2018). [🎞️](https://sites.google.com/view/upn-public/home)
* **`DeepMimic`** [DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills](https://xbpeng.github.io/projects/DeepMimic/index.html), Peng X. B. et al. (2018). [🎞️](https://www.youtube.com/watch?v=vppFvq2quQ0&feature=youtu.be)
* **`R2P2`** [Deep Imitative Models for Flexible Inference, Planning, and Control](https://arxiv.org/abs/1810.06544), Rhinehart N. et al. (2018). [🎞️](https://sites.google.com/view/imitativeforecastingcontrol)
* [Learning Agile Robotic Locomotion Skills by Imitating Animals](https://xbpeng.github.io/projects/Robotic_Imitation/index.html), Bin Peng X. et al (2020). [🎞️](https://www.youtube.com/watch?v=lKYh6uuCwRY)
* [Deep Imitative Models for Flexible Inference, Planning, and Control](https://openreview.net/pdf?id=Skl4mRNYDr), Rhinehart N., McAllister R., Levine S. (2020). 

### Applications to Autonomous Driving :car:

* [ALVINN, an autonomous land vehicle in a neural network](https://papers.nips.cc/paper/95-alvinn-an-autonomous-land-vehicle-in-a-neural-network), Pomerleau D. (1989).
* [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316), Bojarski M. et al. (2016). [🎞️](https://www.youtube.com/watch?v=qhUvQiKec2U)
* [End-to-end Learning of Driving Models from Large-scale Video Datasets](https://arxiv.org/abs/1612.01079), Xu H., Gao Y. et al. (2016). [🎞️](https://www.youtube.com/watch?v=jxlNfUzbGAY)
* [End-to-End Deep Learning for Steering Autonomous Vehicles Considering Temporal Dependencies](https://arxiv.org/abs/1710.03804), Eraqi H. et al. (2017).
* [Driving Like a Human: Imitation Learning for Path Planning using Convolutional Neural Networks](https://www.semanticscholar.org/paper/Driving-Like-a-Human%3A-Imitation-Learning-for-Path-Rehder-Quehl/a1150417083918c3f5f88b7ddad8841f2ce88188), Rehder E. et al. (2017).
* [Imitating Driver Behavior with Generative Adversarial Networks](https://arxiv.org/abs/1701.06699), Kuefler A. et al. (2017).
* **`PS-GAIL`** [Multi-Agent Imitation Learning for Driving Simulation](https://arxiv.org/abs/1803.01044), Bhattacharyya R. et al. (2018). [🎞️](https://github.com/sisl/ngsim_env/blob/master/media/single_multi_model_2_seed_1.gif) [:octocat:](https://github.com/sisl/ngsim_env)
* [Deep Imitation Learning for Autonomous Driving in Generic Urban Scenarios with Enhanced Safety](https://arxiv.org/abs/1903.00640), Chen J. et al. (2019).

## Inverse Reinforcement Learning

* **`Projection`** [Apprenticeship learning via inverse reinforcement learning](http://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf), Abbeel P., Ng A. (2004).
* **`MMP`** [Maximum margin planning](https://www.ri.cmu.edu/pub_files/pub4/ratliff_nathan_2006_1/ratliff_nathan_2006_1.pdf), Ratliff N. et al. (2006).
* **`BIRL`** [Bayesian inverse reinforcement learning](https://www.aaai.org/Papers/IJCAI/2007/IJCAI07-416.pdf), Ramachandran D., Amir E. (2007).
* **`MEIRL`** [Maximum Entropy Inverse Reinforcement Learning](https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf), Ziebart B. et al. (2008).
* **`LEARCH`** [Learning to search: Functional gradient techniques for imitation learning](https://www.ri.cmu.edu/pub_files/2009/7/learch.pdf), Ratliff N., Siver D. Bagnell A. (2009).
* **`CIOC`** [Continuous Inverse Optimal Control with Locally Optimal Examples](http://graphics.stanford.edu/projects/cioc/), Levine S., Koltun V. (2012). [🎞️](http://graphics.stanford.edu/projects/cioc/cioc.mp4)
* **`MEDIRL`** [Maximum Entropy Deep Inverse Reinforcement Learning](https://arxiv.org/abs/1507.04888), Wulfmeier M. (2015).
* **`GCL`** [Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization](https://arxiv.org/abs/1603.00448), Finn C. et al. (2016). [🎞️](https://www.youtube.com/watch?v=hXxaepw0zAw)
* **`RIRL`** [Repeated Inverse Reinforcement Learning](https://arxiv.org/abs/1705.05427), Amin K. et al. (2017).
* [Bridging the Gap Between Imitation Learning and Inverse Reinforcement Learning](http://ieeexplore.ieee.org/document/7464854/), Piot B. et al. (2017).

### Applications to Autonomous Driving :taxi:

* [Apprenticeship Learning for Motion Planning, with Application to Parking Lot Navigation](http://ieeexplore.ieee.org/document/4651222/), Abbeel P. et al. (2008).
* [Navigate like a cabbie: Probabilistic reasoning from observed context-aware behavior](http://www.cs.cmu.edu/~bziebart/publications/navigate-bziebart.pdf), Ziebart B. et al. (2008).
* [Planning-based Prediction for Pedestrians](http://ieeexplore.ieee.org/abstract/document/5354147/), Ziebart B. et al. (2009). [🎞️](https://www.youtube.com/watch?v=XOZ69Bg4JKg)
* [Learning for autonomous navigation](https://www.ri.cmu.edu/pub_files/2010/6/Learning%20for%20Autonomous%20Navigation-%20Advances%20in%20Machine%20Learning%20for%20Rough%20Terrain%20Mobility.pdf), Bagnell A. et al. (2010).
* [Learning Autonomous Driving Styles and Maneuvers from Expert Demonstration](https://www.ri.cmu.edu/pub_files/2012/6/iser12.pdf), Silver D. et al. (2012).
* [Learning Driving Styles for Autonomous Vehicles from Demonstration](http://ieeexplore.ieee.org/document/7139555/), Kuderer M. et al. (2015).
* [Learning to Drive using Inverse Reinforcement Learning and Deep Q-Networks](https://arxiv.org/abs/1612.03653), Sharifzadeh S. et al. (2016).
* [Watch This: Scalable Cost-Function Learning for Path Planning in Urban Environments](https://arxiv.org/abs/1607.02329), Wulfmeier M. (2016). [🎞️](https://www.youtube.com/watch?v=Sdfir_1T-UQ)
* [Planning for Autonomous Cars that Leverage Effects on Human Actions](https://robotics.eecs.berkeley.edu/~sastry/pubs/Pdfs%20of%202016/SadighPlanning2016.pdf), Sadigh D. et al. (2016).
* [A Learning-Based Framework for Handling Dilemmas in Urban Automated Driving](http://ieeexplore.ieee.org/document/7989172/), Lee S., Seo S. (2017).
* [Learning Trajectory Prediction with Continuous Inverse Optimal Control via Langevin Sampling of Energy-Based Models](https://arxiv.org/abs/1904.05453), Xu Y. et al. (2019).
* [Analyzing the Suitability of Cost Functions for Explaining and Imitating Human Driving Behavior based on Inverse Reinforcement Learning](https://ras.papercept.net/proceedings/ICRA20/0320.pdf), Naumann M. et al (2020).



# Motion Planning :running_man:

## Search

* **`Dijkstra`** [A Note on Two Problems in Connexion with Graphs](http://www-m3.ma.tum.de/foswiki/pub/MN0506/WebHome/dijkstra.pdf), Dijkstra E. W. (1959).
* **`A*`** [ A Formal Basis for the Heuristic Determination of Minimum Cost Paths ](http://ieeexplore.ieee.org/document/4082128/), Hart P. et al. (1968).
* [Planning Long Dynamically-Feasible Maneuvers For Autonomous Vehicles](https://www.cs.cmu.edu/~maxim/files/planlongdynfeasmotions_rss08.pdf), Likhachev M., Ferguson D. (2008).
* [Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame](https://www.researchgate.net/profile/Moritz_Werling/publication/224156269_Optimal_Trajectory_Generation_for_Dynamic_Street_Scenarios_in_a_Frenet_Frame/links/54f749df0cf210398e9277af.pdf), Werling M., Kammel S. (2010). [🎞️](https://www.youtube.com/watch?v=Cj6tAQe7UCY)
* [3D perception and planning for self-driving and cooperative automobiles](http://www.mrt.kit.edu/z/publ/download/2012/StillerZiegler2012SSD.pdf), Stiller C., Ziegler J. (2012).
* [Motion Planning under Uncertainty for On-Road Autonomous Driving](https://www.ri.cmu.edu/pub_files/2014/6/ICRA14_0863_Final.pdf), Xu W. et al. (2014).
* [Monte Carlo Tree Search for Simulated Car Racing](http://julian.togelius.com/Fischer2015Monte.pdf), Fischer J. et al. (2015). [🎞️](https://www.youtube.com/watch?v=GbUMssvolvU)

## Sampling

* **`RRT*`** [Sampling-based Algorithms for Optimal Motion Planning](https://arxiv.org/abs/1105.1186), Karaman S., Frazzoli E. (2011). [🎞️](https://www.youtube.com/watch?v=p3nZHnOWhrg)
* **`LQG-MP`** [LQG-MP: Optimized Path Planning for Robots with Motion Uncertainty and Imperfect State Information](https://people.eecs.berkeley.edu/~pabbeel/papers/vandenBergAbbeelGoldberg_RSS2010.pdf), van den Berg J. et al. (2010).
* [Motion Planning under Uncertainty using Differential Dynamic Programming in Belief Space](http://rll.berkeley.edu/~sachin/papers/Berg-ISRR2011.pdf), van den Berg J. et al. (2011).
* [Rapidly-exploring Random Belief Trees for Motion Planning Under Uncertainty](https://groups.csail.mit.edu/rrg/papers/abry_icra11.pdf), Bry A., Roy N. (2011).
* **`PRM-RL`** [PRM-RL: Long-range Robotic Navigation Tasks by Combining Reinforcement Learning and Sampling-based Planning](https://arxiv.org/abs/1710.03937), Faust A. et al. (2017).

## Optimization

* [Trajectory planning for Bertha - A local, continuous method](https://pdfs.semanticscholar.org/bdca/7fe83f8444bb4e75402a417053519758d36b.pdf), Ziegler J. et al. (2014).
* [Learning Attractor Landscapes for Learning Motor Primitives](https://papers.nips.cc/paper/2140-learning-attractor-landscapes-for-learning-motor-primitives.pdf), Ijspeert A. et al. (2002).
* [Online Motion Planning based on Nonlinear Model Predictive Control with Non-Euclidean Rotation Groups](https://arxiv.org/abs/2006.03534), Rösmann C. et al (2020). [:octocat:](https://github.com/rst-tu-dortmund/mpc_local_planner)

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
