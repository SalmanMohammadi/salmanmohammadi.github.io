---
title: The theory of Proximal Policy Optimization implementations
layout: post
---
## Prelude

The aim of this post is to share my understanding of some of the conceptual and theoretical background behind implementations of the [Proximal Policy Algorithm](https://arxiv.org/pdf/1506.02438.pdf) (PPO) reinforcement learning (RL) algorithm. PPO is widely used due to its stability and sample efficiency - popular applications include [beating the Dota 2 world champions](https://openai.com/research/openai-five-defeats-dota-2-world-champions) and [aligning language models](https://arxiv.org/pdf/2203.02155.pdf). While the PPO paper provides quite a general and straightforward overview of the algorithm, modern implementations of PPO use several additional techniques to achieve state-of-the-art  performance in complex environments{% sidenote "0" "[Procgen, Karle Cobbe et al.](https://openai.com/research/procgen-benchmark)<br>[Atari, OpenAI Gymnasium](https://gymnasium.farama.org/environments/atari/)" %}. You might discover this if you try to implement the algorithm solely based on the paper. I try and present a coherent narrative here around these additional techniques.

I'd recommend reading parts [one](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html), [two](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html), and [three](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) of [SpinningUp](https://spinningup.openai.com/en/latest/index.html) if you're new to reinforcement learning. There's a few longer-form educational resources that I'd recommend if you'd like a broader understanding of the field{% sidenote "1" "- [A (Long) Peek into Reinforcement Learning, Lilian Weng](https://lilianweng.github.io/posts/2018-02-19-rl-overview/)<br> - [Artificial Intelligence: A Modern Approach, Stuart Russell and Peter Norvig](https://aima.cs.berkeley.edu/) <br>- [Reinforcement Learning: An Introduction, Sutton and Barto](http://incompleteideas.net/book/the-book-2nd.html)<br>[CS285 at UC Berkeley, Deep Reinforcement Learning, Sergey Levine](https://rail.eecs.berkeley.edu/deeprlcourse/)" %}, but this isn't comprehensive. You should be familiar with common concepts and terminology in RL{% sidenote "2" "[Lilian Weng's list of RL notation is very useful here](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/#notations)" %}. For clarity, I'll try to spell out any jargon I use here.

## Recap

### Policy Gradient Methods
PPO is an on-policy reinforcement learning algorithm. It directly learns a stochastic policy function parameterised by $\theta$ representing the likelihood of action $a$ in state $s$, $\pi_{\theta}(a\vert s)$. Consider that we have some differentiable function, $J(\theta)$, which is a continuous performance measure of the policy $\pi_{\theta}$.  In the simplest case, we have  $J(\theta)={\mathbb{E}}\_{\tau\sim{\pi_{\theta}}}[R(\tau)]$, which is known as the [*return*](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#reward-and-return){% sidenote "3" "The return represents the sum of rewards achieved over some time frame. This can be over a fixed timescale, i.e. the *finite-horizon* return, or over all time, i.e. the *infinite-horizon* return." %} over a [*trajectory*](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#trajectories){% sidenote "2" "A trajectory, $\tau$, (also known as an episode or rollout) describes a sequence of interactions between the agent and the environment." %}, $\tau$. PPO is a kind of policy gradient method{% sidenote "3" "- [Policy Gradient Algorithms, Lilian Weng](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/#policy-gradient-theorem)<br>- [Policy Gradients, CS285 UC Berkeley, Lecture 5, Sergey Levine](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf)" %} which directly optimizes the policy parameters $\theta$ against $J(\theta)$. The [policy gradient theorem](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/#proof-of-policy-gradient-theorem) shows that:
 
$$
\nabla_{\theta}J(\theta) = \mathbb{E}[\sum\limits_{t=0}^{\inf}\nabla_{\theta}\,ln\,\pi_\theta(a_t|s_t)\,R_t]
$$

In other words, the gradient of our performance measure $J(\theta)$ with respect to our policy parameters $\theta$ points in the direction of maximising the return $R_t$. Crucially, this shows that we can estimate the true gradient using an expectation of the sample gradient - the core idea behind the REINFORCE{% sidenote "4" "[Reinforcement Learning: An Introduction, 13.3 REINFORCE: Monte Carlo Policy Gradient, Sutton and Barto](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)" %} algorithm. This is great. This expression has the more general form which substitutes $R_t$ for some lower-variance estimator of the total expected reward, $\Phi${% sidenote "5" "[Section 2, High Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et. al. 2016](https://arxiv.org/pdf/1506.02438.pdf)" %}.

$$
\begin{equation}
\nabla_{\theta}J(\theta) = \mathbb{E}[\sum\limits_{t=0}^{\inf}\nabla_{\theta}\,ln\,\pi_\theta(a_t|s_t)\,\Phi_t ]
\end{equation}
$$

Modern implementations of PPO make the choice of $\Phi_t=A^{\pi}(s_{t,}a_t)$, the [*advantage function*](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#advantage-functions). This function estimates the *advantage* of a particular action in a given state over the expected value of following the policy, i.e. how much better is taking this action in this state over all other actions? Briefly described here, the advantage function takes the form

$$
A^{\pi}(s, a)= Q^\pi(s, a) - V^{\pi}(s)
$$

where $V(s)$ is the state-value function, and $Q(s, a)$ is the state-action -value function, or Q-function{% sidenote "6" "The value function, $V(s)$, describes the expected return from starting in state $s$. Similarly the state-action value function, $Q(s, a)$, describes the expected return from starting in state $s$ and taking action $a$. See also [Reinforcement Learning: An Introduction, 3.7 Value Functions, Sutton and Barto](http://incompleteideas.net/book/ebook/node34.html)" %}. I've found it easier to intuit the nuances of PPO by following the narrative around its motivations and predecessor. PPO iterates on the [Trust Region Policy Optimization](https://arxiv.org/pdf/1502.05477.pdf)(TRPO) method which constrains the objective function with respect to the size of the policy update. The TRPO objective function is defined as:{% sidenote "6" "I'm omitting $ln$ from $ln\,\pi(a_{t}\vert s_{t})$ for brevity from here on." %} {% sidenote "7" "- [Proximal Policy Optimization Algorithms, Section 2.2, Schulman et al.](https://arxiv.org/abs/1707.06347)<br>- [Trust Region Policy Optimization, Schulman et al.](https://arxiv.org/pdf/1502.05477.pdf) for further details on the constraint."%}

$$
\begin{aligned}
J(\theta) = \,&\mathbb{E} [\frac{\pi_\theta(a_t,s_t)}{\pi_{\theta_{old}}(a_t,s_t)}A_t]\\
\textrm{subject to}\,\, &\mathbb{E} [\textrm{KL}(\pi_{\theta_{old}}\vert \vert \pi_{\theta} )] \le \delta
\end{aligned}
$$

Where KL is the Kullback-Liebler divergence (a measure of distance between two probability distributions), and the size of policy update is defined as the ratio between the new policy and the old policy:

$$
r(\theta)=\frac{\pi_{\theta}(a_t,s_{t})}{\pi_{\theta_{old}}(a_t,s_{t})}$$

Policy gradient methods optimise policies through (ideally small) iterative gradient updates to parameters $\theta$. The old policy, $\pi_{\theta_{old}}(a_t,s_{t})$, is the one used to generate the current trajectory, and the new policy, $\pi_{\theta}(a_t,s_{t})$ is the policy currently being optimised{% sidenote "8" "Note: at the start of a series of policy update steps, we have $\pi_{\theta_{old}}(a_t,s_{t})=\pi_{\theta}(a_t,s_{t})$, so $r(\theta)=1$." %}. If the advantage is positive, then the new policy becomes greedier relative to the old policy{% sidenote "9" "The new policy will place higher density on actions relative to the old policy, i.e. $\pi_{\theta}(a_t,s_{t}) > \pi_{\theta_{old}}(a_t,s_{t})$." %}, and we have $r(\theta) > 1$ - the inverse applies for negative advantage, where we have $r(\theta) < 1$.  The core principle of TRPO (and PPO) is to prevent large parameter updates which occur due to variance in environment and training dynamics, thus helping to stabilise optimisation. Updating using the ratio between old and new policies in this way allows for selective reinforcement or penalisation of actions whilst grounding updates relative to the original, stable policy{% sidenote "10" "Consider optimising our policy using eqn. 1 - without normalising the update w.r.t. the old policy, updates to the policy can lead to catastrophically large updates." %}.

### PPO
PPO modifies the TRPO objective function by constraining the objective function:

$$
J(\theta) = \mathbb{E} [min(\frac{\pi_\theta(a_t,s_t)}{\pi_{\theta_{old}}(a_t,s_t)}A_t, clip(\frac{\pi_\theta(a_t,s_t)}{\pi_{\theta_{old}}(a_t,s_{t)}}, 1 - \epsilon, 1 + \epsilon)A_t]
$$

To break this somewhat dense equation down, we can substitute our earlier expression $r(\theta)$ in:
 
 $$
 J(\theta)=\mathbb{E}[\textrm{min}(r(\theta)\,A_t, \textrm{clip}(r(\theta), 1-\epsilon, 1+\epsilon)\,A_t)]
 $$
 
 The *clip* term constrains the policy ratio, $r$, to lie between $1-\epsilon$ and $1+\epsilon${% sidenote "11" "$\epsilon$ is usually set to $\sim0.1$." %}. In this manner, the objective function disincentives greedy policy updates in the direction of improving the objective. When the new policy places lower density on actions compared to the previous policy, i.e. may be more conservative, the advantage update is smaller. When the new policy places higher density on actions compared to the previous policy i.e. is greedier, the advantage update is *also* smaller. This is why PPO is considered to place a lower pessimistic bound on policy updates. The policy is only updated by  $1+\epsilon$ or $1-\epsilon$, depending on whether the advantage function is positive or negative, respectively.

So far, we've introduced the concept of policy gradients, objective functions in RL, and the core concepts behind PPO. Reinforcement learning algorithms place significant emphasis in reducing variance during optimisation{% sidenote "12" "Stochasticity in environment dynamics, delayed rewards, and exploration-exploitation tradeoffs all contribute to unstable training." %}. This becomes apparent in estimating the advantage function which relies on rewards obtained during a trajectory. Practical implementations of policy gradient algorithms take additional steps to trade variance for bias here by also estimating an *on-policy* state-value function $V^{\pi}_\phi(s_t)$, which is the expected return an agent receives from starting in state $s_t$, and following policy $\pi$ thereafter. Jointly learning a value function and policy function in this way is known as the Actor-Critic framework{% sidenote "13" "- [Reinforcement Learning: An Introduction, 6.6 Actor-Critic Methods, Sutton and Barto](http://incompleteideas.net/book/ebook/node66.html)<br>- [A (Long) Peek into Reinforcement Learning, Value Function, Lilian Weng](https://lilianweng.github.io/posts/2018-02-19-rl-overview/#value-function)" %}.  
### Value Learning and Actor-Critics
Value-function learning involves approximating the future rewards of following a policy from a current state. The value function is learned alongside the policy, and the simplest method is to minimise a mean-squared-error objective against the discounted return{% sidenote "14" "The *discounted return* down-weights rewards obtained in the future by an exponential discount factor $\gamma^t$, i.e. rewards in the distant future aren't worth as much as near-term rewards." %}, $R(\tau) = \sum\limits_{t=0}^{\inf}\gamma^{t}r_t$:

$$
\phi^{*}=\text{arg min}_{\phi}\mathbb{E}_{s_{t,R_{t}\sim}\pi}[(V_\phi(s_t)-\gamma^tr(s_t))^2]
$$

We can now use this to learn a lower-variance estimator of the advantage function {% sidenote "15" "The on-policy Q-function is defined as the expected reward of taking action $a_t$ in state $s_t$, and following policy $\pi$ thereafter: $Q^\pi(s, a) = \mathbb{E}[r_t+\gamma V^\pi(s_{t+1})]$" %}:

$$
\begin{aligned}
\hat{A}^{\pi}(s, a) &= \mathbb{E}_{s_{t+1}} [ Q^{\pi}(s_t, a_t) - V_\phi^\pi(s_t)]\\
 &=  \mathbb{E}_{s_{t+1}} [r_t+\gamma V_{\phi}^{\pi}(s_{t+1})- V_{\phi}^{\pi}(s)]\\
\end{aligned}
$$

We end up with an estimator of the advantage function that solely relies on samples rewards and our learned approximate value function. In fact, our expression shows that the advantage function can be estimated with the temporal-difference{% sidenote "16" "[Reinforcement Learning: An Introduction, 6. Temporal-Difference Learning, Sutton and Barto](http://incompleteideas.net/book/ebook/node60.html)" %} residual error, $\delta_t$, of the value function:

$$
\begin{aligned}
V_\phi(s_t)&\leftarrow V_\phi(s_{t)}+ \alpha(r_t+\gamma V_\phi(s_{t+1}) - V_\phi(s_t))\\
V_\phi(s_t)&\leftarrow V_\phi(s_{t)} + \alpha\,\delta_t\\
\textrm{where}\,\,\delta_t&=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)
\end{aligned}
$$

### Generalised Advantage Estimation (GAE)

There's one more thing we can do to reduce variance. The current advantage estimator estimates reward by samples collected from a single trajectory. However, there is a huge amount of variance in the possible trajectories that may evolve from a given state. These trajectories may look similar in the short-term (except for policies early in the optimisation process which are far more random), but longer-term rewards can vary significantly. We could consider a lower-variance estimate of the advantage by sampling trajectories - once again trading variance for bias. This is the central idea behind *n*-step returns{% sidenote "17" "- [DeepMimic Supplementary A, Peng et al.](https://arxiv.org/pdf/1804.02717.pdf)<br>- [Reinforcement Learning: An Introduction, 7.1 n-step TD Prediction, Sutton and Barto](http://incompleteideas.net/book/ebook/node73.html)" %}. Consider the term $\delta_t$ in our estimation of the advantage function. we take the initial reward observed from the environment, $r_t$, then bootstrap future estimated discounted rewards, and subtract a baseline estimated value function for the state{% sidenote "18" "Daniel Takeshi's [post](https://danieltakeshi.github.io/2017/03/28/going-deeper-into-reinforcement-learning-fundamentals-of-policy-gradients/) on using baselines to reduce variance of gradient estimates is useful here." %}. For a single time-step, this can be denoted as:

$$
\hat{A^{(1)}_{t}} = r_t+\gamma V_{\phi}^{\pi}(s_{t+1})- V_{\phi}^{\pi}(s) = \delta_t^{V_\phi}
$$

What if we sample rewards from multiple timesteps, and then estimate the future discounted rewards from then on? Let's denote $\hat{A_t}^{(k)}$ as follows:

$$
\begin{aligned}
\hat{A_{t}}^{(1)} &= r_t+\gamma V_{\phi}^{\pi}(s_{t+1})- V_{\phi}^{\pi}(s)&&=\delta_t^{V_\phi}\\

\hat{A_{t}}^{(2)} &= r_{t}+\gamma r_{t+1} + \gamma^{2}V_{\phi}^{\pi}(s_{t+2})- V_{\phi}^{\pi}(s) &&= \delta_t^{V_{\phi}}+ \gamma\delta_{t+1}^{V_\phi}\\

\hat{A_{t}}^{(k)} &= r_{t}+\gamma r_{t+1} + ... + r_{t+k-1} + \gamma^{k}V_{\phi}^{\pi}(s_{t+k})- V_{\phi}^{\pi}(s)&&= \delta_t^{V_{\phi}}+ \gamma\delta_{t+1}^{V_{\phi}}+\gamma^2\delta_{t+1}^{V_{\phi}}\\

\hat{A_{t}}^{(\infty)} &= r_{t}+\gamma r_{t+1} +  +\gamma^{2}  r_{t+2}+ ... -  V_{\phi}^{\pi}(s) && = \sum\limits_{l=0}^{k-1}\gamma^l\delta_{t+1}^{V_\phi}\\
\end{aligned}
$$

Observe that for $k=\infty$ we recover an unbiased, high-variance expectation of the infinite-horizon discounted return, minus our baseline estimate value function. GAE{% sidenote "19" "- [High Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et. al. 2016](https://arxiv.org/pdf/1506.02438.pdf)<br>- [Notes on the Generalized Advantage Estimation Paper, Daniel Takeshi](https://danieltakeshi.github.io/2017/04/02/notes-on-the-generalized-advantage-estimation-paper/)" %} introduces a discount factor, $\lambda \in[0,1]$, to take an exponentially weighted average over every $k$-th step estimator. Using notation from the paper{% sidenote "20" "The identity $\frac{1}{1-k} = 1 + k + k^{2}+ k^{3} + ...$ for $|k|<1$ is useful to note here."%}, we can derive a *generalized advantage estimator* for cases where $0<\lambda<1$:

$$
\begin{aligned}
\hat{A_{t}}^{\gamma \lambda} &= (1-\lambda)(\hat{A_{t}}^{(1)} + \lambda\hat{A_{t}}^{(2)} + \lambda^2\hat{A_{t}}^{(3)} + ...)\\

&= (1-\lambda)(\delta_{t}^{V_\phi} + \lambda(\delta_{t}^{V_{\phi}}+ \gamma\delta_{t+1}^{V_\phi})  + \lambda^{2}(\delta_{t}^{V_\phi}+\gamma\delta_{t+1}^{V_{\phi}}+ \gamma^2\delta_{t+2}^{V_\phi})+ ...)\\

&= (1-\lambda)(\delta_{t}^{V_\phi}(1+\lambda+\lambda^2+...) + \gamma\delta_{t+1}^{V_\phi}(\lambda+\lambda^2+\lambda^3+...)+...)\\

&= (1-\lambda)\left(\delta_{t}^{V_\phi}\left(\frac{1}{1-\lambda}\right)+\gamma\delta_{t}^{V_\phi}\left(\frac{\lambda}{1-\lambda}\right)+\gamma^2\delta_{t}^{V_\phi}(\frac{\lambda^2}{1-\lambda})\right)\\

&= \sum\limits_{l=1}^{\infty}(\gamma \lambda)^{l}\delta_{t+1}^{V_\phi}

\end{aligned}
$$

Great. As you may have noticed, there's two special cases for this expression - $\lambda=0$, and $\lambda=1$: 

$$
\begin{aligned}
\hat{A_t}^{\gamma*0}&=\hat{A_t}^{(1)}=r_t+\gamma V_{\phi}^{\pi}(s_{t+1})- V_{\phi}^{\pi}(s)=\delta_t\\

\hat{A_t}^{\gamma*1}&= \sum\limits_{l=1}^{\infty}\gamma^{l}\delta_{t+1}^{V_\phi}&&\\

&=(r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_{t})) \\

&+ \gamma(r_{t+1}+\gamma V_\phi(s_{t+2})-V_\phi(s_{t+1}))\\

&+ \gamma^2(r_{t+2}+\gamma V_\phi(s_{t+3})-V_\phi(s_{t+2}))+ ... \\

&= r_{t+} \bcancel{\gamma V_\phi(s_{t+1})}-V_\phi(s_{t})\\

&+ \gamma r_{t+1} + \bcancel{\gamma^{2}V_\phi(s_{t+2})}-\bcancel{\gamma V_\phi(s_{t+1})}\\

&+ \gamma^2 r_{t+2} + \gamma^{3}V_\phi(s_{t+3})-\bcancel{\gamma^2 V_\phi(s_{t+2})}+...\\

&=\sum\limits_{l=1}^{\infty}\gamma^{l}r_{t+1}- V_{\phi}(s_t)
\end{aligned}
$$

We see that $\lambda=0$ obtains the original biased low-variance actor-critic advantage estimator. For $\lambda=1$, we obtain a low-bias, high-variance advantage estimator, which is simply the discounted return minus our baseline estimated value function. For $\lambda \in (0,1)$, we obtain an advantage estimator which allows control of the bias-variance tradeoff. The authors note that the two parameters, $\gamma$ and $\lambda$, control variance in different ways. Lower values of $\gamma$ discount future rewards, which will always result in a biased estimate of the return, since there's an implicit prior that future rewards are less valuable. The authors note that $\gamma$ controls the strength of this prior regardless of how accurate our value function is. On the other hand, since $n$-step returns are unbiased estimators of the advantage function, lower values of $\lambda$ reduce variance when the value function is accurate. In other words, when $V^{\phi}(s_{t})=V(s_t)$ and for $0 < \lambda < 1$ , we obtain an unbiased estimator of the advantage function.
### Pseudocode

Tying everything together, we can show the general process of updating our policy and value function using PPO for a single trajectory of fixed length $N$:<br><br>
> Given policy estimator $\pi_\theta$, value function estimator $V^\phi$, $\gamma$, $\lambda$, $N$ time-steps.
>>For $t=1,2,...,N:$
>>
>>>Run policy $\pi_\theta$ in environment and collect rewards, observations, actions, and value function estimates $r_{t}, s_{t}, a_r, v_t$ where $v_t=V^\phi(s_t)$
>>
>>Compute $\delta_t^{V^{\phi}}=r_t+\gamma v_{t+1}-v_t$ for all $t$.
>>
>>Compute generalized advantage estimate, $\hat{A_{t}}=\sum\limits_{l=0}^{N}(\gamma\lambda)^t\delta_t$ for all t.
>>
>>Sample $\pi_\theta(a_t, s_t)$ $log$-probabilities from stored actions and states.
>>
>>Optimise $\theta$ using $J(\theta)$, the PPO objective{% sidenote "21" "This is usually done over $M$ minibatch steps for $M \le N$. $\pi_{\theta_{old}}$ is fixed as the initial policy at the start of the trajectory, and $\pi_\theta$ is taken as the policy at the current optimisation step." %}. 
>>
>> Optimise $\phi$ using $L=(V^\phi(s_t)-\gamma^t r_t)^2$ using stored $r_t$ and $s_t${% sidenote "22" " Similarly to the policy optimisation step, this is also done over $M$ steps. $V^\phi$ is taken as the value function at the current optimisation step, i.e. value estimates are bootstrapped." %}.
>
>... repeat!


Feedback and corrections are very welcomed and appreciated. I'll follow this post with implementation details soon. For now, I'd recommend the excellent resource by [Shengyi Huang](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) on reproducing PPO implementations from popular RL libraries. If you're able to implement the policy update from the PPO paper, hopefully there's enough detail here that you're able to reproduce other implementations of PPO. Thank you so much for reading.  