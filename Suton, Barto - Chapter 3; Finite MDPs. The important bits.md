---
tags:
  - ML
  - CS
  - Agents
---
# Chapter 3 - Finite Markov Decision Processes 
---
## 3.1 Agent-Environment Interface
![[Screenshot 2026-02-06 at 4.39.25 AM.png]]

Simply speaking, the agent is the model, the agent is the ==policy==, **it is what we control,** it can be a bunch of 'if-else' statements, or it can be a stochastic neural network, or whatever. 

The environment, at a ==state== $S_t$ is where the agent acts, with an ==action== $A_t : t:=$ time step. The environment progresses one time step after the 'receiving' the action to time step $t+1$, and returns a reward and state $R_{t+1}, S_{t+1}$ for that action.

From the agent's perspective, the agent receives a state of the current time-step, and the reward from the action taken at the previous time-step.

More mathematical definition from Suton, Barto:
![[Screenshot 2026-02-06 at 4.48.14 AM.png]]

At each time-step the agent implements a mapping $\pi : \mathcal S \to \mathcal P(\mathcal A)$, this is known as the *policy.* Where $\mathcal P$ is the distribution over the action space $\mathcal A$.  $\pi_t(a|s)$ is the probability of choosing the action $a$ for state $s$ at time-step $t$. (The $t$ subscript matters if our policy changes over the time-step)

---
## 3.2 Goals & Rewards
The _reward hypothesis_ states that what we mean by our goals can be fully accomplished by maximising the total scalar quantity of $R_t$, the reward over all time-steps. 
$$\max \sum_t R_t$$

Remember; We choose what is rewarded, this is important as it is effectively 'prompt-engineering' for RL. We can reward a walking robot to move forward, and it instead of walking, starts moving like a centipede.

---
## 3.3 Returns
If the sequence of rewards received after time-step $t$ is is $R_{t}, R_{t+1}, R_{t+2}\dots$ then what are we maximising?
Well for the time step $t$ we maximise $G_t=\sum_{t}^T R_{t+1}$, T being the final time step.

This works well for cases like chess, or games with repeated interactions. These are called *episodes.* Think 1 episode $=$ 1 game of chess. Each episode ends in a terminal state after which we restart to the starting state (or just end things I guess.)

$\mathcal S$ for non-terminal states.
$\mathcal S^+$ also includes the terminal state.

There are cases where breaking things down into different episodes may not be possible$\dots$ Here we use continual tasks.

> On a side note, can't we just rephrase $G_{t_0}=\int_{t_0+1}^T R(t) dt$ ? $R(t)$ as reward depends on state, action and both of those also change as $a_t$? But I get they aren't technically functions of time.

### Discounting Returns
We can add in a ratio $\gamma$, discount rate, at which the later rewards are discounted. Essentially, some now > more later.
$$G_t=\sum_{k=0}^\infty\gamma^kR_{t+k+1}:\gamma\in[0, 1]$$
---
## 3.4 Unified notation for episodic and continuing tasks

For episodic tasks: Use the un-discounted, "obvious" return
For the continuing tasks: Use the discounted return which is $0\iff|R_t|$ is bounded and $\gamma < 1$

We create a unified notation by treating the terminal state in $\mathcal S^+$ as a 'self-absorbing' state which feeds into itself, and has a reward of $0$ so we can still use the discounted rate.

![[Screenshot 2026-02-06 at 5.36.05 AM.png]]

This gives us the unified notation for return as:

$$G_t = \sum_{k=0}^{T-t-1}\gamma^k R_{t+k+1}$$

> This can be solved using the *average* reward. 
> 
> $\rho^\pi=\lim_{n\to\infty}\frac 1nE[\sum_{i=1}^n r_t]$
>

---
## 3.5 Markov Property

We can't expect to always know *everything*, the state signal that we're receiving should only contain relevant information. Also - we can't also know *everything useful* always, take blackjack, you don't know what the other player's cards are.

Finally, what we want is a state signal with the *markov property* I.E. the current state contains all the relevant information and you do not need to look at previous states.

![[Screenshot 2026-02-06 at 5.51.30 AM.png]]

---
## 3.6 Markov Decision Process

>[!quote] Suton, Bartol
>Finite MDPs are all you need to know to understand $\sim$ 90% of modern reinforcement learning.

If a reinforcement learning task satisfies the markov property, it is an Markov Decision Process if the state and action spaces are finite then it is a finite MDP.

A particular finite MDP is defined by its state and action sets and by one-step dynamics of the environment.
>one-step dynamics -> can only model for immediate next step and next reward

Given a pair state $s$, action $a$ this is the probability for the next state $s'$ and reward $r$,
$$p(s', r|s, a)=Pr\{S_{t+1}=s', R_{t+1}=r|S_t=s, A_t=a\}$$

Completely specifies the dynamics of a finite MDP. Given the dynamics specified by this equation, we can compute everything else we want to know, such as expected reward for state-action pairs as:

$$r(s, a)=\mathbb E[R_{t+1}|S_t=s, A_t=a]=\sum_{r\in\mathcal R}r\sum_{s\in\mathcal S}p(s', r|s, a)$$

The state transition properties,
$$p(s' \mid s, a)
= \Pr\{S_{t+1} = s' \mid S_t = s, A_t = a\}
= \sum_{r \in \mathcal{R}} p(s', r \mid s, a)
$$

The expected reward for state-action-next-state triple
$$
r(s, a, s')
= \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a, S_{t+1} = s']
= \frac{\displaystyle \sum_{r \in \mathcal{R}} r\, p(s', r \mid s, a)}{p(s' | s, a)}
$$

---
## 3.7 Value Functions

Value function $v_\pi(s)$ tells you how good it is to be in a given state $s$, it is defined by,
$$v_\pi(s)=\mathbb E_\pi[G_t|S_t=s]=\mathbb E_\pi\left[\sum_{k=0}^\infty \gamma^kR_{t+k+1}\Bigg|S_t=s\right]$$

$v_\pi(s)$ is known as the state-value function for policy $\pi$, similarly we can have a function for action given as.$$q_\pi(s, a)=\mathbb E_\pi[G_t|S_t=s, A_t=a]=\mathbb E_\pi\left[\sum_{k=0}^\infty\gamma^k R_{t+k+1}\Bigg|S_t=s, A_t=a\right]$$
Known as the action-value function. Both value functions can be estimated from experience. Eg. Using *monte carlo methods.*

If there are very many states, it may not be practical to keep separate averages for each state, instead we maintain $v_\pi$ and $q_\pi$ as parameterised functions and adjust the parameters to better match the observed returns. 

Fundamental property of value functions used throughout RL is that they satisfy recursive relationships. For any policy $\pi$ and any state $s$ the following consistency condition holds between value of $s$ and value of its possible successor states:
$$v_\pi(s)=\mathbb E_\pi[G_t|S_t=s]$$
$$= \mathbb E_\pi\left[\sum_{k=0}^\infty\gamma^kR_{t+k+1}\Bigg|S_t=s\right]$$
$$= \mathbb E_\pi\left[R_{t+1}+\gamma\sum_{k=0}^\infty\gamma^kR_{t+k+2}\Bigg|S_t=s\right]$$
$$= \sum_a \pi(a|s)\sum_{s'}\sum_{r}p(s', r|s, a)\left[r+\gamma\mathbb E_\pi\left[\sum_{k=0}^\infty\gamma^kR_{t+k+2}\Bigg|S_{t+1}=s'\right]\right]$$
$$=\sum_a\pi(a|s)\sum_{s', r}p(s', r|s, a)\left[r+\gamma v_\pi(s')\right]$$
This is known as the ==bellman equation==. Expresses relationship between value of a state and the value of its successor states.Think: Look ahead to possible successors. 

![[Screenshot 2026-02-06 at 8.32.56 AM.png]]

---
## 3.8 Optimal Value Functions

>[!quote] Donald Trump
>The optimal policy, mm yes, the best policy the most amazing policy this nation has, no this world has ever seen. Yes. 

Defined as,
$$v^*(s)=\max_\pi v_\pi(s)\forall s\in\mathcal S$$
Similarly we have,
$$q^*(s, a) = \max_\pi q_\pi(s, a)\forall s\in\mathcal S, a\in\mathcal A$$
We can rewrite this as,
$$q^*(s, a)=\mathbb E\left[R_{t+1}+\gamma v^*(S_{t+1})|S_t=s, A_t=a\right]$$
Because every state is optimal, so v is also optimal.

Because $v^*$ is the value function for a policy, it must satisfy  self-consistency condition by the bellman equation $-$ however, since it is optimal, we can write it's consistency condition without reference to any specific policy. This is the ==Bellman Optimality Equation.==
Intuitively it expresses that value of state under optimal policy MUST equal expected return for best action from that state or $v^*(s)=\max_{a\in\mathcal A(s)} q_{\pi^*}(s, a)$ Using the same steps as we did for the bellman equation, we get,
$$v^*(s)=\max_{a\in\mathcal A(s)}\sum_a\pi(a|s)\sum_{s', r}p(s', r|s, a)\left[r+\gamma v^*(s')\right]$$
Similarly for action-value we get
![[Screenshot 2026-02-06 at 8.56.53 AM.png]]

---
## 3.9 Optimality and Approximation
For large/continuous MDPs you usually can not computer exact optimal value functions or policies. So instead we'll just work with approximations. The idea then becomes which approximations you are willing to work with. 

---