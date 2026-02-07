---
tags:
  - ML
  - CS
  - Agents
---
# Part I
For markov property: [[Suton, Barto - Chapter 3; Finite MDPs. The important bits]]

> Current state is independent of all previous states.

Which action is better or worse? We use reward functions for this.

## Rewards
A state where car is driving quickly is high reward, colliding is low reward.

Objective of RL is not just to get high rewards now, but also to get a high reward OVERALL. To get a high Grand total $G_t$.

## Definitions
Markov Chain; $\mathcal M = \{\mathcal S, \mathcal T\}$
Where $\mathcal S$ is state space, states $s\in\mathcal S$ discrete or continuous

$\mathcal T$ is transition operator or dynamics function, $p(s_{t+1}|s)$.

Why operator? --> let $\mu_{t, i}=p(s_{t+1}=i|s_t=j)$

We can write $\vec \mu_t$ then let $\mathcal T_{i, j}=p(s_{t+1}=i|s_t=j)$ Probability going from state j to state i.

So $\boxed{\vec \mu_{t+1}=\mathcal T \vec{\mu_t}}$


Markov Decision Process: [[Suton, Barto - Chapter 3; Finite MDPs. The important bits]]

![[Screenshot 2026-02-07 at 6.04.13 AM.png]]

Partially observed MDP; $\mathcal M=\{\mathcal S, \mathcal A, \mathcal O, \mathcal T, \mathcal E, r\}$

$\mathcal O$ - Observation Space
$\mathcal E$ - emission probability $p(o_t|s_t)$ (chance to see this, given we're here.)

## The goal of reinforcement learning
Get a policy that maximises rewards over trajectory $\tau={s_1, a_1, \dots, s_T, a_T}$.

![[Screenshot 2026-02-07 at 6.19.03 AM.png]]

![[Screenshot 2026-02-07 at 6.18.45 AM.png]]

Have covered this part for finite and infinite horizon in [[Suton, Barto - Chapter 3; Finite MDPs. The important bits]] so I don't want to rewrite all that. 

![[Screenshot 2026-02-07 at 6.24.11 AM.png]]

![[Screenshot 2026-02-07 at 6.24.25 AM.png]]

Just to keep up with the notation.

> Also $\mu=\mathcal T\mu$, when things are stationary, $\mu$ will be same even after the $\mathcal T$ operator. So this just means $\mu$ is the eigenvector of $\mathcal T$ so this means $\mu$ will stay the same... forever...

## Expectations and stochastic systems

RL is all about optimising expectations. We'll talk about a lot of things, but in the end we care about  maxxing the expected value.

> $\mathbb E[f(x)]$ can be continuous even if $f(x)$ is not continuous. This is why RL algorithms can use smooth algos like gradient descent.

![[Screenshot 2026-02-07 at 6.28.49 AM.png]]

![[Screenshot 2026-02-07 at 6.29.51 AM.png]]

(seems like cheating but hey it gets the job done)

# Part II
## The anatomy of a reinforcement learning algorithm
Consists of three (3) basic parts.

Generate samples (run the policy) --> Fit a model/estimate the return ($G_t$) --> improve the policy . --> Generate samples --> $\dots$

### Policy Gradient
That is also the basic high level scheme for policy gradients. 

![[Screenshot 2026-02-07 at 8.13.56 AM.png]]

### Improve the policy
descend dat gradient
![[Screenshot 2026-02-07 at 8.18.26 AM.png]]

# Part III
## How to deal with all this expectation?
$$\mathbb E_{\tau\sim p_\theta(\tau)}\left[\sum_{t=1}^Tr(s_t, a_t)\right]$$
We can write this expectation out recursively, (using chain rule, expectation follows chain rule)
$$\mathbb E_{s_1\sim p(s_1)}\left[r(s_1, a_1) +\ \ \ \ \ \ \ \ \ \ \ \ \ \ \|s_1\right]$$
$$\mathbb E_{s_1\sim p(s_1)}\left[r(s_1, a_1) + \mathbb E_{s_2\sim p(s_2|s_1, a_1)}\left[\mathbb E_{a_2\sim \pi(a_2|s_2)}\left[r(s_2, a_2)+\dots|s_2\right]|s_1, a_1\right] |s_1\right]$$
![[Screenshot 2026-02-07 at 10.31.16 AM.png]]
Let's define some function $Q$
![[Screenshot 2026-02-07 at 10.32.04 AM.png]]

Then we can write our original RL objective as,

![[Screenshot 2026-02-07 at 10.59.07 AM.png]]

So if we find $Q$ it would be *extremely easy* to modify the policy!

This can be expanded to the a more general concept.

## Definition: Q-function
![[Screenshot 2026-02-07 at 11.00.22 AM.png]]

### Definition: Value Function
> Closely related

![[Screenshot 2026-02-07 at 11.00.50 AM.png]]

## Using Q & Value Functions
### Idea 1: If we have a policy $\pi$ and we know $Q^\pi(s, a)$ then we can *improve* $\pi$
![[Screenshot 2026-02-07 at 11.01.50 AM.png]]

### Idea 2: Compute gradient to increase probability of good actions a:
![[Screenshot 2026-02-07 at 11.02.19 AM.png]]

# Part IV

---

