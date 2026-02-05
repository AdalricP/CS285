---
tags:
  - ML
  - CS
  - Agents
---
# Part I
$\pi_\theta\rightarrow \text{policy}$, $a\rightarrow \text{action}$, $s\rightarrow\text{state}$

Policy looks at observation $o$ and determines an action $a$.

$\pi_\theta (a_t|o_t)$ --> Chance of picking $a_t$ when $o_t$ observation

$t\rightarrow \text{timestep}$

Write policies as distributions given $s$ rather than $o$, $s$ is state.

Difference between *state* and *observation*.

*observation* is say, a picture, or the entire world at time-step $t$, state $s_t$ is what *produced* that observation. It's the concise & complete description of what's relevant. 
May not be able to always infer $s_t$ given $o_t$, but it is always possible to infer $o_t$ given $s_t$.

![[Screenshot 2026-02-06 at 12.38.56 AM.png]]

The current state is *independent* of previous states. $-$ Markov Property.

Using [this video for a bit of reference as well](https://www.youtube.com/watch?v=VnpRp7ZglfA&t=2331s)

## Imitation Learning

Say we have a model for driving, we have people create training data for observations and actions while driving $\rightarrow$ give this to supervised learning. This is behavioural cloning, we attempt to clone the behaviour of humans. 

ALVINN --> original deep learning implementation system, didn't work super well

![[Screenshot 2026-02-06 at 12.59.28 AM.png]]

![[Screenshot 2026-02-06 at 1.01.02 AM.png]]

This doesn't happen in supervised learning, why? --> I.I.D., in supervised learning, each training point doesn't affect others, but in RL it does!

It still does get the job done somewhat well! --> You get bad turns and all, but shit still gets done.

Here's how the model worked

![[Screenshot 2026-02-06 at 1.04.23 AM.png]]

# Part II
## The distributional shift problem
![[Screenshot 2026-02-06 at 1.07.24 AM.png]]

$p_{\pi_\theta}(o_t)\neq p_{\text{data}}(o_t)$

We train under $p_{\text{data}}$ 
So our training objective is,
$$\max_\theta \mathbb E_{o_t\sim p_\text{data}(o_t)}\left[\log\pi_\theta(a_t|o_t)\right]$$

We test under $p_{\pi_\theta}$ so $p_{\pi_\theta}(o_t)\neq p_{\text{data}}(o_t)$, which means its easy to come up with examples where $\pi_\theta$ fails

So what makes $\pi_\theta(a_t|o_t)$ good or bad?
Well it shouldn't be $\max_\theta \mathbb E_{o_t\sim p_\text{data}(o_t)}\left[\log\pi_\theta(a_t|o_t)\right]$
We need a better measure of 'goodness'

One measure is cost,
$$c(s_t, a_t)=\begin{cases}0\ \text{if}\ a_t=\pi^*(s_t)\\1\ \text{otherwise}\end{cases}$$

Assuming $\pi^*(s_t)$ is deterministic (it can be probabilistic too, let's just go w this for easy notation)

Goal:
$$\min \mathbb E_{s_t~p_{\pi_\theta (s_t)}}\left[c(s_t, a_t)\right]$$

ANALYSIS
assume: $\pi_\theta(a\neq\pi^*(s)|s)\leq \epsilon\forall s\in\mathcal D_{\text{train}}$  ("supervised learning worked")

If the probability of making a mistake is $\leq$ $\epsilon$ then how many mistakes will you make over the trajectory on average? (making a mistake once = making mistakes for the rest of your life, since you're trying to *imitate*) (Upper bound!!)
Well, on first time step you will make at least $\epsilon T$ mistakes, the next step has a probability $(1-\epsilon)$ (since you didn't make a mistake), with $T-1$ time steps and so on,

$\underset{O(\epsilon T^2)}{E\left[\sum_t c(s_t, a_t)\right]}\leq \underbrace{\epsilon T + (1-\epsilon)(\epsilon(T-1)+(1-\epsilon)(\dots))}_{\text{T terms, each}\ O(\epsilon T)}$

Upper bound of $O(\epsilon T^2)$

More generally we can assume $\pi_\theta(a\neq\pi^*(s)|s)\leq \epsilon\ \text{for}s\sim p_\text{train}(s)$ and get
$\mathbb E_{p_\text{train}(s)}[\pi_\theta(a\neq \pi^* (s)|s)]\leq \epsilon$

if $p_{train}(s)\neq p_\theta(s):$
$p_\theta(s_t)=(1-\epsilon)^tp_{train}(s_t)+(1-(1-e)^t))p_{mistake}(s_t)$

so $$|p_\theta(s_t) - p_{train}(s_t)|=(1-(1-\epsilon)^t)|p_{mistake}(s_t)-p_{train}(s_t)|\leq 2(1-(1-\epsilon)^t)\leq 2\epsilon t$$
useful identity: $(1-\epsilon)^t\geq 1-\epsilon t\ \text{for}\ \epsilon\in[0, 1]$

so we get

![[Screenshot 2026-02-06 at 2.07.30 AM.png]]

---

