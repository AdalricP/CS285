---
tags:
  - ML
  - CS
  - Agents
---
> Pretty chill, intro course, kinda tired today but I'm starting now so it's just a good recap. You are likely to know most of what is here already.

---

# Part I

Let's say you wanna build a system to get a robot to pick up objects. 

Problem? See objects through camera --> Output Co-ordinates $(x, y, z)$.

Can't just localize where objects are in the picture and output position, much more complex problem than that --> Lot's of special cases and exceptions. Very hard. Eg. Weird shape object, weird center of mass, soft/deformable, etc.

Too many special cases --> Appealing to use ML.

Could just use a supervised learning algorithm for this. Problem!! Supervised learning needs data, need a LOT of data, very hard!! Even people can't solve this problem very easily through data, human intuition doesn't give good data for robo fingers.

What if we just run the robots to run trials??

This is *Reinforcement learning.*

Do --> Collect Data --> Improve policy

Recent advancements in AI? LLMs!! Drawings!! biology stuffs

What are you learning? --> The distribution of the data, where the data comes from is very important, if data comes from say, a bunch of images from the web, the kind of pictures on the web, the way people type on keyboards?

> move 37, interesting --> Weird move that humans wouldn't have made, but actually good move. Emergent behaviour.

# Part II
![[l1p2img1.png]]

## What is RL?
Mathematical formalism for learning-based decision making

Approach for learning decision making & Control from experience

## How is RL different?

Standard --> Supervised

Given $\mathcal D = \set{(x_i, y_i)}$
Learn $f(x)\approx y$

Assumptions: 
Independent, Identically Distributed Data ($x_1, x_2$ are independent, $f(x)$ is same for all $x_i$) (obviously)
Known ground truth outputs in training (all x has a y, everything is labeled)

Interacts with environment to learn
![[l1p2img2.png]]

Approximates function from underlying data to learn
![[l1p2img3.png]]

# Part III

*The bitter lesson*

![[l1p3img1.png]]

If we want very powerful learning machines, we should build machines that use data well and scale arbitrarily.

Build scalable machines, not how the problem should be solved.

(doesn't mean shove data into GPU, it says *LEARNING* and *SEARCH* not *LEARNING* and *GPU*)

What's Learning? --> Use data to extract patterns.
What's search? --> Use computation to extract inferences. (Use what you got to reach interesting/meaningful conclusions)

Learning allows you to understand the world, search allows you to get interesting, emergent behaviour.

Basic RL deals with maximising rewards. But there are other methods as well: 
- Learning reward functions from examples (inverse reinforcement learning)
- Transferring knowledge between domains (transfer learning, meta-learning)
- Learning to predict and using prediction to act --> Very useful for robobots

## How do we build intelligent machines?

Where do you even start? - Learning as the basis of intelligence? -> Some things we can all do (Eg. Walking --> Evo algo over time)
Some things we can only learn --> Eg driving a car

$\therefore$ our learning mechanisms are likely powerful enough to do everything we associate with intelligence

## A single algorithm?
--> Radical statement; Single flexible algo that handles all our learning
--> Biological evidence for seeing through your tongue

---

