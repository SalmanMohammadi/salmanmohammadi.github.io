---
title: Reproducing Proximal Policy Optimisation
layout: post
hidden: false
---
### Prelude
I've previously covered [the theory of proximal policy optimisation (PPO) implementations]({%link _content/ppo.md %}) which gives an understanding of the concepts used in modern PPO implementations. This post will cover a practical implementation.

I'll be using `Python 3.7` in this post. I won't cover environment setup here, there's a lot of resources around. I'm assuming familiarity with `torch` and the `gymnasium` APIs.

### Policy and Value networks


