# Markov chain Monte-Carlo
Implementation of stochastic modeling methods for approximating complex probability distributions and solving optimization problems.

## Task 1: Bayesian Inference (Metropolis-Hastings)
* **Description:** Implements the Metropolis-Hastings algorithm to sample from the posterior distribution $P(\theta|y)$ of a simple Bayesian model (Normal likelihood, Normal prior). It also analyzes the MCMC chain's behavior for different proposal variances (`d`).

## Task 2: Ising Model

* **Description:** Simulates the 2D Ising model using the Metropolis algorithm. This model demonstrates emergent behavior and phase transitions in spin particles on a lattice based on the inverse temperature parameter ($\beta$).

## Task 3: Discrete Distribution Sampling (Metropolis-Hastings)

* **Description:** Uses MCMC with an asymmetric random walk proposal to sample from a discrete, heavy-tailed target distribution ($\Pi_i \propto i^{-3/2}$).

## Task 4: Beta Distribution (Direct Monte Carlo)

* **Description:** Implements a direct Monte Carlo simulation to generate samples from a Beta(a, b) distribution by leveraging its fundamental relationship with the Gamma distribution ($Z = X / (X + Y)$).

## Task 5: MCMC Cryptography (Simulated Annealing)

* **Description:** Decrypts a simple substitution cipher using a Metropolis-Hastings algorithm combined with **Simulated Annealing**. The algorithm "scores" potential decryptions against a 2-gram (bigram) statistical model of the Ukrainian language.
