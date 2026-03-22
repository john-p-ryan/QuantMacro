# Computational methods for two asset models


## Model setup
Take the simple dynamic programming problem with a liquid asset $b$ and illiquid asset $k$:

$$
V(k, b, z) = \max_{c, k', b'} \quad u(c) + \beta \mathbb E_z [V(k', b', z')]
$$
subject to 

$$
\begin{align*}
c + b' + k' &= R_k k + R_b b + z - g(k, k')\\
b' &\ge \underbar b \\
k' &\ge \underbar k
\end{align*}
$$

Where $g(k, k')$ represents adjustment costs for the illiquid asset, and we assume that the return on the illiquid asset is greater than the return on the liquid asset: $R_k > R_b$. The period utility function $u$ is standard (concave, increasing, inada conditions, for example, CRRA). We also assume that not adjusting the illiquid asset is costless: $g(k, k) = 0$.

We also assume $z$ follows a finite state markov process: $z' \sim \Gamma(z, z')$. For any computational solution method, we have to discretize the state space, so that we assume an exogenous grid for each of the assets $b \in B_{grid}$ and $k \in K_{grid}$.

Also, any solution method will consist of solving the Bellman equation above to update the value functions or policy functions. The algorithm for doing so is standard: $V_{n+1} = \mathcal T V_n$, where $\mathcal T$ is the Bellman operator.

## Simplest method: double grid search

The simplest solution method for the bellman equation, which also happens to be the slowest and the most stable, is a double grid search. The logic is, for each state $(k, b, z)$, we loop through the grids $k' \in K_{grid}$ and $b' \in B_{grid}$. We evaluate the objective function for the bellman equation for each choice of $(k', b')$ and store the highest value. These are the policy functions, and the objective function evaluated at the optimum is the updated value function. This is the benchmark, as it is stable and well behaved. However, it is incredibly slow as a single update of the value function requires $N_k^2 N_b^2 N_z$ evaluations of the Bellman objective. 

## Nested EGM

Note that we do not assume in general that the adjustment costs $g(k, k')$ are smooth. Thus, we cannot, in-general, obtain an Euler equation in $k$. But $b$ has no adjustment costs, and we should therefore be able to obtain an Euler equation in $b$ when the respective borrowing constraint does not bind. 

First, define an intermediate problem for an agent who is free to adjust $b$, but is forced to adjust their $k$ to a value of $k'$:

$$ \tilde V(k, b, z; k') = \max_{c, b'} u(c) + \beta \mathbb E_z [V(k', b', z')] $$
subject to
$$ \begin{align*}
c + b' + k' &= R_k k + R_b b + z - g(k, k')\\
b' &\ge \underbar b
\end{align*}$$


The Euler equation in $b$ associated with this problem is:

$$ u'(c) \ge \beta R_b \mathbb{E} [u'(c')] $$

Which holds in complementary slackness with the $b' \ge \underbar{b}$. We can utilize this to use an inner Endogenous-Grid Method. That is, 

1. for each $(k, k')$, we assume that $b' \in B_{grid}$, and use the consumption policy function tomorrow to find the $c$ today by inverting the Euler equation above:

$$ c = (u')^{-1} \beta R_b \mathbb{E} [u'(c'(k', b', z'))] $$

2. Then, we use the budget constraint to back out $b$ today:

$$ b = \frac{1}{R_b} \left[ c + k' + b' - R_k k - z + g(k, k')\right]$$

3. We then interpolate across $(b, c)$ and $(b, b')$ to get policy functions today. We are now back on the exogenous grid $b \in B_{grid}$ for the state space.

4. If the implied $b' < \underbar b$ at any point in the state space $b \in B_{grid}$, then we sweep through again and solve for $c$ from the budget constraint, assuming that the borrowing constraint binds there: 

$$ c = R_k k + R_b b + z - g(k, k') - \underbar b - k'$$

5. We now check the maximum value of $k' \in K_{grid}$ for each $k$, taking into account the decision rule from the EGM step:
$$ V(k, b, z) = \max_{k'} \tilde V(k, b, z; k')$$

We can now obtain updated policy functions and value functions using these decision rules. This effectively reduces the number of values to loop through by a factor of $N_b$. 

## Graves' two-step

We can do even better. Graves' method is a further improvement on the solution method which exploits the intuition that a person could pay the adjustment cost in $b$ to move to a different state space. It's a clever recognition that some agents' problems are equivalent. First, define the No-Adjust problem:

$$V_{NA} (k, b, z) = \max_{c, b'} u(c) + \beta \mathbb{E}_z [V(k, b', z')]$$
subject to
$$ \begin{align*}
c + b' &= (R_k - 1) k + R_b b + z\\
b' &\ge \underbar b
\end{align*} $$

where the adjustment costs disappear due to the assumption that $g(k, k) = 0$. 

The advantage of this method comes from the clever realization (rooted in the logic above) that 

$$\tilde V(k, b, z; k') = V_{NA}(k, b^*, z),$$

Where $b^* = b + \frac{R_k}{R_b}(k-k') - \frac{g(k, k')}{R_b}$.

So Graves' method is:

1. Solve the No-Adjust problem above. Note that this CAN be solved using the endogenous grid method. Also note that this has to be solved on a wider grid (including points $b < \underbar b$) because $b^*$ is not guaranteed to be bounded onto the feasible range. We also have to be careful about our interpolator here. We want to use something that is monotone and performs decently for a modest extrapolation, like PCHIP for example. From this, we get $V_{NA}$.
2. Obtain $\tilde V$ from the above relationship $\tilde V(k, b, z; k') = V_{NA}(k, b^*, z)$. 
3. Obtain $V$ from $\tilde V$ by getting the maximum value: $V(k, b, z) = \max_{k'} \tilde V(k, b, z; k')$. Get the policy functions from the argmax.