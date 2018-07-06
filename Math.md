# Probability distributions

### Uniform distribution

A distribution that has constant probability.
- pdf (probability density function) = 1/(b-a) for x e \[a,b\], 0 otherwise
- mean, median = 1/2 (a+b)
- variance = 1/12 (b-a)^2
- maximum estimation (~german tank problem)
  - maximum likelihood: m, where m is the sample maximum
  - minimum-variance unbiased estimator (UMVU): i) for continuous sample: ((k+1)/k)m = m + m/k, ii) for discrete distribution: m + (m-k)/k = m + m/k - 1, where k is the sample size
    
### Binomial distribution

The discrete probability distribution of the number of successes in a sequence of n independent experiments, each asking a yes/no question, and each with its own boolean-valued outcome.
  - pmf (probability mass function) = (n k) p^k (1-p)^(n-k)
  - mean, median = np
  - variance: np(1-p)
   
### Bernoulli distribution

The probability distribution of a random variable which takes the value 1 with probability p and the value 0 with probability q = 1 - p - i.e., the probability distribution of any single experiment that asks a yes/no question
  - It can be used to represent a coin toss where 1 and 0 would represent "head" and "tail" (or vice versa), respectively
  - the Bernoulli distribution is a special case of the Binomial distribution where a single experiment/trial is conducted (n=1)
  - pmf = q = (1-p) for k = 0, and p for k = 1
  - mean = p
  - variance = p(1-p) = pq

### Negative binomial distribution

It is a discrete probability distribution of the number of successes in a sequence of independent and identically distributed Bernoulli trials before a specified (non-random) number of failures (denoted r) occurs. For example, if we define a 1 as failure, all non-1s as successes, and we throw a dice repeatedly until the third time 1 appears (r = three failures), then the probability distribution of the number of non-1s that had appeared will be a negative binomial.
  - pmf: (k+r-1 k) (1-p)^r p^k
  - mean: pr/(1-p)
  - variance: pr/(1-p)^2

### Multinomial distribution

The generalization of the binomial distribution. For example, it models the probability of counts for rolling a k-sided die n times. For n independent trials each of which leads to a success for exactly one of k categories, with each category having a given fixed success probability, the multinomial distribution gives the probability of any particular combination of numbers of successes for the various categories.
  - When n is 1 and k is 2, the multinomial distribution is the Bernoulli distribution. When k is 2 and number of trials are more than 1, it is the binomial distribution. When n is 1, it is the categorical distribution.

### Categorical distribution

Generalization of the Bernoulli distribution for a categorical random variable, i.e. for a discrete variable with more than two possible outcomes, such as the roll of a die.

### Poisson distribution

The probability of a given number of events occurring in a fixed interval of time or space if these events occur with a known constant rate and independently of the time since the last event. FOr instance, the number of patients arriving in an emergency room between 10 and 11 pm is Poisson distributed.
  - pmf: p(k)= lambda^k exp(-lambda) / k!
  - The number of patients arriving in an emergency room between 10 and 11 pm is Poisson
  - mean = lambda
  - variance = lambda

### Chi-squared distribution

If Z follows a N(0,1) standard distribution, then Q_1 = Z^2 follows a chi-squared distribution
  - Z can follow N(0,1) as a result of the central limit theorem
  - If we have two variables connected as Y_1 = n - Y_2 (like males and females), then Q_1 = Summa_i=1^2 (Y_i - np_i)^2 / np_i = Summa_i=1^2 (observed - expected)^2 / expected and Q_1 follows a chi-squared distribution with one degree of freedom.
  - The example I had is for Y_1 and Y_2 being binomial sample counts (number of males and females)
  - The expected values might be coming from any kind of distribution, a poisson, binomial, etc.
  - Extension for k categories: Q_1 = Summa_i=1^k (Y_i - np_i)^2 / np_i follows a chi-squared distribution with k-1 degrees of freedom.
  - Can also be used to test distributions with unspecified probabilities. We can keep p as a parameter, and we can minimize chi2 to determine p, however in this case we need to further reduce the degrees of freedom by 1. p would become then the minimum chi-squared estimator.
  - In general, the degrees of freedom is reduced by d if we need to estimate d parameters.
  - If finding minimum is difficult, we can i) estimate the s parameters using the maximum likelihood method, ii) calculate Q_k-1 using the obtained estimates, iii) compare the chi-squared statistics to a chi-squared distribution with (k-1)-d degrees of freedom.

### Hypergeometric distribution

It is a discrete probability distribution that describes the probability of k successes (random draws for which the object drawn has a specified feature) in n draws, without replacement, from a finite population of size N that contains exactly K objects with that feature, wherein each draw is either a success or a failure. In contrast, the binomial distribution describes the probability of k successes in n draws with replacement.
  - the hypergeometric test uses the hypergeometric distribution to calculate the statistical significance of having drawn a specific k successes (out of n total draws) from the aforementioned population. The test is often used to identify which sub-populations are over- or under-represented in a sample. This test has a wide range of applications. For example, a marketing group could use the test to understand their customer base by testing a set of know n customers for over-representation of various demographic subgroups (e.g., women, people under 30)
  - pmf: (K k)(N-K n-k)/(N n)
  - mean = nK/N
  - variance = nK/N (N-K)/N (N-n)/(N-1)

### Negative hypergeometric distribution

It describes probabilities for when sampling from a finite population without replacement in which each sample can be classified into two mutually exclusive categories like Pass/Fail, Male/Female or Employed/Unemployed. As random selections are made from the population, each subsequent draw decreases the population causing the probability of success to change with each draw. Unlike the standard hypergeometric distribution, which describes the number of successes in a fixed sample size, in the negative hypergeometric distribution, samples are drawn until r failures have been found, and the distribution describes the probability of finding k successes in such a sample. In other words, the negative hypergeometric distribution describes the likelihood of k successes in a sample with exactly r failures.

# Definitions and Theorems

- Independence of two random variables: P(A \bigcup B) = P(A)P(B) (joint probabilities equal to the product of the probabilities)
  - joint probability distribution: given at least two random variables X, Y, ..., that are defined on a probability space, the joint probability distribution for X, Y, ... is a probability distribution that gives the probability that each of X, Y, ... falls in any particular range or discrete set of values specified for that variable. In the case of only two random variables, this is called a bivariate distribution, but the concept generalizes to any number of random variables, giving a multivariate distribution. Can be used to find two other types of distributions:
    - the marginal distribution giving the probabilities for any one of the variables with no reference to any specific range of values for the other variables
    - the conditional probability distribution giving the probabilities for any subset of the variables conditional on particular values of the remaining variables. In case of independence: P(A, B) = P(A)P(B). This is useful to show dependence of variables!
  - The multivariate normal distribution, which is a continuous distribution, is the most commonly encountered distribution in statistics. When there are specifically two random variables, this is the bivariate normal distribution.
  - In general, random variables may be uncorrelated but statistically dependent. But if a random vector has a multivariate normal distribution then any two or more of its components that are uncorrelated are independent.

- Expected value: Sum_i=0^n p_i*x_i
- Variance: Var(X) = E\[(X-mu)^2\] = Cov(X,X) = E\[X^2\] - E\[X\]^2
  - Var(X) = Sum_i=0^n p_i*(x_i - mu)^2
  - Var(X) = 1/n Sum_i=0^n (x_i - mu)^2 , if p_i is independent of i (equally likely outcomes)
  - There are biased and unbiased sample variances (the 1/(n-1) factor instead of 1/n for normal distr.)
- Covariance:
  - cov(X,Y) = E\[(X-E\[X\])(Y-E\[Y\])\] = E\[XY\] - E\[X\]E\[Y\]
  - cov(X,Y) = 1/n Summa_i=1^n (x_i- E\[X\])(y_i-E\[Y\]) = 1/n^2 Summa_i=1^n Summa_j=1^n 1/2 (x_i-x_j)(y_i-y_j)
  - cov(X,Y) = Summa_(x,y)eS f(x,y)(x-mu_x)(y-mu_y), where f(x,y) is the joint probability mass function
  - For independent variables, cov(X,Y) = 0 and thus corr(X,Y) = 0, but cov(X,Y) = 0 or corr(X,Y) does not mean independence!
- Autocovariance: the autocovariance is a function that gives the covariance of the process with itself at pairs of time points:
  - C_XX(t,s) = cov(X_t, X_s) = E\[(X_t - mu_t)(X_s - mu_s)\] = E\[X_tX_s\] - mu_tmu_s
- Correlation:
  - rho_X,Y = corr(X,Y)=cov(X,Y)/sigma_Xsigma_Y = E\[(X-mu_X)(Y-mu_Y)\]/sigma_Xsigma_Y
  - correlation coefficient detects only linear dependencies between two variables
  - if the variables are independent -> corr = 0; the converse is not true!
  - in the special case when X and Y are jointly normal, uncorrelatedness is equivalent to independence
- Autocorrelation: also known as serial correlation, is the correlation of a signal with a delayed copy of itselfas a function of delay. Informally, it is the similarity between observations as a function of the time lag between them.
  - R(s, t) = E\[(X_t - mu_t)(X_s - mu_s)\] / sigma_tsigma_s
- Confidence interval : a confidence interval (CI) is a type of interval estimate (of a population parameter) that is computed from the observed data. The confidence level is the frequency (i.e., the proportion) of possible confidence intervals that contain the true value of their corresponding parameter. In other words, if confidence intervals are constructed using a given confidence level in an infinite number of independent experiments, the proportion of those intervals that contain the true value of the parameter will match the confidence level. Confidence intervals consist of a range of values (interval) that act as good estimates of the unknown population parameter.

### The Central Limit Theorem

In most situations, when independent random variables are added, their properly normalized sum tends toward a normal distribution even if the original variables themselves are not normally distributed. For example, suppose that a sample is obtained containing a large number of observations, each observation being randomly generated in a way that does not depend on the values of the other observations, and that the arithmetic average of the observed values is computed. If this procedure is performed many times, the central limit theorem says that the computed values of the average will be distributed according to a normal distribution. A simple example of this is that if one flips a coin many times the probability of getting a given number of heads in a series of flips will approach a normal curve, with mean equal to half the total number of flips in each series. The central limit theorem has a number of variants. In its common form, the random variables must be identically distributed. In variants, convergence of the mean to the normal distribution also occurs for non-identical distributions or for non-independent observations, given that they comply with certain conditions. In more general usage, a central limit theorem is any of a set of weak-convergence theorems in probability theory. They all express the fact that a sum of many independent and identically distributed (i.i.d.) random variables, or alternatively, random variables with specific types of dependence, will tend to be distributed according to one of a small set of attractor distributions. The central limit theorem also tells us how fast the convergence of the sample mean to the population mean is: the squared error will typically be about Var\[Y\]/n

### Box-Cox transformation

It transforms non-normal dependent variable into a normal shape using a lambda exponent, which is between -5 and 5:
  - y(lambda) = (y^lambda - 1)/lambda, if lambda != 0
  - y(lambda) = log(y), if lambda = 0


# Trivia

- Combination and permutation:
  - When the order doesn't matter, it is a Combination.
  - When the order does matter it is a Permutation.

