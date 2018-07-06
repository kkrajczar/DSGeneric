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


