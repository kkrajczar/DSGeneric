# Probability distributions

- Uiform distribution U(0,b):
  - PDF (probability density function) = 1/(b-a) for x e [a, b], 0 otherwise
  - mean, median = 1/2 (a+b)
  - variance = 1/12 (b-a)^2
  - maximum estimation (~german tank problem)
    - maximum likelihood: m, where m is the sample maximum
    - minimum-variance unbiased estimator (UMVU): i) for continuous sample: ((k+1)/k)m = m + m/k, ii) for discrete distribution: m + (m-k)/k = m + m/k - 1, where k is the sample size
