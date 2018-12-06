## Clustering of mixed data

### References

- [Kaggle discussion](https://www.kaggle.com/general/19741)
- [stackoverflow discussion](https://datascience.stackexchange.com/questions/22/k-means-clustering-for-mixed-numeric-and-categorical-data)
- k-modes and k-prototype [paper](https://pdfs.semanticscholar.org/d42b/b5ad2d03be6d8fefa63d25d02c0711d19728.pdf) by Zhexue Huang
- Zenghiou He's [paper](https://arxiv.org/ftp/cs/papers/0603/0603120.pdf) on approximation algorithms for k-modes and sensitivity to initial conditions.
- Nico de Vos' [github](https://github.com/nicodv/kmodes) with a python implementation of k-modes and k-prototyping 
- Cao et al.'s [paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.474.8181&rep=rep1&type=pdf) on initialization for categorical data clustering  
- Thomas Filaire using [PAM in R](https://towardsdatascience.com/clustering-on-mixed-type-data-8bbd0a2569c3)

### Introduction

The sample space for categorical data is discrete, and doesn't have a natural origin. A Euclidean distance function on such a space isn't really meaningful.

### Genetic approaches

- Apply a algorithm specific for mixed cases: k-prototyping, partitioning around medoids (PAM).
For categorical data, typically Hamming dissimilarity or Gower distance measure is used. With Hamming dissimilarity, the distance is 1 for each feature that differs (rather than the difference between the numeric values assigned to the categories). Is label encoding or onehotencoding needed for Hamming distance, or can we keep the variables as whatever categories?

- "If your scale your numeric features to the same range as the binarized categorical features then cosine similarity tends to yield very similar results to the Hamming approach above. I don't have a robust way to validate that this works in all cases so when I have mixed cat and num data I always check the clustering on a sample with the simple cosine method I mentioned and the more complicated mix with Hamming. If the difference is insignificant I prefer the simpler method." cwharland, in the linked stackoverflow discussion.

- "If your data contains both numeric and categorical variables, the best way to carry out clustering on the dataset is to create principal components of the dataset and use the principal component scores as input into the clustering.<br>
Remember that u can always get principal components for categorical variables using a multiple correspondence analysis (MCA), which will give principal components, and you can get then do a separate PCA for the numerical variables, and use the combined as input into your clustering.<br>
OR u could use the R package called FactorMineR or PCAmix to carry Factor analysis of mixed data, with the output being principal components, and then using the principal components as input into your clustering." Hycene in the linked Kaggle discussion

- Converting categorical attributes to binary values, and then doing k-means as if these were numeric values. Earlier method, see [Ralambondrainy](https://www.sciencedirect.com/science/article/abs/pii/016786559500075R).


### k-means vs k-medians vs k-modes vs k-prototyping
