# Vadim Litvinov
import collections
from random import choices
from random import gauss
from math import sqrt
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

K = 3
N = 200
N_ITER = 1000
PARAMETERS = [[-1.0, sqrt(2)],
              [4.0, sqrt(3)],
              [9.0, sqrt(1)]]
WEIGHTS = [0.2, 0.3, 0.5]
EPS = 1e-8


def PDF(data, means, variances):
    return 1/(np.sqrt(2 * np.pi * variances) + EPS) * np.exp(-1/2 * (np.square(data - means) / (variances + EPS)))


def getRandomParams(n):
    x = np.random.normal(0, 1, size=(n, n))
    return np.dot(x, x.transpose())


def emGmm(data, k, n_iter):
    weights = np.ones((k, 1)) / k  # shape=(k, 1)
    means = np.random.choice(data, k)[:, np.newaxis]  # shape=(k, 1)
    print('starting point means: ', means)
    #variances = np.random.random_sample(size=k)[:, np.newaxis]  # shape=(k, 1)
    data = np.repeat(data[np.newaxis, :], k, 0)  # shape=(k, n)
    vars = np.expand_dims(np.mean(np.square(data - means), axis=1), -1)
    p_list = []
    for step in range(n_iter):
        p = PDF(data, means, vars)
        b = p * weights
        denom = np.expand_dims(np.sum(b, axis=0), 0) + EPS
        b = b / denom
        p_list.append(np.sum(np.log(np.amax(b, axis=0))))
        #print('average likelihood: ', np.sum(np.amax(b, axis=0)) / (N))
        means_n = np.sum(b * data, axis=1)
        means_d = np.sum(b, axis=1) + EPS
        means = np.expand_dims(means_n / means_d, -1)
        vars = np.sum(b * np.square(data - means), axis=1) / means_d
        vars = np.expand_dims(vars, -1)
        weights = np.expand_dims(np.mean(b, axis=1), -1)
    print('log likelihood: ', np.sum(np.log(np.amax(b, axis=0))))
    #plt.plot(range(n_iter), p_list)
    #plt.show()
    return means, vars, weights


if __name__ == '__main__':
    sampled_gaussians = choices([0, 1, 2], WEIGHTS, k=N)
    print(collections.Counter(sampled_gaussians))
    sampled_gaussians = collections.Counter(sampled_gaussians)
    #sampled_gaussians = list(sampled_gaussians.items())
    samples = np.empty([0, N])
    for i in range(len(sampled_gaussians)):
        #samples.append(gauss(PARAMETERS[sampled_gaussians[i]][0],
        #                PARAMETERS[sampled_gaussians[i]][1]))
        mu_i = PARAMETERS[i][0]
        sigma_i = PARAMETERS[i][1]
        sample_size = sampled_gaussians[i]
        samples = np.concatenate((samples, np.random.normal(mu_i, sigma_i, size=sampled_gaussians[i])), axis=None)
    samples = np.asarray(samples, dtype=np.float32)
    #check weights is properly working
    #print(sampled_gaussians.count(0), sampled_gaussians.count(1), sampled_gaussians.count(2))
    #print(sampled_gaussians)
    print('sampled_gaussians: ', sampled_gaussians)
    print('samples: ', samples)
    #plt.hist(samples, bins=200)
    #plt.show()
    #random_params = initializeParams()

    for i in range(10):
        means, variances, weights = emGmm(samples, K, N_ITER)
        print('means: ', means, 'variences: ', variances, 'weights: ', weights, '\n')
