import ncephes
import numpy as np
from matplotlib import pyplot as plt

"""
Code employed tu graph the required number of packets to attain certain confidence over results (depending on the true
underlying PRR)
"""


def pos_interval(m, n, c):
    """
    Computes the positive interval for the Bernoulli distribution according to
    https://arxiv.org/pdf/1105.1486.pdf
    :param m: the positive cases (number of received packets)
    :param n: total cases (number of packets sent)
    :param c: c * 100% = confidence interval
    :return: With c*100% confidence interval, the mean will be smaller than the returned value
    """
    return ncephes.cprob.incbi(m + 1, n - m + 1, 0.5 * (1 + c))


def neg_interval(m, n, c):
    """
    Computes the negative interval for the Bernoulli distribution according to
    https://arxiv.org/pdf/1105.1486.pdf
    :param m: the positive cases (number of received packets)
    :param n: total cases (number of packets sent)
    :param c: c * 100% = confidence interval
    :return: With c*100% confidence interval, the mean will be larger than the returned value
    """
    return ncephes.cprob.incbi(m + 1, n - m + 1, 0.5 * (1 - c))


def diff_interval(m, n, c):
    """
    Computes the mean uncertainty in the estimation of the PRR
    :param m: the positive cases (number of received packets)
    :param n: total cases (number of packets sent)
    :param c: c * 100% = confidence interval
    :return: With c*100% confidence interval, the difference between the upper and the lower bounds
    """
    return pos_interval(m, n, c) - neg_interval(m, n, c)


c = 0.9     # confidence interval
d = 0.15    # maximum allowed difference between upper and lower bound
num_experiments = 1000  # 1000 experiments per point of PRR
result = {}
probs = np.linspace(0, 1, 100)  # divide the PRR (from 0 to 1) in 100 points

for p in probs:
    required = []
    for _ in range(num_experiments):
        tosses = np.random.uniform(size=(10000,)) > p
        for n in range(10000):  # will send up to 10k packets at most
            pos = tosses[:n].tolist().count(True)
            diff = diff_interval(pos, n, c)
            if diff <= d:
                required.append(n)
                break

    result[p] = (np.mean(required), np.std(required))
    print('Done', p, result[p])

print(result)

means = [result[v][0] for v in probs]
stds = [result[v][1] for v in probs]
plt.plot(probs, means, linewidth=2)
plt.xlabel('True PRR', fontsize=13)
plt.ylabel('Number of probe packets to be sent', fontsize=13)
plt.title('Number of packets to be sent vs true underlying PRR', fontsize=14)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='minor', labelsize=12)
plt.savefig('packets_vs_prr_low.png', dpi=100)
plt.savefig('packets_vs_prr_high.png', dpi=300)
plt.show()
