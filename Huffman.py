import numpy as np


def huffman(p):
    if not np.isclose(np.sum(p) , 1) or np.any(p < 0):
        raise ValueError('Invalid probability distribution')
    n = p.size
    if n == 2:
        return ('0', '1')
    i = p.argsort()
    q = np.empty(n - 1)
    q[0] = p[i[0]] + p[i[1]]
    q[1:] = p[i[2:]]
    b = huffman(q)
    x = [None]*n
    x[i[0]] = b[0] + '0'
    x[i[1]] = b[0] + '1'
    for j in range(2, n):
        x[i[j]] = b[j - 1]
    return x


# p = np.array(input('Enter probability distribution: ').removeprefix('(').removesuffix(')').split(','), dtype=float)
p = np.array([0.5,0.25] + [1/248]*62)
x = huffman(p)
print(x)
print(sum(p[i] * len(x[i]) for i in range(p.size)))