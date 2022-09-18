import numpy as np
import multiprocessing as mp
from time import perf_counter


def solve(f, p):
    m = np.size(p, 1) - 1
    pool = mp.Pool(mp.cpu_count())
    while True:
        choices = pool.starmap(maximise_entropy, ((m, n, p) for n in range(1, m)))
        max = -1
        for c, e in choices:
            if e > max:
                opt_c = c.copy()
                max = e
        c = opt_c
        N = np.nonzero(c == 0)[0]
        L = np.nonzero(c == 1)[0]
        R = np.nonzero(c == 2)[0]
        print(f'L: {L + 1},  R: {R + 1}')
        simulate(f, N, L, R, p, m)
        if np.count_nonzero(p) == 1:
            w, f = np.argwhere(p)[0]
            if f == m:
                return 'All coins are genuine'
            if w == 0:
                return f'Coin {f + 1} is light'
            return f'Coin {f + 1} is heavy'
    

def simulate(f, N, L, R, p, m):
    # Bayes => normalise non-zero probabilities
    # Balanced
    if f[0] - 1 in N or f[0] == 0:
        # Left or Right impossible
        p[0, L] = p[0, R] = p[1, L] = p[1, R] = 0
    # Left low = (left & heavy) or (right & light)
    elif (f[0] - 1 in L and f[1] == 'H') or (f[0] - 1 in R and f[1] == 'L'):
        # Left light, right heavy, neither & allgenuine impossible
        p[0, L] = p[1, R] = p[0, N] = p[1, N] = p[0, m] = 0
    # Left high = (left & light) or (right & heavy)
    elif (f[0] - 1 in L and f[1] == 'L') or (f[0] - 1 in R and f[1] == 'H'):
        # Left heavy, right light, neither & allgenuine impossible
        p[1, L] = p[0, R] = p[0, N] = p[1, N] = p[0, m] = 0
    p /= np.sum(p)



def entropy(c, p):
    if np.count_nonzero(c == 1) != np.count_nonzero(c == 2):
        return 0
    # Remove allgenuine column so can use boolean indexing
    q = p[:,:-1]
    leftLow = np.sum(q[0, c == 2]) + np.sum(q[1, c == 1])
    leftHigh = np.sum(q[0, c == 1]) + np.sum(q[1, c == 2])
    balanced = 1 - leftLow - leftHigh
    return - xlogx(leftLow) - xlogx(leftHigh) - xlogx(balanced)


def xlogx(x):
    if x == 0:
        return 0
    return x * np.log2(x)


def maximise_entropy(m, n, p):
    # 1 <= n < m
    c = np.zeros(m, int)
    start_point = m - n
    c[start_point - 1] = 1
    max = -1
    while True:
        for i in range(start_point, m):
            if c[i] < 2:
                c[i] += 1
                tmp = entropy(c, p)
                if tmp > max:
                    optc = c.copy()
                    max = tmp
                break
            c[i] = 0
        else:
            break
    return optc, max
    

def main():
    m = int(input('Enter # of coins (<=12 is sensible): '))
    f0 = int(input('Enter forged coin number (0 for all genuine): '))
    if f0 > 0:
        f1 = input('Enter H for heavy or L for light: ')
    else:
        f1 = ''
    f = (f0, f1)
    # First row is light, second row is heavy
    # p[0, m] = Prob(all genuine)
    p = np.full((2, m + 1), 1 / (2 * m + 1))
    p[1, m] = 0
    print(solve(f, p))


if __name__ == '__main__':
    t = perf_counter()
    main()
    print(f'done in {perf_counter() - t} seconds')