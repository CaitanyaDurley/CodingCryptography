import numpy as np


def solve(f, p):
    m = np.size(p, 1) - 1
    c = np.zeros(m, int)
    max = -1
    while c is not None:
        tmp = entropy(c, p)
        if tmp > max:
            optc = c.copy()
            max = tmp
        c = increment(c)
    c = optc
    N = np.where(c == 0)[0]
    L = np.where(c == 1)[0]
    R = np.where(c == 2)[0]
    print(f'L: {L + 1},  R: {R + 1}')
    p = simulate(f, N, L, R, p, m)
    if np.count_nonzero(p) == 1:
        w, f = np.argwhere(p)[0]
        if f == m:
            return 'All coins are genuine'
        if w == 0:
            return f'Coin {f + 1} is light'
        return f'Coin {f + 1} is heavy'
    return solve(f, p)
    

def simulate(f, N, L, R, p, m):
    # Bayes => normalise non-zero probabilities
    # Balanced
    if f[0] - 1 in N or f[0] == 0:
        # Left or Right impossible
        p[0, L] = p[0, R] = p[1, L] = p[1, R] = 0
        p /= np.sum(p)
    # Left low = (left & heavy) or (right & light)
    if (f[0] - 1 in L and f[1] == 'H') or (f[0] - 1 in R and f[1] == 'L'):
        # Left light, right heavy, neither & allgenuine impossible
        p[0, L] = p[1, R] = p[0, N] = p[1, N] = p[0, m] = 0
        p /= np.sum(p)
    # Left high = (left & light) or (right & heavy)
    if (f[0] - 1 in L and f[1] == 'L') or (f[0] - 1 in R and f[1] == 'H'):
        # Left heavy, right light, neither & allgenuine impossible
        p[1, L] = p[0, R] = p[0, N] = p[1, N] = p[0, m] = 0
        p /= np.sum(p)
    return p


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


def increment(c):
    for i in range(c.size):
        if c[i] < 2:
            c[i] += 1
            return c
        else:
            c[i] = 0
    return None


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
    main()