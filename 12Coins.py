import numpy as np


def solve(f, p):
    m = np.size(p, 1)
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
    p = simulate(f, N, L, R, p)
    # input(p)
    if np.count_nonzero(p) == 0:
        return 'All coins are genuine'
    if np.count_nonzero(p) == 1:
        w, f = np.argwhere(p)[0]
        if w == 0:
            return f'Coin {f + 1} is light'
        else:
            return f'Coin {f + 1} is heavy'
    return solve(f, p)
    

def simulate(f, N, L, R, p):
    # Balanced
    if f[0] - 1 in N:
        # Left or Right impossible
        p[0, L] = p[0, R] = p[1, L] = p[1, R] = 0
        # Bayes => normalise non-zero probabilities
        p /= np.sum(p)
    # Left low = (left & heavy) or (right & light)
    if (f[0] - 1 in L and f[1] == 'H') or (f[0] - 1 in R and f[1] == 'L'):
        # Left light, right heavy & neither impossible
        p[0, L] = p[1, R] = p[0, N] = p[1, N] = 0
        # Bayes => normalise non-zero probabilities
        p /= np.sum(p)
    # Left high = (left & light) or (right & heavy)
    if (f[0] - 1 in L and f[1] == 'L') or (f[0] - 1 in R and f[1] == 'H'):
        # Left heavy, right light & neither impossible
        p[1, L] = p[0, R] = p[0, N] = p[1, N] = 0
        # Bayes => normalise non-zero probabilities
        p /= np.sum(p)
    return p


def entropy(c, p):
    if np.count_nonzero(c == 1) != np.count_nonzero(c == 2):
        return 0
    leftLow = np.sum(p[0, c == 2]) + np.sum(p[1, c == 1])
    leftHigh = np.sum(p[0, c == 1]) + np.sum(p[1, c == 2])
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


#First row is light, second row is heavy
m = 12
f = (1, 'H')
print('Forgery:', solve(f, np.full((2, m), 1 / (2*m))))