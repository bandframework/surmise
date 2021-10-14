import numpy as np


def rmse(cal):
    """Return the root mean squared error between the data and the mean of the calibrator."""
    ypred = cal.predict().mean()

    error = np.sqrt(np.mean((ypred - cal.y)**2))
    return error


def energyScore(cal):
    """Return the empirical energy score between the data and the predictive distribution. """
    S = 1000
    ydist = cal.predict().rnd(S)  # shape = (num of samples, dimension of x)

    def norm(y1, y2):
        return np.sqrt(np.sum((y1 - y2)**2))

    lin_part = np.array([norm(ydist[i], cal.y) for i in range(S)]).mean()
    G = ydist @ ydist.T
    D = np.diag(G) + np.diag(G).reshape(1000, 1) - 2*G
    quad_part = -1/2 * np.mean(np.sqrt(D))

    score = lin_part + quad_part

    return score


def energyScore_naive(cal):
    """Return the empirical energy score between the data and the predictive distribution. """
    S = 1000
    ydist = cal.predict().rnd(S)  # shape = (num of samples, dimension of x)

    def norm(y1, y2):
        return np.sqrt(np.sum((y1 - y2)**2))

    lin_part = np.array([norm(ydist[i], cal.y) for i in range(S)]).mean()
    quad_part = 0
    for s in range(S):
        quad_part += -1/2 * 1/S * np.array([norm(ydist[i], ydist[s]) for i in range(S)]).mean()

    score = lin_part + quad_part

    return score
