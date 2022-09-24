import numpy as np


def setthetacovf(emuinfo, covname):
    if covname == 'exp':
        emuinfo['covthetaf'] = covmat_exp
    else:
        emuinfo['covthetaname'] = 'matern'
        emuinfo['covthetaf'] = covmat_maternone

    emuinfo['gammathetacovhyp0'], emuinfo['gammathetacovhypLB'], \
    emuinfo['gammathetacovhypUB'], emuinfo['diffTheta'], \
    emuinfo['sameTheta'] = emuinfo['covthetaf'](emuinfo['theta'], emuinfo['theta'], None, gethyp=True)
    return None


def setxcovf(emuinfo, covname):
    if covname == 'exp':
        emuinfo['covxf'] = covmat_exp
    else:
        emuinfo['covxname'] = 'matern'
        emuinfo['covxf'] = covmat_maternone

    emuinfo['gammaxcovhyp0'], emuinfo['gammaxcovhypLB'], \
    emuinfo['gammaxcovhypUB'], emuinfo['diffX'], \
    emuinfo['sameX'], emuinfo['sameType'] = \
        emuinfo['covxf'](emuinfo['xval'], emuinfo['xval'], None,
                         type1=emuinfo['xcat'], type2=emuinfo['xcat'], gethyp=True)
    emuinfo['gammaxcovhyp0'][-1] = 10 ** (0)
    emuinfo['gammaxcovhypLB'][-1] = 10 ** (-4)
    emuinfo['gammaxcovhypUB'][-1] = 10 ** (1)

    return


def covmat_exp(x1, x2, gammav, type1=None, type2=None,
               returndir=False, gethyp=False,
               diffX=None, sameX=None, sameType=None):
    if gethyp:
        xrange = np.max(x1, axis=0) - np.min(x1, axis=0)
        hyp0 = np.concatenate((np.log(xrange), [np.log(1e-3)], [np.log(1e-3)]))
        hypLB = np.concatenate((np.log(xrange) - 4, [np.log(1e-5)], [np.log(1e-7)]))
        hypUB = np.concatenate((np.log(xrange) + 3, [np.log(10)], [np.log(1e-1)]))

        valdiff = np.zeros([x1.shape[0], x2.shape[0], xrange.shape[0]])
        valsame = np.ones([x1.shape[0], x2.shape[0]])
        if x1.ndim < 1.5:
            d = x1.shape[0]
            x1 = x1.reshape(1, d)
        else:
            d = x1.shape[1]
        for k in range(0, d):
            valdiff[:, :, k] = np.abs(np.subtract.outer(x1[:, k], x2[:, k]))
            valdiffadj = valdiff[:, :, k] / np.exp(hyp0[k])
            valsame *= (valdiffadj < (10 ** -12))

        if type1 is None:
            return hyp0, hypLB, hypUB, valdiff, valsame
        else:
            typesame = np.equal.outer(type1, type2)
            valsame *= typesame
            return hyp0, hypLB, hypUB, valdiff, valsame, typesame
    else:
        if x1.ndim < 1.5:
            d = x1.shape[0]
            x1 = x1.reshape(1, d)
        else:
            d = x1.shape[1]

        if x2.ndim < 1.5:
            x2 = x2.reshape(1, d)

        S = np.zeros([x1.shape[0], x2.shape[0]])
        V = np.zeros([x1.shape[0], x2.shape[0]])
        Dc = np.ones([x1.shape[0], x2.shape[0]])
        for k in range(0, d):
            S = np.abs(np.subtract.outer(x1[:, k], x2[:, k]) / np.exp(gammav[k]))
            V += -S
            Dc *= (S < (10 ** -8))
        if type1 is None:
            equaltype = np.equal.outer(type1, type2)
            Dc *= equaltype
            R = equaltype * np.exp(V)
            Dn = equaltype
        else:
            Dn = np.ones([x1.shape[0], x2.shape[0]])
            R = np.exp(V)

        Rt = np.exp(gammav[d]) * Dn + np.exp(gammav[d + 1]) * Dc + R  # low freq stuff

        if returndir:
            dR = np.zeros([x1.shape[0], x2.shape[0], d + 2])
            for k in range(0, d):
                S = np.abs(np.subtract.outer(x1[:, k], x2[:, k]) / np.exp(gammav[k]))
                S2 = np.power(S, 2)
                dR[:, :, k] = R * S

            dR[:, :, d] = np.exp(gammav[d]) * Dn  # low freq stuff
            dR[:, :, d + 1] = np.exp(gammav[d + 1]) * Dc  # high freq stuff
            return Rt, dR
        else:
            return Rt


def covmat_maternone(x1, x2, gammav, type1=None, type2=None,
                     returndir=False, gethyp=False,
                     diffX=None, sameX=None, sameType=None):
    if gethyp:
        xrange = np.max(x1, axis=0) - np.min(x1, axis=0)

        hyp0 = np.concatenate((np.log(xrange), [np.log(1e-3)], [np.log(1e-3)]))
        hypLB = np.concatenate((np.log(xrange) - 4, [np.log(1e-5)], [np.log(1e-7)]))
        hypUB = np.concatenate((np.log(xrange) + 3, [np.log(10)], [np.log(1e-1)]))

        valdiff = np.zeros([x1.shape[0], x2.shape[0], xrange.shape[0]])
        valsame = np.ones([x1.shape[0], x2.shape[0]])
        if x1.ndim < 1.5:
            d = x1.shape[0]
            x1 = x1.reshape(1, d)
        else:
            d = x1.shape[1]
        for k in range(0, d):
            valdiff[:, :, k] = np.abs(np.subtract.outer(x1[:, k], x2[:, k]))
            valdiffadj = valdiff[:, :, k] / np.exp(hyp0[k])
            valsame *= (valdiffadj < (10 ** -12))

        if type1 is None:
            return hyp0, hypLB, hypUB, valdiff, valsame
        else:
            typesame = np.equal.outer(type1, type2)
            valsame *= typesame
            return hyp0, hypLB, hypUB, valdiff, valsame, typesame
    else:
        if x1.ndim < 1.5:
            d = x1.shape[0]
            x1 = x1.reshape(1, d)
        else:
            d = x1.shape[1]

        if x2.ndim < 1.5:
            x2 = x2.reshape(1, d)

        V = np.zeros([x1.shape[0], x2.shape[0]])
        R = np.ones([x1.shape[0], x2.shape[0]])
        if returndir:
            dR = np.zeros([x1.shape[0], x2.shape[0], d + 2])

        if sameX is None:
            Dc = np.ones([x1.shape[0], x2.shape[0]])
        for k in range(0, d):
            if diffX is None:
                S = np.abs(np.subtract.outer(x1[:, k], x2[:, k]) / np.exp(gammav[k]))
            else:
                S = diffX[:, :, k] / np.exp(gammav[k])
            R *= (1 + S)
            V -= S
            if returndir:
                dR[:, :, k] = (S * S) / (1 + S)
            if sameX is None:
                Dc *= (S < (10 ** -12))
        if sameX is not None:
            Dc = sameX

        R *= np.exp(V)
        if type1 is None:
            if sameType is None:
                equaltype = np.equal.outer(type1, type2)
                Dc *= equaltype
                R *= equaltype
                Dn = equaltype
            else:
                Dn = sameType
                if sameX is None:
                    Dc *= Dn
        else:
            Dn = np.ones([x1.shape[0], x2.shape[0]])

        Rt = np.exp(gammav[d]) * Dn + np.exp(gammav[d + 1]) * Dc + R  # low freq stuff

        if returndir:
            for k in range(0, d):
                dR[:, :, k] = R * dR[:, :, k]
            dR[:, :, d] = np.exp(gammav[d]) * Dn  # low freq stuff
            dR[:, :, d + 1] = np.exp(gammav[d + 1]) * Dc  # high freq stuff
            return Rt, dR
        else:
            return Rt
