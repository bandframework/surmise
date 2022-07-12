import numpy as np


def emulation_setthetacovfunction(emuinfo, covname):
    if covname == 'exp':
        emuinfo['covthetaf'] = emulation_covmat_exp
    else:
        emuinfo['covthetaname'] = 'matern'
        emuinfo['covthetaf'] = emulation_covmat_maternone
    
    emuinfo['gammathetacovhyp0'], emuinfo['gammathetacovhypLB'], emuinfo['gammathetacovhypUB'], emuinfo['diffTheta'], emuinfo['sameTheta'] = emuinfo['covthetaf'](emuinfo['thetaval'], emuinfo['thetaval'], None, gethyp = True)
    return None

def emulation_setxcovfunction(emuinfo, covname):
    
    if covname == 'exp':
        emuinfo['covxf'] = emulation_covmat_exp
    else:
        emuinfo['covxname'] = 'matern'
        emuinfo['covxf'] = emulation_covmat_maternone
        
    emuinfo['gammaxcovhyp0'], emuinfo['gammaxcovhypLB'], emuinfo['gammaxcovhypUB'], emuinfo['diffX'], emuinfo['sameX'], emuinfo['sameType'] = emuinfo['covxf'](emuinfo['xval'], emuinfo['xval'], None, type1 = emuinfo['typeval'], type2 = emuinfo['typeval'], gethyp = True)
    emuinfo['gammaxcovhyp0'][-1] = 10 ** (0)
    emuinfo['gammaxcovhypLB'][-1] = 10 ** (-4)
    emuinfo['gammaxcovhypUB'][-1] = 10 ** (1)

    return

def emulation_corderivativetester(fname, x1, x2, gammav, type1, type2):
    M0, dM0 = fname(x1, x2, gammav, type1, type2, returndir = True)
    for k in range(0,gammav.shape[0]):
        gammap = 1*gammav[:]
        gammap[k] += 10 ** (-4)
        M1, dM1 = fname(x1, x2, gammap, type1, type2, returndir = True)
        print((M1-M0) * (10 ** 4))
        print(dM0[:,:,k])
        
    return None


def emulation_covmat_maternone(x1, x2, gammav, type1=None, type2=None,
                               returndir=False, gethyp=False, diffX=None, sameX=None, sameType=None):
    if gethyp:
        xrange = np.max(x1,axis=0) - np.min(x1,axis=0)
        gammacovhypls0 = np.log(xrange) - 0
        gammacovhyplsUB = np.log(xrange) + 3
        gammacovhyplsLB = np.log(xrange) - 4
        gammacovhyplf0 = np.array(np.log( 10 ** (-3)))
        gammacovhyplfLB = np.array(np.log( 10 ** (-5)))
        gammacovhyplfUB =  np.array(np.log( 10 ** (1)))
        gammacovhyphf0 = np.array(np.log( 10 ** (-4)))
        gammacovhyphfLB = np.array(np.log( 10 ** (-7)))
        gammacovhyphfUB = np.array(np.log( 10 ** (-1)))
        gammacovhyp0 = np.concatenate((gammacovhypls0,[gammacovhyplf0], [gammacovhyphf0]))
        gammacovhypLB = np.concatenate((gammacovhyplsLB,[gammacovhyplfLB], [gammacovhyphfLB]))
        gammacovhypUB = np.concatenate((gammacovhyplsUB,[gammacovhyplfUB], [gammacovhyphfUB]))
        
        valdiff = np.zeros([x1.shape[0], x2.shape[0],gammacovhypls0.shape[0]])
        valsame = np.ones([x1.shape[0], x2.shape[0]])
        if x1.ndim < 1.5:
            d = x1.shape[0]
            x1 = x1.reshape(1,d)
        else:
            d = x1.shape[1]
        for k in range(0, d):
            valdiff[:,:,k] = np.abs(np.subtract.outer(x1[:,k],x2[:,k]))
            valdiffadj = valdiff[:,:,k]/np.exp(gammacovhyp0[k])
            valsame *= (valdiffadj < (10 ** -12))
        
        if type1 is None:
            return gammacovhyp0, gammacovhypLB, gammacovhypUB, valdiff, valsame
        else:
            typesame = np.equal.outer(type1, type2)
            valsame *= typesame
            return gammacovhyp0, gammacovhypLB, gammacovhypUB, valdiff, valsame, typesame
    else:
        if x1.ndim < 1.5:
            d = x1.shape[0]
            x1 = x1.reshape(1,d)
        else:
            d = x1.shape[1]
    
        if x2.ndim < 1.5:
            x2 = x2.reshape(1,d)
        
        V = np.zeros([x1.shape[0], x2.shape[0]])
        R = np.ones([x1.shape[0], x2.shape[0]])
        if returndir:
            dR = np.zeros([x1.shape[0], x2.shape[0],d+2])
        
        if sameX is None:
            Dc = np.ones([x1.shape[0], x2.shape[0]])
        for k in range(0, d):
            if diffX is None:
                S = np.abs(np.subtract.outer(x1[:,k],x2[:,k])/np.exp(gammav[k]))
            else:
                S = diffX[:,:,k] / np.exp(gammav[k])
            R *= (1 + S)
            V -= S
            if returndir:
                dR[:,:, k] = (S * S) / (1 + S)
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
            
    
        Rt = np.exp(gammav[d]) * Dn + np.exp(gammav[d+1]) * Dc + R #low freq stuff
    
        if returndir:
            for k in range(0, d):
                dR[:,:,k] = R * dR[:,:, k]
            dR[:,:,d] = np.exp(gammav[d]) * Dn#low freq stuff
            dR[:,:,d+1] = np.exp(gammav[d+1]) * Dc#high freq stuff
            return Rt, dR
        else:
            return Rt

def emulation_covmat_exp(x1, x2, gammav, type1 = None, type2 = None, returndir = False, gethyp = False, diffX = None, sameX = None, sameType = None):
    if gethyp:
        xrange = np.max(x1,axis=0) - np.min(x1,axis=0)
        gammacovhypls0 = np.log(xrange) - 0
        gammacovhyplsUB = np.log(xrange) + 3
        gammacovhyplsLB = np.log(xrange) - 4
        gammacovhyplf0 = np.array(np.log( 10 ** (-3)))
        gammacovhyplfLB = np.array(np.log( 10 ** (-5)))
        gammacovhyplfUB =  np.array(np.log( 10 ** (1)))
        gammacovhyphf0 = np.array(np.log( 10 ** (-3)))
        gammacovhyphfLB = np.array(np.log( 10 ** (-7)))
        gammacovhyphfUB = np.array(np.log( 10 ** (-1)))
        gammacovhyp0 = np.concatenate((gammacovhypls0,[gammacovhyplf0], [gammacovhyphf0]))
        gammacovhypLB = np.concatenate((gammacovhyplsLB,[gammacovhyplfLB], [gammacovhyphfLB]))
        gammacovhypUB = np.concatenate((gammacovhyplsUB,[gammacovhyplfUB], [gammacovhyphfUB]))
        
        valdiff = np.zeros([x1.shape[0], x2.shape[0],gammacovhypls0.shape[0]])
        valsame = np.ones([x1.shape[0], x2.shape[0]])
        if x1.ndim < 1.5:
            d = x1.shape[0]
            x1 = x1.reshape(1,d)
        else:
            d = x1.shape[1]
        for k in range(0, d):
            valdiff[:,:,k] = np.abs(np.subtract.outer(x1[:,k],x2[:,k]))
            valdiffadj = valdiff[:,:,k]/np.exp(gammacovhyp0[k])
            valsame *= (valdiffadj < (10 ** -12))
        
        if type1 is None:
            return gammacovhyp0, gammacovhypLB, gammacovhypUB, valdiff, valsame
        else:
            typesame = np.equal.outer(type1, type2)
            valsame *= typesame
            return gammacovhyp0, gammacovhypLB, gammacovhypUB, valdiff, valsame, typesame
    else:
        if x1.ndim < 1.5:
            d = x1.shape[0]
            x1 = x1.reshape(1,d)
        else:
            d = x1.shape[1]
    
        if x2.ndim < 1.5:
            x2 = x2.reshape(1,d)
    
        S = np.zeros([x1.shape[0], x2.shape[0]])
        V = np.zeros([x1.shape[0], x2.shape[0]])
        Dc = np.ones([x1.shape[0], x2.shape[0]])
        for k in range(0, d):
            S = np.abs(np.subtract.outer(x1[:,k],x2[:,k])/np.exp(gammav[k]))
            V += -S
            Dc *= (S < (10 ** -8))
        if type1 is None:
            equaltype = np.equal.outer(type1, type2)
            Dc *= equaltype
            R = equaltype*np.exp(V)
            Dn = equaltype
        else:
            Dn = np.ones([x1.shape[0], x2.shape[0]])
            R = np.exp(V)
    
        Rt = np.exp(gammav[d]) * Dn + np.exp(gammav[d+1]) * Dc + R #low freq stuff
    
        if returndir:
            dR = np.zeros([x1.shape[0], x2.shape[0],d+2])
            for k in range(0, d):
                S = np.abs(np.subtract.outer(x1[:,k],x2[:,k])/np.exp(gammav[k]))
                S2 = np.power(S, 2)
                dR[:,:,k] = R*S
    
            dR[:,:,d] = np.exp(gammav[d]) * Dn#low freq stuff
            dR[:,:,d+1] = np.exp(gammav[d+1]) * Dc#high freq stuff
            return Rt, dR
        else:
            return Rt