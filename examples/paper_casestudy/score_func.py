def score_func(x, y, alpha, cal):
    # computes interval score
    alpha = 0.05
    z = sps.norm.ppf(1 - alpha/2)
    pr = cal.predict(x)
    mean_pre = pr.mean() # prediction mean of the average of 1000 random thetas at x
    var_pre = pr.var()  # variance of the mean of 1000 random thetas at x
    lower_bound = mean_pre - z*np.sqrt(var_pre)
    upper_bound = mean_pre + z*np.sqrt(var_pre)

    int_score = -(upper_bound - lower_bound) \
        - (2/alpha)*np.maximum(np.zeros(len(y)), lower_bound - y) \
            -np.maximum(np.zeros(len(y)), y - upper_bound)
    return(np.mean(int_score))

def score_func_emuvar(x, y, alpha, cal, emu):
    alpha = 0.05
    z = sps.norm.ppf(1 - alpha/2)
    cal_theta = cal.theta.rnd(500)
    pre = emu.predict(x=x, theta=cal_theta)
    pre_mean = pre.mean()
    mean_pi = pre_mean.mean(axis=1)
    pre_var = pre.var()
    var_pi = pre_var.mean(axis=1)
    lower_bound = mean_pi - z*np.sqrt(var_pi)
    upper_bound = mean_pi + z*np.sqrt(var_pi)
    
    int_score = -(upper_bound - lower_bound) \
        - (2/alpha)*np.maximum(np.zeros(len(y)), lower_bound - y) \
            -np.maximum(np.zeros(len(y)), y - upper_bound)
    return(np.mean(int_score))