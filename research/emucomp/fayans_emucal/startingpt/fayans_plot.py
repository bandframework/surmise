
# Plot training set failures
orderftrainfail = np.argsort(errvalssimple[thetatopinds].sum(1))
import matplotlib.pyplot as plt
plt.style.use(['science', 'high-vis'])
plt.imshow(~np.isnan(ftrain[orderftrainfail]), aspect='auto', cmap='gray', interpolation='none')
plt.xlabel('observables', fontsize=15)
plt.ylabel('rearranged parameters', fontsize=15)
plt.tight_layout()
plt.savefig('fayans_startingpt_fail.png', dpi=150)