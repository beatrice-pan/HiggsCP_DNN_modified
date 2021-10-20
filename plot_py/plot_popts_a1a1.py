import sys
import os, errno
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

popts = np.load('../HiggsCP_data/a1a1/popts.npy')
pcovs = np.load('../HiggsCP_data/a1a1/pcovs.npy')

weights = np.load('../HiggsCP_data/a1a1/a1a1_raw.w.npy')

pathOUT = "figures/"

def weight_fun(x, a, b, c):
    return a + b * np.cos(x) + c * np.sin(x)

x_weights = np.array([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]) * np.pi
x_fit = np.linspace(0, 2*np.pi)

#---------------------------------
# ERW
# graphic of the plots requires more work and automatation
# fit error should be printed in the legend

i = 0
filename = "popts_event_1"

chi2 = np.sqrt(np.diag(pcovs[i]))

plt.plot(x_weights, weights[:,i], 'o', label='generated')
plt.plot(x_fit, weight_fun(x_fit, *popts[i]),label='fit')
plt.xlabel('2 phiCP (radians)')
plt.ylabel('weight CP')
plt.legend()

ax = plt.gca()
#ERW
# what is wrong with line below?
#ax.annotate("chi2/Ndof = {%0.3f}\n".format(chi2), xy=(0.7, 0.85), xycoords='axes fraction', fontsize=12)
plt.tight_layout()

if filename:
    try:
        os.makedirs(pathOUT)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    plt.savefig(pathOUT + filename+".eps")
else:
    plt.show()

print('error: ' + str(np.sqrt(np.diag(pcovs[i]))))

plt.clf()


#---------------------------------

i = 10 
plt.plot(x_weights, weights[:,i], 'o')
plt.plot(x_fit, weight_fun(x_fit, *popts[i]))
plt.xlabel('2 phi')
plt.ylabel('w')
print('error: ' + str(np.sqrt(np.diag(pcovs[i]))))

plt.show()
plt.clf()

i = 1000 
plt.plot(x_weights, weights[:,i], 'o')
plt.plot(x_fit, weight_fun(x_fit, *popts[i]))
plt.xlabel('2 phi')
plt.ylabel('w')
print('error: ' + str(np.sqrt(np.diag(pcovs[i]))))

plt.show()
plt.clf()

i = 2000 
plt.plot(x_weights, weights[:,i], 'o')
plt.plot(x_fit, weight_fun(x_fit, *popts[i]))
plt.xlabel('2 phi')
plt.ylabel('w')
print('error: ' + str(np.sqrt(np.diag(pcovs[i]))))

plt.show()
plt.clf()

i = 8000 
plt.plot(x_weights, weights[:,i], 'o')
plt.plot(x_fit, weight_fun(x_fit, *popts[i]))
plt.xlabel('2 phi')
plt.ylabel('w')
print('error: ' + str(np.sqrt(np.diag(pcovs[i]))))

plt.show()
plt.clf()

i = 10000 
plt.plot(x_weights, weights[:,i], 'o')
plt.plot(x_fit, weight_fun(x_fit, *popts[i]))
plt.xlabel('2 phi')
plt.ylabel('w')
print('error: ' + str(np.sqrt(np.diag(pcovs[i]))))

plt.show()
plt.clf()

