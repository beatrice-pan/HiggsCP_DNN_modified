import sys
import os, errno
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

popts = np.load('../HiggsCP_data/rhorho/popts.npy')
pcovs = np.load('../HiggsCP_data/rhorho/pcovs.npy')

weights = np.load('../HiggsCP_data/rhorho/rhorho_raw.w.npy')

pathOUT = "figures/"

def weight_fun(x, a, b, c):
    return a + b * np.cos(x) + c * np.sin(x)

x_weights = np.array([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]) * 3.14
x_fit = np.linspace(0, 6.28)

#---------------------------------
# ERW
# graphic of the plots requires more work and automatation
# fit error should be printed in the legend

i = 0
filename = "popts_rhorho_event_1"

plt.plot(x_weights, weights[:,i], 'o', label="generated")
plt.plot(x_fit, weight_fun(x_fit, *popts[i]), label="function")
plt.ylim([0.0, 2.0])
plt.xlabel(r'$\alpha^{CP}$ [rad]')
plt.ylabel('wt')
#plt.legend('loc=1')
plt.legend()
plt.tight_layout()

if filename:
    try:
        os.makedirs(pathOUT)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    plt.savefig(pathOUT + filename+".eps")
    print('Saved '+pathOUT + filename+".eps")
    plt.savefig(pathOUT + filename+".pdf")
    print('Saved '+pathOUT + filename+".pdf")
else:
    plt.show()

print('error: ' + str(np.sqrt(np.diag(pcovs[i]))))

#---------------------------------
plt.clf()
#---------------------------------

i = 10 
filename = "popts_rhorho_event_10"

plt.plot(x_weights, weights[:,i], 'o')
plt.plot(x_fit, weight_fun(x_fit, *popts[i]))
plt.ylim([0.0, 2.0])
plt.xlabel(r'$\alpha^{CP}$ [rad]')
plt.ylabel('wt')
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
    print('Saved '+pathOUT + filename+".eps")
    plt.savefig(pathOUT + filename+".pdf")
    print('Saved '+pathOUT + filename+".pdf")
else:
    plt.show()

print('error: ' + str(np.sqrt(np.diag(pcovs[i]))))

plt.clf()

#---------------------------------

i = 1000 
filename = "popts_rhorho_event_1000"

plt.plot(x_weights, weights[:,i], 'o', label='generated')
plt.plot(x_fit, weight_fun(x_fit, *popts[i]), label="function")
plt.ylim([0.0, 2.0])
plt.xlabel(r'$\alpha^{CP}$ [rad]')
plt.ylabel('wt')
plt.legend()

ax = plt.gca()
plt.tight_layout()

if filename:
    try:
        os.makedirs(pathOUT)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    plt.savefig(pathOUT + filename+".eps")
    print('Saved '+pathOUT + filename+".eps")
    plt.savefig(pathOUT + filename+".pdf")
    print('Saved '+pathOUT + filename+".pdf")
else:
    plt.show()

print('error: ' + str(np.sqrt(np.diag(pcovs[i]))))

plt.clf()

#---------------------------------

i = 2000 
filename = "popts_rhorho_event_2000"

plt.plot(x_weights, weights[:,i], 'o', label='generated')
plt.plot(x_fit, weight_fun(x_fit, *popts[i]),label='function')
plt.ylim([0.0, 2.0])
plt.xlabel(r'$\alpha^{CP}$ [rad]')
plt.ylabel('wt')
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
    print('Saved '+pathOUT + filename+".eps")
    plt.savefig(pathOUT + filename+".pdf")
    print('Saved '+pathOUT + filename+".pdf")
else:
    plt.show()

print('error: ' + str(np.sqrt(np.diag(pcovs[i]))))
#---------------------------------------------------------------------
plt.clf()
#---------------------------------------------------------------------
