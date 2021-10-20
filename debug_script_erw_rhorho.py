from rootpy.plotting import Hist2D
from rootpy.io import root_open
import numpy as np
from particle import Particle
import matplotlib.pyplot as plt
import warnings
import sys
warnings.filterwarnings('ignore')

DIRECTORY = "./debug_plots/rhorho/"

def get_tau1(p):
	p_tau1_nu = p[0]
	l_tau1_pi = p[1:3]                              # pi-, pi0

	l_tau1_rho = [None]
	l_tau1_rho[0] = l_tau1_pi[0] + l_tau1_pi[1]     # pi- + pi0

	return p_tau1_nu, l_tau1_pi, l_tau1_rho


def get_tau2(p):
	p_tau2_nu = p[3]
	l_tau2_pi = p[4:6]                              # pi+, pi0

	l_tau2_rho = [None]
	l_tau2_rho[0] = l_tau2_pi[0] + l_tau2_pi[1]     # pi+ + pi0

	return p_tau2_nu, l_tau2_pi, l_tau2_rho


def boost_and_rotate(particle, phi, theta, ref_part):
	particle = particle.boost(ref_part)
	particle = particle.rotate_xy(-phi)
	particle = particle.rotate_xz(np.pi - theta)
	return particle

def rotate_xy(x, y, phi):
	cos_phi = np.cos(phi)
	sin_phi = np.sin(phi)
	return (cos_phi*x - sin_phi*y, sin_phi*x + cos_phi*y)

def calc_angles(part, ref_part):
	b_part = part.boost(ref_part)
	phi = b_part.angle_phi
	b_part = b_part.rotate_xy(-phi)
	theta = b_part.angle_theta 
	return phi, theta

def smear_exp(x, lambd):
	noise = np.random.exponential(scale=lambd, size=x.shape)
	sign = np.random.choice(np.array([-1,1]), size=x.shape)
	xsmear = x + sign*noise
	return xsmear


def approx_alpha_A(v_ETmiss_x, v_ETmiss_y, p_tau1_h, p_tau2_h):
	c1 = v_ETmiss_x*p_tau1_h.y
	c2 = v_ETmiss_y*p_tau1_h.x
	d1 = p_tau1_h.x*p_tau2_h.y
	d2 = p_tau1_h.y*p_tau2_h.x
	alpha2 = (-c1 + c2) / (d1 - d2)

	c1 = v_ETmiss_x
	c2 = alpha2*p_tau2_h.x
	d1 = p_tau1_h.x
	alpha1 = (c1 - c2) / d1
	return alpha1, alpha2

def approx_alpha_B(v_ETmiss_x, v_ETmiss_y, p_tau1_h, p_tau2_h):
	M_H = 125.0
	M_TAU = 1.77

	c1 = v_ETmiss_x*p_tau1_h.y
	c2 = v_ETmiss_y*p_tau1_h.x
	d1 = p_tau1_h.x*p_tau2_h.y
	d2 = p_tau1_h.y*p_tau2_h.x
	alpha2 = (-c1 + c2) / (d1 - d2)

	c1 = M_H**2 / 2
	c2 = M_TAU**2
	d1 = ([-1,-1,-1, 1]*p_tau1_h*p_tau2_h).sum(1)
	d2 = 1 + alpha2
	alpha1 = (c1 - c2) / d1
	alpha1 = (alpha1 / d2) - 1
	return alpha1, alpha2

def approx_alpha_C(v_ETmiss_x, v_ETmiss_y, p_tau1_h, p_tau2_h):
	M_H = 125.0
	M_TAU = 1.77
	
	c1 = v_ETmiss_x*p_tau2_h.y
	c2 = v_ETmiss_y*p_tau2_h.x
	d1 = p_tau2_h.x*p_tau1_h.y
	d2 = p_tau2_h.y*p_tau1_h.x
	alpha1 = (-c1 + c2) / (d1 - d2)

	c1 = M_H**2 / 2
	c2 = M_TAU**2
	d1 = ([-1,-1,-1, 1]*p_tau1_h*p_tau2_h).sum(1)
	d2 = 1 + alpha1
	alpha2 = (c1 - c2) / d1
	alpha2 = (alpha2 / d2) - 1
	return alpha1, alpha2


def approx_E_nu(p_tau_h, v_tau_nu_z):
	M_H = 125.0
	M_TAU = 1.77
	
	c1 = M_TAU**2 - np.square(p_tau_h.e) + np.square(p_tau_h.z)
	c2 = 2*v_tau_nu_z*p_tau_h.z
	d = 2*p_tau_h.e
	v_tau_nu_E = (c1 + c2) / d
	return v_tau_nu_E

def plot(x, y, sizes=None, marksize=1, filename=None, xlabel=None, ylabel=None):
	plt.plot(x, y, 'k.', ms=marksize)
	ax = plt.gca()
	if xlabel:
		ax.set_xlabel(xlabel, fontsize=12)
		ax.xaxis.set_label_coords(0.92,-0.06)
	if ylabel:
		ax.set_ylabel(ylabel, fontsize=12)
		ax.yaxis.set_label_coords(-0.06,0.86)
	if sizes:
		plt.axis(sizes)
	plt.tight_layout()
	if filename:
		plt.savefig(DIRECTORY + filename+".png")
		b = Hist2D(5000, sizes[0], sizes[1], 5000, sizes[2], sizes[3], name=filename, title=filename)
		b.fill_array(np.stack((x, y), axis=-1))
		b.GetXaxis().SetTitle(xlabel)
		b.GetYaxis().SetTitle(ylabel)
		with root_open(DIRECTORY +filename+".root", 'w') as f:
			b.Write()
			b.Draw()
				


	else:
		plt.show()
	plt.clf()

def error_plot(exact, approx, sizes=None, step=0.05, filename=None, title=None, barrier=1, relflag=True, absflag=True):
	err = exact - approx
	if relflag:
		err /= exact
	if absflag:
		err = np.abs(err)
	err[err>barrier] = barrier+step
	err[err<-barrier] = -(barrier+step)
	bins = int(1/step) + 1

	plt.hist(err, bins)
	ax = plt.gca()
	if sizes:
		plt.axis(sizes)
	if title:
		ax.annotate(title, xy=(0.6, 0.9), xycoords='axes fraction', fontsize=18)
	#ax.set_xlabel(r'$\|\delta\|$')
	ax.xaxis.set_label_coords(1.05, -1.05)
	#ax.set_ylabel(r'$n_{events}$')
	#ax.yaxis.set_label_coords(-0.12,0.9)
	plt.tight_layout()

	if filename:
		plt.savefig(DIRECTORY + filename+".eps")
		
	else:
		plt.show()
	plt.clf()

def smear_plot(exact, approx, sizes=None, step=0.05, filename=None, title=None, Xlabel=None, Ylabel=None, relflag=True, absflag=True):
	delta = exact - approx
	if relflag:
		delta /= exact
	if absflag:
		delta = np.abs(delta)
	bins = int(1/step) + 1

	plt.hist(delta, bins)
	ax = plt.gca()
	if title:
		ax.annotate(title, xy=(0.6, 0.9), xycoords='axes fraction', fontsize=12)
	ax.set_xlabel(Xlabel, fontsize=12)
	ax.set_ylabel(Ylabel, fontsize=12)
	if sizes:
		plt.axis(sizes)
	plt.tight_layout()

	if filename:
		plt.savefig(DIRECTORY + filename+".eps")
	else:
		plt.show()
	plt.clf()

#------------------------------------------------------------
np.set_printoptions(suppress=True)

data = np.load("/home/kacper/doktorat/FAIS/Higgs_CP_state/HiggsCP_data/" + "rhorho_raw.data.npy")

M_H = 125.0
M_TAU = 1.77

#------------------------------------------------------------
# [n, pi-, pi0, an, pi+, pi0]

tab = [data[:, 5 * i:5 * i + 4] for i in range(6)]
tab = np.array(tab)

p = [Particle(i) for i in tab]
#p = [Particle(data[:, 5 * i:5 * i + 4]) for i in range(7)]

p_tau1_nu, l_tau1_pi, l_tau1_rho = get_tau1(p)
p_tau1_h = sum(l_tau1_pi)

p_tau2_nu, l_tau2_pi, l_tau2_rho = get_tau2(p)
p_tau2_h = sum(l_tau2_pi)

#------------------------------------------------------------
p_rho_rho = sum(l_tau1_pi + l_tau2_pi)

PHI, THETA = calc_angles(p_tau1_h, p_rho_rho)

pb_tau1_h = boost_and_rotate(p_tau1_h, PHI, THETA, p_rho_rho)
pb_tau2_h = boost_and_rotate(p_tau2_h, PHI, THETA, p_rho_rho)
pb_tau1_nu = boost_and_rotate(p_tau1_nu, PHI, THETA, p_rho_rho)
pb_tau2_nu = boost_and_rotate(p_tau2_nu, PHI, THETA, p_rho_rho)

#------------------------------------------------------------

ve_x1_cms = pb_tau1_h.z / (pb_tau1_h + pb_tau1_nu).z
ve_x2_cms = pb_tau2_h.z / (pb_tau2_h + pb_tau2_nu).z

ve_x1_lab = p_tau1_h.length / (p_tau1_h + p_tau1_nu).length
ve_x2_lab = p_tau2_h.length / (p_tau2_h + p_tau2_nu).length

ve_alpha1_cms = 1/ve_x1_cms - 1
ve_alpha2_cms = 1/ve_x2_cms - 1
ve_alpha1_lab = 1/ve_x1_lab - 1
ve_alpha2_lab = 1/ve_x2_lab - 1


#------------------------------------------------------------

v_ETmiss_x = p_tau1_nu.x + p_tau2_nu.x
v_ETmiss_y = p_tau1_nu.y + p_tau2_nu.y

vr_ETmiss_x, vr_ETmiss_y = rotate_xy(v_ETmiss_x, v_ETmiss_y, PHI)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

ve_tau1_nu_trans = np.sqrt(np.square(pb_tau1_nu.x) + np.square(pb_tau1_nu.y))
ve_tau2_nu_trans = np.sqrt(np.square(pb_tau2_nu.x) + np.square(pb_tau2_nu.y))

#------------------------------------------------------------

va_alpha1_A, va_alpha2_A = approx_alpha_A(v_ETmiss_x, v_ETmiss_y, p_tau1_h, p_tau2_h)
va_alpha1_B, va_alpha2_B = approx_alpha_B(v_ETmiss_x, v_ETmiss_y, p_tau1_h, p_tau2_h)
va_alpha1_C, va_alpha2_C = approx_alpha_C(v_ETmiss_x, v_ETmiss_y, p_tau1_h, p_tau2_h)

va_x1_A = 1 / (1 + va_alpha1_A)
va_x2_A = 1 / (1 + va_alpha2_A)

#------------------------------------------------------------

va_tau1_nu_long_A = va_alpha1_A * pb_tau1_h.z
va_tau1_nu_long_B = va_alpha1_B * pb_tau1_h.z 
va_tau1_nu_long_C = va_alpha1_C * pb_tau1_h.z

va_tau2_nu_long_A = va_alpha2_A * pb_tau2_h.z
va_tau2_nu_long_B = va_alpha2_B * pb_tau2_h.z 
va_tau2_nu_long_C = va_alpha2_C * pb_tau2_h.z


va_tau1_nu_e_A = va_alpha1_A * pb_tau1_h.e
va_tau1_nu_e_B = va_alpha1_B * pb_tau1_h.e 
va_tau1_nu_e_C = va_alpha1_C * pb_tau1_h.e

va_tau2_nu_e_A = va_alpha2_A * pb_tau2_h.e
va_tau2_nu_e_B = va_alpha2_B * pb_tau2_h.e 
va_tau2_nu_e_C = va_alpha2_C * pb_tau2_h.e

#------------------------------------------------------------

va_tau1_nu_E_A = approx_E_nu(pb_tau1_h, va_tau1_nu_long_A)
va_tau1_nu_E_B = approx_E_nu(pb_tau1_h, va_tau1_nu_long_B)
va_tau1_nu_E_C = approx_E_nu(pb_tau1_h, va_tau1_nu_long_C)

va_tau2_nu_E_A = approx_E_nu(pb_tau2_h, va_tau2_nu_long_A)
va_tau2_nu_E_B = approx_E_nu(pb_tau2_h, va_tau2_nu_long_B)
va_tau2_nu_E_C = approx_E_nu(pb_tau2_h, va_tau2_nu_long_C)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

va_tau1_nu_trans_A = np.sqrt(np.square(va_tau1_nu_E_A) - np.square(va_tau1_nu_long_A))
va_tau1_nu_trans_B = np.sqrt(np.square(va_tau1_nu_E_B) - np.square(va_tau1_nu_long_B))
va_tau1_nu_trans_C = np.sqrt(np.square(va_tau1_nu_E_C) - np.square(va_tau1_nu_long_C))

va_tau2_nu_trans_A = np.sqrt(np.square(va_tau2_nu_E_A) - np.square(va_tau2_nu_long_A))
va_tau2_nu_trans_B = np.sqrt(np.square(va_tau2_nu_E_B) - np.square(va_tau2_nu_long_B))
va_tau2_nu_trans_C = np.sqrt(np.square(va_tau2_nu_E_C) - np.square(va_tau2_nu_long_C))

#--NU-X-Y--approx--------------------------------------------

try:
	lambda_noise = float(sys.argv[1])
except:
	lambda_noise = 0
v_tau1_nu_phi = np.arctan2(pb_tau1_nu.x, pb_tau1_nu.y)
v_tau2_nu_phi = np.arctan2(pb_tau2_nu.x, pb_tau2_nu.y)

vn_tau1_nu_phi = smear_exp(v_tau1_nu_phi, lambda_noise) if lambda_noise else v_tau1_nu_phi
vn_tau2_nu_phi = smear_exp(v_tau2_nu_phi, lambda_noise) if lambda_noise else v_tau2_nu_phi

va_tau1_nu_x = ve_tau1_nu_trans*np.sin(vn_tau1_nu_phi)
va_tau1_nu_y = ve_tau1_nu_trans*np.cos(vn_tau1_nu_phi)
va_tau2_nu_x = ve_tau2_nu_trans*np.sin(vn_tau2_nu_phi)
va_tau2_nu_y = ve_tau2_nu_trans*np.cos(vn_tau2_nu_phi)

#------------------------------------------------------------

isFilter = np.full(p_rho_rho.e.shape, True, dtype=bool)
for alpha in [va_alpha1_A, va_alpha1_B, va_alpha1_C, va_alpha2_A, va_alpha2_B, va_alpha2_C]:
	isFilter *= (alpha > 0)
for energy in [va_tau1_nu_E_A, va_tau1_nu_E_B, va_tau1_nu_E_C, va_tau2_nu_E_A, va_tau2_nu_E_B, va_tau2_nu_E_C]:
	isFilter *= (energy > 0)
for trans_comp in [va_tau1_nu_trans_A, va_tau1_nu_trans_B, va_tau1_nu_trans_C, va_tau2_nu_trans_A, va_tau2_nu_trans_B, va_tau2_nu_trans_C]:
	isFilter *= np.logical_not(np.isnan(trans_comp))

print(isFilter.sum())
print(isFilter.sum()/isFilter.shape[0])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

va_tau1_nu_long_A = va_tau1_nu_long_A[isFilter]
va_tau1_nu_long_B = va_tau1_nu_long_B[isFilter]
va_tau1_nu_long_C = va_tau1_nu_long_C[isFilter]
va_tau2_nu_long_A = va_tau2_nu_long_A[isFilter]
va_tau2_nu_long_B = va_tau2_nu_long_B[isFilter]
va_tau2_nu_long_C = va_tau2_nu_long_C[isFilter]

va_tau1_nu_e_A = va_tau1_nu_e_A[isFilter]
va_tau1_nu_e_B = va_tau1_nu_e_B[isFilter]
va_tau1_nu_e_C = va_tau1_nu_e_C[isFilter]
va_tau2_nu_e_A = va_tau2_nu_e_A[isFilter]
va_tau2_nu_e_B = va_tau2_nu_e_B[isFilter]
va_tau2_nu_e_C = va_tau2_nu_e_C[isFilter]

va_tau1_nu_trans_A = va_tau1_nu_trans_A[isFilter]
va_tau1_nu_trans_B = va_tau1_nu_trans_B[isFilter]
va_tau1_nu_trans_C = va_tau1_nu_trans_C[isFilter]
va_tau2_nu_trans_A = va_tau2_nu_trans_A[isFilter]
va_tau2_nu_trans_B = va_tau2_nu_trans_B[isFilter]
va_tau2_nu_trans_C = va_tau2_nu_trans_C[isFilter]

v_ETmiss_x = v_ETmiss_x[isFilter]
v_ETmiss_y = v_ETmiss_y[isFilter]
vr_ETmiss_x = vr_ETmiss_x[isFilter]
vr_ETmiss_y = vr_ETmiss_y[isFilter]

va_tau1_nu_x = va_tau1_nu_x[isFilter]
va_tau1_nu_y = va_tau1_nu_y[isFilter]
va_tau2_nu_x = va_tau2_nu_x[isFilter]
va_tau2_nu_y = va_tau2_nu_y[isFilter]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

particles = [pb_tau1_nu, pb_tau2_nu]
for i in range(len(particles)):
	particles[i].set(particles[i][isFilter])

ve_tau1_nu_trans = ve_tau1_nu_trans[isFilter]
ve_tau2_nu_trans = ve_tau2_nu_trans[isFilter]

#------------------------------------------------------------

plot(ve_x1_lab, ve_x1_cms, [0, 1.0, 0, 1.0], 0.01, filename="x1", xlabel=r"$x_1 \, (lab)$", ylabel=r"$x_1 \, (rho-rho)$")
plot(ve_x2_lab, ve_x2_cms, [0, 1.0, 0, 1.0], 0.01, filename="x2", xlabel=r"$x_2 \, (lab)$", ylabel=r"$x_2 \, (rho-rho)$")

plot(ve_alpha1_lab, va_alpha1_A, [-0.0, 3, -0.0, 3], 0.01, filename="x_alpha1_A", xlabel=r"$\alpha_1 \, (true)$", ylabel=r"$\alpha_1 \, (approx.)$")
plot(ve_alpha2_lab, va_alpha2_A, [-0.0, 3, -0.0, 3], 0.01, filename="x_alpha2_A", xlabel=r"$\alpha_2 \, (true)$", ylabel=r"$\alpha_2 \, (approx.)$")

plot(ve_alpha1_lab, va_alpha1_B, [-0.0, 3, -0.0, 3], 0.01, filename="x_alpha1_B", xlabel=r"$\alpha_1 \, (true)$", ylabel=r"$\alpha_1 \, (approx.)$")
plot(ve_alpha2_lab, va_alpha2_B, [-0.0, 3, -0.0, 3], 0.01, filename="x_alpha2_B", xlabel=r"$\alpha_2 \, (true)$", ylabel=r"$\alpha_2 \, (approx.)$")

plot(ve_alpha1_lab, va_alpha1_C, [-0.0, 3, -0.0, 3], 0.01, filename="x_alpha1_C", xlabel=r"$\alpha_1 \, (true)$", ylabel=r"$\alpha_1 \, (approx.)$")
plot(ve_alpha2_lab, va_alpha2_C, [-0.0, 3, -0.0, 3], 0.01, filename="x_alpha2_C", xlabel=r"$\alpha_2 \, (true)$", ylabel=r"$\alpha_2 \, (approx.)$")


smear_plot(np.arccos(np.cos(v_tau1_nu_phi)), np.arccos(np.cos(vn_tau1_nu_phi)), [0.0, np.pi, 0.0, 800000], step=0.05, filename="tau1_nu_detphiSmear_"+str(lambda_noise)+"_hist", title=r"Smear param. $\lambda=1.2$", Xlabel= r"$|\Delta\phi^{\nu_1}| \, (true-smeared)$", Ylabel= r"Number of Events", relflag=False, absflag=True)
smear_plot(np.arccos(np.cos(v_tau1_nu_phi)), np.arccos(np.cos(vn_tau1_nu_phi)), [0.0, np.pi, 0.0, 1500000], step=0.25, filename="tau1_nu_detphiSmear_"+str(lambda_noise)+"_coarse_hist", title=r"Smear param. $\lambda=1.2$", Xlabel= r"$|\Delta\phi^{\nu_1}| \, (true-smeared)$", Ylabel= r"Number of Events", relflag=False, absflag=True)
smear_plot(np.arccos(np.cos(v_tau1_nu_phi)), np.arccos(np.cos(vn_tau1_nu_phi)), [0.0, np.pi, 0.0, 800000], step=0.05, filename="tau2_nu_detphiSmear_"+str(lambda_noise)+"_hist", title=r"Smear param. $\lambda=1.2", Xlabel= r"$|\Delta\phi^{\nu_1}| \, (true-smeared)$",Ylabel= r"Number of Events", relflag=False, absflag=True)
smear_plot(np.arccos(np.cos(v_tau2_nu_phi)), np.arccos(np.cos(vn_tau2_nu_phi)), [0.0, np.pi, 0.0, 1500000], step=0.25, filename="tau2_nu_detphiSmear_"+str(lambda_noise)+"_coarse_hist", title=r"Smear param. $\lambda=1.2", Xlabel= r"$|\Delta\phi^{\nu_1}| \, (true-smeared)$", Ylabel= r"Number of Events", relflag=False, absflag=True)

#------------------------------------------------------------
error_plot(pb_tau1_nu.z, va_tau1_nu_long_A, [-1.1, 1.1, 0.0, 200000], 0.02, filename="tau1_nu_long_A_hist", title=r"$p_{\nu_1}^z \, (\tau \to \rho \nu)$", barrier=1, relflag=True, absflag=False)
error_plot(pb_tau1_nu.z, va_tau1_nu_long_B, [-1.1, 1.1, 0.0, 200000], 0.02, filename="tau1_nu_long_B_hist", title=r"$p_{\nu_1}^z \, (approx. B)$", barrier=1, relflag=True, absflag=False)
error_plot(pb_tau1_nu.z, va_tau1_nu_long_C, [-1.1, 1.1, 0.0, 200000], 0.02, filename="tau1_nu_long_C_hist", title=r"$p_{\nu_1}^z \, (approx. C)$", barrier=1, relflag=True, absflag=False)

error_plot(pb_tau2_nu.z, va_tau2_nu_long_A, [-1.1, 1.1, 0.0, 200000], 0.02, filename="tau2_nu_long_A_hist", title=r"$p_{\nu_2}^z \, (\tau \to \rho \nu)$", barrier=1, relflag=True, absflag=False)
error_plot(pb_tau2_nu.z, va_tau2_nu_long_B, [-1.1, 1.1, 0.0, 200000], 0.02, filename="tau2_nu_long_B_hist", title=r"$p_{\nu_2}^z \, (approx. B)$", barrier=1, relflag=True, absflag=False)
error_plot(pb_tau2_nu.z, va_tau2_nu_long_C, [-1.1, 1.1, 0.0, 200000], 0.02, filename="tau2_nu_long_C_hist", title=r"$p_{\nu_2}^z \, (approx. C)$", barrier=1, relflag=True, absflag=False)

error_plot(pb_tau1_nu.e, va_tau1_nu_e_A, [-1.1, 1.1, 0.0, 200000], 0.02, filename="tau1_nu_e_A_hist", title=r"$E_{\nu_1} \, (\tau \to \rho \nu)$", barrier=1, relflag=True, absflag=False)
error_plot(pb_tau1_nu.e, va_tau1_nu_e_B, [-1.1, 1.1, 0.0, 200000], 0.02, filename="tau1_nu_e_B_hist", title=r"$E_{\nu_1} \, (approx. B)$", barrier=1, relflag=True, absflag=False)
error_plot(pb_tau1_nu.e, va_tau1_nu_e_C, [-1.1, 1.1, 0.0, 200000], 0.02, filename="tau1_nu_e_C_hist", title=r"$E_{\nu_1} \, (approx. C)$", barrier=1, relflag=True, absflag=False)

error_plot(pb_tau2_nu.e, va_tau2_nu_e_A, [-1.1, 1.1, 0.0, 200000], 0.02, filename="tau2_nu_e_A_hist", title=r"$E_{\nu_2} \, (\tau \to \rho \nu)$", barrier=1, relflag=True, absflag=False)
error_plot(pb_tau2_nu.e, va_tau2_nu_e_B, [-1.1, 1.1, 0.0, 200000], 0.02, filename="tau2_nu_e_B_hist", title=r"$E_{\nu_2} \, (approx. B)$", barrier=1, relflag=True, absflag=False)
error_plot(pb_tau2_nu.e, va_tau2_nu_e_C, [-1.1, 1.1, 0.0, 200000], 0.02, filename="tau2_nu_e_C_hist", title=r"$E_{\nu_2} \, (approx. C)$", barrier=1, relflag=True, absflag=False)

error_plot(ve_tau1_nu_trans, va_tau1_nu_trans_A, [-1.1, 1.1, 0.0, 400000], 0.02, filename="tau1_nu_trans_A_hist", title=r"$p^T_{\nu_1} \, (\tau \to \rho \nu)$", barrier=1, relflag=True, absflag=False)
error_plot(ve_tau1_nu_trans, va_tau1_nu_trans_B, [-1.1, 1.1, 0.0, 400000], 0.02, filename="tau1_nu_trans_B_hist", title=r"$p^T_{\nu_1} \, (approx. B)$", barrier=1, relflag=True, absflag=False)
error_plot(ve_tau1_nu_trans, va_tau1_nu_trans_C, [-1.1, 1.1, 0.0, 400000], 0.02, filename="tau1_nu_trans_C_hist", title=r"$p^T_{\nu_1} \, (approx. C)$", barrier=1, relflag=True, absflag=False)

error_plot(ve_tau2_nu_trans, va_tau2_nu_trans_A, [-1.1, 1.1, 0.0, 400000], 0.02, filename="tau2_nu_trans_A_hist", title=r"$p^T_{\nu_2} \, (\tau \to \rho \nu)$", barrier=1, relflag=True, absflag=False)
error_plot(ve_tau2_nu_trans, va_tau2_nu_trans_B, [-1.1, 1.1, 0.0, 400000], 0.02, filename="tau2_nu_trans_B_hist", title=r"$p^T_{\nu_2} \, (approx. B)$", barrier=1, relflag=True, absflag=False)
error_plot(ve_tau2_nu_trans, va_tau2_nu_trans_C, [-1.1, 1.1, 0.0, 400000], 0.02, filename="tau2_nu_trans_C_hist", title=r"$p^T_{\nu_2} \, (approx. C)$", barrier=1, relflag=True, absflag=False)

#------------------------------------------------------------

