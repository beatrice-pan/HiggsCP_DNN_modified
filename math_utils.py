import numpy as np


def p_norm_cross_product(v1, v2):
    v1 = v1.vec[:, :3]
    v2 = v2.vec[:, :3]
    x = np.cross(v1, v2)
    return x / np.linalg.norm(x, axis=1).reshape([-1, 1])


def compute_costheta(v1, v2, v3, v4):
    normal1 = p_norm_cross_product(v1, v2)
    normal2 = p_norm_cross_product(v3, v4)
    return (normal1 * normal2).sum(axis=1)


def get_costheta(v1, v2, v3, v4, frame):
    return compute_costheta(
            v1.boost(frame), v2.boost(frame),
            v3.boost(frame), v4.boost(frame))

def get_y(v1, v2, frame):
    v1 = v1.boost(frame)
    v2 = v2.boost(frame)
    return (v1.e - v2.e) / (v1.e + v2.e)

def get_y2(a1, rho, other_pi, frame):
    rho = rho.boost(frame)
    other_pi = other_pi.boost(frame)
    y1 = rho.e / (rho.e + other_pi.e)
    x1 = (np.power(a1.recalculated_mass, 2) - np.power(other_pi.recalculated_mass, 2) + np.power(rho.recalculated_mass, 2)) / (2 * np.power(a1.recalculated_mass, 2))
    y1 = np.where(y1 > x1, -y1, y1)
    return y1

def compute_acoplanar_angle(v1, v2, v3, v4):
    normal1 = p_norm_cross_product(v1, v2)
    normal2 = p_norm_cross_product(v3, v4)
    costheta = (normal1 * normal2).sum(axis=1)

    theta = np.arccos(costheta)

    threshold = (v1.vec[:, :3] * normal2).sum(axis=1)
    theta = np.where(threshold > 0, 2 * np.pi - theta, theta)
    return theta


def get_acoplanar_angle(v1, v2, v3, v4, frame):
    return compute_acoplanar_angle(
            v1.boost(frame), v2.boost(frame),
            v3.boost(frame), v4.boost(frame))

def compute_sintheta(v1, v2, v3, v4):
    normal1 = p_norm_cross_product(v1, v2)
    normal2 = p_norm_cross_product(v3, v4)
    costheta = (normal1 * normal2).sum(axis=1)

    theta = np.arccos(costheta)

    threshold = (v1.vec[:, :3] * normal2).sum(axis=1)
    theta = np.where(threshold > 0, 2 * np.pi - theta, theta)
    return np.sin(theta)


def get_sintheta(v1, v2, v3, v4, frame):
    return compute_sintheta(
            v1.boost(frame), v2.boost(frame),
            v3.boost(frame), v4.boost(frame))

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
    return x + sign*noise

def smear_expnorm(x, lambd, loc, scale):
    noise = np.random.exponential(scale=lambd, size=x.shape) + np.random.normal(loc = loc, scale=scale, size=x.shape)
    sign = np.random.choice(np.array([-1,1]), size=x.shape)
    return x + sign*noise

def smear_log(x):
    rn = np.random.random()
    return x * (- np.log(rn))

def polynomial_density(x, beta, b, c): return np.exp(-(1/beta)*x) * (1 + (b**2)*(x**2) + (c**2)*(x**4))

def smear_polynomial(x, beta, b, c):
	s = x.size	
	noise = np.zeros(s)
	limit = polynomial_density(np.arange(0, 10, 0.01), beta, b, c).max()
	i = 0
	while i < noise.shape[0]:
		new_noise = np.random.uniform(0,10, size=s-i)
		isOK = polynomial_density(new_noise, beta, b, c) > np.random.uniform(0, limit, new_noise.size)
		new_noise = new_noise[isOK]
		noise[i:i+new_noise.shape[0]] = new_noise
		i += new_noise.shape[0]
	sign = np.random.choice(np.array([-1,1]), size=s)
	return x + sign*noise

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

def scale_lifetime(particle):
    rn = np.random.random(size = len(particle.x))
    scale = (- np.log(rn))
    print(scale)
    part = particle.scale_lifetime(scale)
    return part
