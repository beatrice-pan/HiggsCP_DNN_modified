import numpy as np
from src_py.particle import Particle
from math_utils import *

class A1A1Event(object):
    def __init__(self, data, args, debug=False):
        # [n, pi-, pi-, pi+, an, pi+, pi+, pi-]
        p = [Particle(data[:, 5 * i:5 * i + 4]) for i in range(8)]
        cols = []
        labels = []
        self.labels_suppl = []
        self.cols_suppl = []

        def get_tau1(p):
            p_tau1_nu = p[0]
            l_tau1_pi = p[1:4]                              # pi-, pi-, pi+
            p_tau1_a1 = sum(l_tau1_pi)
            p_tau1 = p_tau1_nu + p_tau1_a1

            l_tau1_rho = [None]*2
            l_tau1_rho[0] = l_tau1_pi[0] + l_tau1_pi[2]     # pi1- + pi+
            l_tau1_rho[1] = l_tau1_pi[1] + l_tau1_pi[2]     # pi2- + pi+

            return p_tau1_nu, p_tau1_a1, l_tau1_pi, l_tau1_rho, p_tau1

        def get_tau2(p):
            p_tau2_nu = p[4]
            l_tau2_pi = p[5:8]
            p_tau2_a1 = sum(l_tau2_pi)
            p_tau2 = p_tau2_nu + p_tau2_a1

            l_tau2_rho = [None]*2
            l_tau2_rho[0] = l_tau2_pi[0] + l_tau2_pi[2]     # pi1+ + pi-
            l_tau2_rho[1] = l_tau2_pi[1] + l_tau2_pi[2]     # pi2+ + pi-

            return p_tau2_nu, p_tau2_a1, l_tau2_pi, l_tau2_rho, p_tau2


        p_tau1_nu, p_tau1_a1, l_tau1_pi, l_tau1_rho, p_tau1 = get_tau1(p)
        p_tau1_rho = sum(l_tau1_pi)

        p_tau2_nu, p_tau2_a1, l_tau2_pi, l_tau2_rho, p_tau2 = get_tau2(p)
        p_tau2_rho = sum(l_tau2_pi)

        p_a1_a1 = sum(l_tau1_pi + l_tau2_pi)

        PHI, THETA = calc_angles(p_tau1_a1, p_a1_a1)
        beta_noise = args.BETA

        for i, idx in enumerate([0, 1, 2, 3, 4, 5, 6, 7]):
            part = boost_and_rotate(p[idx], PHI, THETA, p_a1_a1)
            if args.FEAT in ["Variant-1.0", "Variant-1.1", "Variant-2.0", "Variant-2.1", "Variant-2.2",
                             "Variant-3.0", "Variant-3.1", "Variant-4.0", "Variant-4.1"]:
                if idx not in [0, 4]:
                    cols.append(part.vec)
            if args.FEAT == "Variant-All":
                cols.append(part.vec)
                
        if args.FEAT == "Variant-4.0":
            part   = boost_and_rotate(p_tau1, PHI, THETA, p_a1_a1)
            cols.append(part.vec)
            part   = boost_and_rotate(p_tau2, PHI, THETA, p_a1_a1)
            cols.append(part.vec)

        if args.FEAT == "Variant-4.1":
            p_tau1_approx = scale_lifetime(p_tau1)
            part   = boost_and_rotate(p_tau1_approx, PHI, THETA, p_a1_a1)
            cols.append(part.vec)
            p_tau2_approx = scale_lifetime(p_tau2)
            part   = boost_and_rotate(p_tau2_approx, PHI, THETA, p_a1_a1)
            cols.append(part.vec)
            self.cols_suppl.append(p_tau1_approx.x/p_tau1.x)
            self.cols_suppl.append(p_tau1_approx.y/p_tau1.y)
            self.cols_suppl.append(p_tau1_approx.z/p_tau1.z)
            self.cols_suppl.append(p_tau1_approx.e/p_tau1.e)
            self.cols_suppl.append(p_tau2_approx.x/p_tau2.x)
            self.cols_suppl.append(p_tau2_approx.y/p_tau2.y)
            self.cols_suppl.append(p_tau2_approx.z/p_tau2.z)
            self.cols_suppl.append(p_tau2_approx.e/p_tau2.e)
            self.cols_suppl.append(p_tau1.vec)
            self.cols_suppl.append(p_tau2.vec)
            self.cols_suppl.append(p_tau1.pt)
            self.cols_suppl.append(p_tau2.pt)

        # rho particles
        if args.FEAT == "Variant-1.1":
            for i, rho in enumerate(l_tau1_rho + l_tau2_rho):
                rho = boost_and_rotate(rho, PHI, THETA, p_a1_a1)
                cols.append(rho.vec)

        # recalculated masses
        if args.FEAT == "Variant-1.1":
            for i, part in enumerate(l_tau1_rho + l_tau2_rho + [p_tau1_a1, p_tau2_a1]):
                part = boost_and_rotate(part, PHI, THETA, p_a1_a1)
                cols.append(part.recalculated_mass)

        if args.FEAT == "Variant-1.1":
            for i in [1, 2]:
                for ii in [5, 6]:
                    rho = p[i] + p[3]
                    other_pi = p[2 if i == 1 else 1]
                    rho2 = p[ii] + p[7]
                    other_pi2 = p[6 if i == 5 else 5]
                    rho_rho = rho + rho2
                    a1_rho = p_tau1_a1 + rho2
                    rho_a1 = rho + p_tau2_a1

                    cols += [get_acoplanar_angle(p[i], p[3], p[ii], p[7], rho_rho),
                             get_acoplanar_angle(rho, other_pi, p[ii], p[7], a1_rho),
                             get_acoplanar_angle(p[i], p[3], rho2, other_pi2, rho_a1),
                             get_acoplanar_angle(rho, other_pi, rho2, other_pi2, p_a1_a1)]

                    cols += [get_y(p[i], p[3], rho_rho), get_y(p[ii], p[7], rho_rho),
                             get_y2(p_tau1_a1, rho, other_pi, a1_rho), get_y(p[ii], p[7], a1_rho),
                             get_y(p[i], p[3], rho_a1), get_y2(p_tau2_a1, rho2, other_pi2, rho_a1),
                             get_y2(p_tau1_a1, rho, other_pi, p_a1_a1), get_y2(p_tau2_a1, rho2, other_pi2, p_a1_a1)]

        #------------------------------------------------------------

        pb_tau1_h = boost_and_rotate(p_tau1_rho, PHI, THETA, p_a1_a1)
        pb_tau2_h = boost_and_rotate(p_tau2_rho, PHI, THETA, p_a1_a1)
        pb_tau1_nu = boost_and_rotate(p_tau1_nu, PHI, THETA, p_a1_a1)
        pb_tau2_nu = boost_and_rotate(p_tau2_nu, PHI, THETA, p_a1_a1)

        #------------------------------------------------------------

        v_ETmiss_x = p_tau1_nu.x + p_tau2_nu.x
        v_ETmiss_y = p_tau1_nu.y + p_tau2_nu.y
        if args.FEAT == "Variant-2.2":
            cols += [v_ETmiss_x, v_ETmiss_y]

        vr_ETmiss_x, vr_ETmiss_y = rotate_xy(v_ETmiss_x, v_ETmiss_y, PHI)

        #------------------------------------------------------------

        if args.METHOD == "A":
            va_alpha1, va_alpha2 = approx_alpha_A(v_ETmiss_x, v_ETmiss_y, p_tau1_rho, p_tau2_rho)
        elif args.METHOD == "B":
            va_alpha1, va_alpha2 = approx_alpha_B(v_ETmiss_x, v_ETmiss_y, p_tau1_rho, p_tau2_rho)
        elif args.METHOD == "C":
            va_alpha1, va_alpha2 = approx_alpha_C(v_ETmiss_x, v_ETmiss_y, p_tau1_rho, p_tau2_rho)

        #------------------------------------------------------------

        va_tau1_nu_long = va_alpha1 * pb_tau1_h.z
        va_tau2_nu_long = va_alpha2 * pb_tau2_h.z

        #------------------------------------------------------------

        va_tau1_nu_E = approx_E_nu(pb_tau1_h, va_tau1_nu_long)
        va_tau2_nu_E = approx_E_nu(pb_tau2_h, va_tau2_nu_long)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        va_tau1_nu_trans = np.sqrt(np.square(va_tau1_nu_E) - np.square(va_tau1_nu_long))
        va_tau2_nu_trans = np.sqrt(np.square(va_tau2_nu_E) - np.square(va_tau2_nu_long))

           
       # FIX THIS
       # lambda_noise = args.LAMBDA
        v_tau1_nu_phi = np.arctan2(pb_tau1_nu.x, pb_tau1_nu.y) #boosted
        v_tau2_nu_phi = np.arctan2(pb_tau2_nu.x, pb_tau2_nu.y)
        vn_tau1_nu_phi = smear_exp(v_tau1_nu_phi, beta_noise)
        vn_tau2_nu_phi = smear_exp(v_tau2_nu_phi, beta_noise)

        tau1_sin_phi = np.sin(vn_tau1_nu_phi)
        tau1_cos_phi = np.cos(vn_tau1_nu_phi)
        tau2_sin_phi = np.sin(vn_tau2_nu_phi)
        tau2_cos_phi = np.cos(vn_tau2_nu_phi) 

        ve_x1_cms = pb_tau1_h.z / (pb_tau1_h + pb_tau1_nu).z
        ve_x2_cms = pb_tau2_h.z / (pb_tau2_h + pb_tau2_nu).z

        ve_alpha1_cms = 1/ve_x1_cms - 1
        ve_alpha2_cms = 1/ve_x2_cms - 1

        ve_tau1_nu_long = ve_alpha1_cms * pb_tau1_h.z
        ve_tau2_nu_long = ve_alpha2_cms * pb_tau2_h.z

        ve_tau1_nu_E = approx_E_nu(pb_tau1_h, ve_tau1_nu_long)
        ve_tau2_nu_E = approx_E_nu(pb_tau2_h, ve_tau2_nu_long)


        ve_tau1_nu_trans = np.sqrt(np.square(ve_tau1_nu_E) - np.square(ve_tau1_nu_long))
        ve_tau2_nu_trans = np.sqrt(np.square(ve_tau2_nu_E) - np.square(ve_tau2_nu_long))


        if args.FEAT in ["Variant-2.1", "Variant-2.2"]:
            cols += [va_tau1_nu_long, va_tau2_nu_long, va_tau1_nu_E, va_tau2_nu_E, va_tau1_nu_trans, va_tau2_nu_trans]
        elif args.FEAT in ["Variant-3.0", "Variant-3.1"]:
            cols += [va_tau1_nu_long, va_tau2_nu_long, va_tau1_nu_E, va_tau2_nu_E, va_tau1_nu_trans * tau1_sin_phi,
                     va_tau2_nu_trans * tau2_sin_phi, va_tau1_nu_trans * tau1_cos_phi, va_tau2_nu_trans * tau2_cos_phi]
        elif args.FEAT == "Variant-2.0":
            cols += [ve_tau1_nu_long, ve_tau2_nu_long, ve_tau1_nu_E, ve_tau2_nu_E, ve_tau1_nu_trans, ve_tau2_nu_trans]

        print(ve_tau1_nu_long)
        # filter
        filt = (p_tau1_a1.pt >= 20) & (p_tau2_a1.pt >= 20)
        for part in (l_tau1_pi + l_tau2_pi):
            filt = filt & (part.pt >= 1)
        #filt = filt.astype(np.float32)

        if args.FEAT in ["Variant-1.0", "Variant-1.1", "Variant-All", "Variant-4.0", "Variant-4.1"]:
            cols += [filt]

        elif args.FEAT in ["Variant-2.1", "Variant-2.2", "Variant-3.0", "Variant-3.1"]:
            isFilter = np.full(p_tau1_a1.e.shape, True, dtype=bool)
            
            va_alpha1_A, va_alpha2_A = approx_alpha_A(v_ETmiss_x, v_ETmiss_y, p_tau1_rho, p_tau2_rho)
            va_alpha1_B, va_alpha2_B = approx_alpha_B(v_ETmiss_x, v_ETmiss_y, p_tau1_rho, p_tau2_rho)
            va_alpha1_C, va_alpha2_C = approx_alpha_C(v_ETmiss_x, v_ETmiss_y, p_tau1_rho, p_tau2_rho)

            va_tau1_nu_long_A = va_alpha1_A * pb_tau1_h.z
            va_tau1_nu_long_B = va_alpha1_B * pb_tau1_h.z 
            va_tau1_nu_long_C = va_alpha1_C * pb_tau1_h.z

            va_tau2_nu_long_A = va_alpha2_A * pb_tau2_h.z
            va_tau2_nu_long_B = va_alpha2_B * pb_tau2_h.z 
            va_tau2_nu_long_C = va_alpha2_C * pb_tau2_h.z

		
            va_tau1_nu_E_A = approx_E_nu(pb_tau1_h, va_tau1_nu_long_A)
            va_tau1_nu_E_B = approx_E_nu(pb_tau1_h, va_tau1_nu_long_B)
            va_tau1_nu_E_C = approx_E_nu(pb_tau1_h, va_tau1_nu_long_C)

            va_tau2_nu_E_A = approx_E_nu(pb_tau2_h, va_tau2_nu_long_A)
            va_tau2_nu_E_B = approx_E_nu(pb_tau2_h, va_tau2_nu_long_B)
            va_tau2_nu_E_C = approx_E_nu(pb_tau2_h, va_tau2_nu_long_C)

            va_tau1_nu_trans_A = np.sqrt(np.square(va_tau1_nu_E_A) - np.square(va_tau1_nu_long_A))
            va_tau1_nu_trans_B = np.sqrt(np.square(va_tau1_nu_E_B) - np.square(va_tau1_nu_long_B))
            va_tau1_nu_trans_C = np.sqrt(np.square(va_tau1_nu_E_C) - np.square(va_tau1_nu_long_C))

            va_tau2_nu_trans_A = np.sqrt(np.square(va_tau2_nu_E_A) - np.square(va_tau2_nu_long_A))
            va_tau2_nu_trans_B = np.sqrt(np.square(va_tau2_nu_E_B) - np.square(va_tau2_nu_long_B))
            va_tau2_nu_trans_C = np.sqrt(np.square(va_tau2_nu_E_C) - np.square(va_tau2_nu_long_C))

            for alpha in [va_alpha1_A, va_alpha1_B, va_alpha1_C, va_alpha2_A, va_alpha2_B, va_alpha2_C]:
                isFilter *= (alpha > 0)
            for energy in [va_tau1_nu_E_A, va_tau1_nu_E_B, va_tau1_nu_E_C, va_tau2_nu_E_A, va_tau2_nu_E_B, va_tau2_nu_E_C]:
                isFilter *= (energy > 0)
            for trans_comp in [va_tau1_nu_trans_A, va_tau1_nu_trans_B, va_tau1_nu_trans_C,
                               va_tau2_nu_trans_A, va_tau2_nu_trans_B, va_tau2_nu_trans_C]:
                isFilter *= np.logical_not(np.isnan(trans_comp))
            cols += [filt * isFilter]


        elif args.FEAT in ["Variant-2.0"]:
            isFilter = np.full(p_tau1_a1.e.shape, True, dtype=bool)
            for alpha in [ve_alpha1_cms, ve_alpha2_cms]:
                isFilter *= (alpha > 0)
            for energy in [ve_tau1_nu_E, ve_tau2_nu_E]:
                isFilter *= (energy > 0)
            for trans_comp in [ve_tau1_nu_trans, ve_tau2_nu_trans]:
                isFilter *= np.logical_not(np.isnan(trans_comp))
            cols += [filt * isFilter]

        for i in range(len(cols)):
            if len(cols[i].shape) == 1:
                cols[i] = cols[i].reshape([-1, 1])
        for i in range(len(self.cols_suppl)):
            if len(self.cols_suppl[i].shape) == 1:
                self.cols_suppl[i] = self.cols_suppl[i].reshape([-1, 1])
             
        self.cols = np.concatenate(cols, 1)
        if len(self.cols_suppl) >0 :
            self.cols_suppl = np.concatenate(self.cols_suppl, 1)


        if args.BETA > 0:
            vn_tau1_nu_phi = smear_polynomial(v_tau1_nu_phi, args.BETA, args.pol_b, args.pol_c)
            vn_tau2_nu_phi = smear_polynomial(v_tau2_nu_phi, args.BETA, args.pol_b, args.pol_c)

            tau1_sin_phi = np.sin(vn_tau1_nu_phi)
            tau1_cos_phi = np.cos(vn_tau1_nu_phi)
            tau2_sin_phi = np.sin(vn_tau2_nu_phi)
            tau2_cos_phi = np.cos(vn_tau2_nu_phi)

        self.valid_cols = [va_tau1_nu_trans * tau1_sin_phi, va_tau2_nu_trans * tau2_sin_phi,
                           va_tau1_nu_trans * tau1_cos_phi, va_tau2_nu_trans * tau2_cos_phi]


        if args.FEAT == "Variant-1.0":
            self.labels = ["tau1_pi_1_px", "tau1_pi_1_py", "tau1_pi_1_pz", "tau1_pi_1_e",
                           "tau1_pi_2_px", "tau1_pi_2_py", "tau1_pi_2_pz", "tau1_pi_2_e",
                           "tau1_pi_3_px", "tau1_pi_3_py", "tau1_pi_3_pz", "tau1_pi_3_e",
                           "tau2_pi_1_px", "tau2_pi_1_py", "tau2_pi_1_pz", "tau2_pi_1_e",
                           "tau2_pi_2_px", "tau2_pi_2_py", "tau2_pi_2_pz", "tau2_pi_2_e",
                           "tau2_pi_3_px", "tau2_pi_3_py", "tau2_pi_3_pz", "tau2_pi_3_e"]
            
        elif args.FEAT == "Variant-1.1":
            self.labels = ["tau1_pi_1_px", "tau1_pi_1_py", "tau1_pi_1_pz", "tau1_pi_1_e",
                           "tau1_pi_2_px", "tau1_pi_2_py", "tau1_pi_2_pz", "tau1_pi_2_e",
                           "tau1_pi_3_px", "tau1_pi_3_py", "tau1_pi_3_pz", "tau1_pi_3_e",
                           "tau2_pi_1_px", "tau2_pi_1_py", "tau2_pi_1_pz", "tau2_pi_1_e",
                           "tau2_pi_2_px", "tau2_pi_2_py", "tau2_pi_2_pz", "tau2_pi_2_e",
                           "tau2_pi_3_px", "tau2_pi_3_py", "tau2_pi_3_pz", "tau2_pi_3_e",
                           "tau1_rho1_px", "tau1_rho1_py", "tau1_rho1_pz", "tau1_rho1_e",
                           "tau1_rho2_px", "tau1_rho2_py", "tau1_rho2_pz", "tau1_rho2_e",
                           "tau2_rho1_px",  "tau2_rho1_py",  "tau2_rho1_pz",  "tau2_rho1_e",
                           "tau2_rho2_px",  "tau2_rho2_py",  "tau2_rho2_pz",  "tau2_rho2_e",
                           "tau1_rho1_mass", "tau1_rho2_mass", "tau2_rho1_mass", "tau2_rho2_mass", "tau1_a1_mass","tau2_a1_mass",
                           "aco_angle_1", "aco_angle_2", "aco_angle_3", "aco_angle_4",
                           "tau1_y1", "tau2_y1", "tau1_y2", "tau2_y2", "tau1_y3", "tau2_y3", "tau1_y4", "tau2_y4",
                           "aco_angle_5", "aco_angle_6", "aco_angle_7", "aco_angle_8",
                           "tau1_y5", "tau2_y5", "tau1_y6", "tau2_y6", "tau1_y7", "tau2_y7", "tau1_y8", "tau2_y8",
                           "aco_angle_9", "aco_angle_10", "aco_angle_11", "aco_angle_12",
                           "tau1_y9", "tau2_y9", "tau1_y10", "tau2_y10", "tau1_y11", "tau2_y11", "tau1_y12", "tau2_y12",
                           "aco_angle_13", "aco_angle_14", "aco_angle_15", "aco_angle_16",
                           "tau1_y13", "tau2_y13", "tau1_y14", "tau2_y14", "tau1_y15", "tau2_y15", "tau1_y16", "tau2_y16"]

        elif args.FEAT ==  "Variant-2.0":            
            self.labels = ["tau1_pi_1_px", "tau1_pi_1_py", "tau1_pi_1_pz", "tau1_pi_1_e",
                           "tau1_pi_2_px", "tau1_pi_2_py", "tau1_pi_2_pz", "tau1_pi_2_e",
                           "tau1_pi_3_px", "tau1_pi_3_py", "tau1_pi_3_pz", "tau1_pi_3_e",
                           "tau2_pi_1_px", "tau2_pi_1_py", "tau2_pi_1_pz", "tau2_pi_1_e",
                           "tau2_pi_2_px", "tau2_pi_2_py", "tau2_pi_2_pz", "tau2_pi_2_e",
                           "tau2_pi_3_px", "tau2_pi_3_py", "tau2_pi_3_pz", "tau2_pi_3_e",
                           "tau1_nu_pL", "tau2_nu_pL", "tau1_nu_e", "tau2_nu_e", "tau1_nu_pT", "tau2_nu_pT"]            
                           
        elif args.FEAT ==  "Variant-2.1":
            self.labels = ["tau1_pi_1_px", "tau1_pi_1_py", "tau1_pi_1_pz", "tau1_pi_1_e",
                           "tau1_pi_2_px", "tau1_pi_2_py", "tau1_pi_2_pz", "tau1_pi_2_e",
                           "tau1_pi_3_px", "tau1_pi_3_py", "tau1_pi_3_pz", "tau1_pi_3_e",
                           "tau2_pi_1_px", "tau2_pi_1_py", "tau2_pi_1_pz", "tau2_pi_1_e",
                           "tau2_pi_2_px", "tau2_pi_2_py", "tau2_pi_2_pz", "tau2_pi_2_e",
                           "tau2_pi_3_px", "tau2_pi_3_py", "tau2_pi_3_pz", "tau2_pi_3_e",
                           "tau1_nu_approx_pL", "tau2_nu_approx_pL", "tau1_nu_approx_e", "tau2_nu_approx_e",
                           "tau1_nu_approx_pT", "tau2_nu_approx_pT"]
        elif args.FEAT ==  "Variant-2.2":
            self.labels = ["tau1_pi_1_px", "tau1_pi_1_py", "tau1_pi_1_pz", "tau1_pi_1_e",
                           "tau1_pi_2_px", "tau1_pi_2_py", "tau1_pi_2_pz", "tau1_pi_2_e",
                           "tau1_pi_3_px", "tau1_pi_3_py", "tau1_pi_3_pz", "tau1_pi_3_e",
                           "tau2_pi_1_px", "tau2_pi_1_py", "tau2_pi_1_pz", "tau2_pi_1_e",
                           "tau2_pi_2_px", "tau2_pi_2_py", "tau2_pi_2_pz", "tau2_pi_2_e",
                           "tau2_pi_3_px", "tau2_pi_3_py", "tau2_pi_3_pz", "tau2_pi_3_e",
                           "ETmiss_px", "ETmiss_py",
                           "tau1_nu_approx_pL", "tau2_nu_approx_pL", "tau1_nu_approx_e", "tau2_nu_approx_e",
                           "tau1_nu_approx_pT", "tau2_nu_approx_pT"]

        elif args.FEAT ==  "Variant-3.0":
            self.labels = ["tau1_pi_1_px", "tau1_pi_1_py", "tau1_pi_1_pz", "tau1_pi_1_e",
                           "tau1_pi_2_px", "tau1_pi_2_py", "tau1_pi_2_pz", "tau1_pi_2_e",
                           "tau1_pi_3_px", "tau1_pi_3_py", "tau1_pi_3_pz", "tau1_pi_3_e",
                           "tau2_pi_1_px", "tau2_pi_1_py", "tau2_pi_1_pz", "tau2_pi_1_e",
                           "tau2_pi_2_px", "tau2_pi_2_py", "tau2_pi_2_pz", "tau2_pi_2_e",
                           "tau2_pi_3_px", "tau2_pi_3_py", "tau2_pi_3_pz", "tau2_pi_3_e",
                           "tau1_nu_approx_px", "tau1_nu_approx_py", "tau2_nu_approx_px", "tau1_nu_approx_py"]
        elif args.FEAT ==  "Variant-3.1":
            self.labels = ["tau1_pi_1_px", "tau1_pi_1_py", "tau1_pi_1_pz", "tau1_pi_1_e",
                           "tau1_pi_2_px", "tau1_pi_2_py", "tau1_pi_2_pz", "tau1_pi_2_e",
                           "tau1_pi_3_px", "tau1_pi_3_py", "tau1_pi_3_pz", "tau1_pi_3_e",
                           "tau2_pi_1_px", "tau2_pi_1_py", "tau2_pi_1_pz", "tau2_pi_1_e",
                           "tau2_pi_2_px", "tau2_pi_2_py", "tau2_pi_2_pz", "tau2_pi_2_e",
                           "tau2_pi_3_px", "tau2_pi_3_py", "tau2_pi_3_pz", "tau2_pi_3_e",
                           "tau1_nu_approx_px", "tau1_nu_approx_py", "tau2_nu_approx_px", "tau1_nu_approx_py"]

        elif args.FEAT ==  "Variant-4.0":
            self.labels = ["tau1_pi_1_px", "tau1_pi_1_py", "tau1_pi_1_pz", "tau1_pi_1_e",
                           "tau1_pi_2_px", "tau1_pi_2_py", "tau1_pi_2_pz", "tau1_pi_2_e",
                           "tau1_pi_3_px", "tau1_pi_3_py", "tau1_pi_3_pz", "tau1_pi_3_e",
                           "tau2_pi_1_px", "tau2_pi_1_py", "tau2_pi_1_pz", "tau2_pi_1_e",
                           "tau2_pi_2_px", "tau2_pi_2_py", "tau2_pi_2_pz", "tau2_pi_2_e",
                           "tau2_pi_3_px", "tau2_pi_3_py", "tau2_pi_3_pz", "tau2_pi_3_e",
                           "tau1_px",      "tau1_py",      "tau1_pz",      "tau1_e",
                           "tau2_px",      "tau2_py",      "tau2_pz",      "tau2_e"]
        elif args.FEAT ==  "Variant-4.1":
            self.labels = ["tau1_pi_1_px", "tau1_pi_1_py", "tau1_pi_1_pz", "tau1_pi_1_e",
                           "tau1_pi_2_px", "tau1_pi_2_py", "tau1_pi_2_pz", "tau1_pi_2_e",
                           "tau1_pi_3_px", "tau1_pi_3_py", "tau1_pi_3_pz", "tau1_pi_3_e",
                           "tau2_pi_1_px", "tau2_pi_1_py", "tau2_pi_1_pz", "tau2_pi_1_e",
                           "tau2_pi_2_px", "tau2_pi_2_py", "tau2_pi_2_pz", "tau2_pi_2_e",
                           "tau2_pi_3_px", "tau2_pi_3_py", "tau2_pi_3_pz", "tau2_pi_3_e",
                           "tau1_approx_px", "tau1_approx_py", "tau1_approx_pz", "tau1_approx_e",
                           "tau2_approx_px", "tau2_approx_py", "tau2_approx_pz", "tau2_approx_e"]
            self.labels_suppl = ["tau1_px_ratio_LAB","tau1_py_ratio_LAB","tau1_pz_ratio_LAB","tau1_e_ratio_LAB",
                                 "tau2_px_ratio_LAB","tau2_py_ratio_LAB","tau2_pz_ratio_LAB","tau2_e_ratio_LAB",
                                 "tau1_px_LAB", "tau1_py_LAB", "tau1_pz_LAB", "tau1_e_LAB",
                                 "tau2_px_LAB", "tau2_py_LAB", "tau2_pz_LAB", "tau2_e_LAB",
                                 "tau1_pT_LAB", "tau2_pT_LAB"]
 
        elif args.FEAT == "Variant-All":
            self.labels = ["tau1_nu_px",   "tau1_nu_py",   "tau1_nu_pz",   "tau1_nu_e",
                           "tau1_pi_1_px", "tau1_pi_1_py", "tau1_pi_1_pz", "tau1_pi_1_e"
                           "tau1_pi_2_px", "tau1_pi_2_py", "tau1_pi_2_pz", "tau1_pi_2_e",
                           "tau1_pi_3_px", "tau1_pi_3_py", "tau1_pi_3_pz", "tau1_pi_3_e",
                           "tau2_nu_px",   "tau2_nu_py",   "tau2_nu_pz",   "tau2_nu_e",
                           "tau2_pi_1_px", "tau2_pi_1_py", "tau2_pi_1_pz", "tau2_pi_1_e"
                           "tau2_pi_2_px", "tau2_pi_2_py", "tau2_pi_2_pz", "tau2_pi_2_e",
                           "tau2_pi_3_px", "tau2_pi_3_py", "tau2_pi_3_pz", "tau2_pi_3_e"]




