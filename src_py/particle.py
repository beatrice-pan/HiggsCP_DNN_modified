import numpy as np


class Particle(object):
    def __init__(self, vec):
        if isinstance(vec, list):
            vec = np.vstack(vec).T
        self.vec = vec
        #self.x = vec[..., 0]
        #self.y = vec[..., 1]
        #self.z = vec[..., 2]
        #self.e = vec[..., 3]

    def __getitem__(self, key):
        return self.vec[key]

    @property
    def x(self):
        return self.vec[..., 0]
    @property
    def y(self):
        return self.vec[..., 1]
    @property
    def z(self):
        return self.vec[..., 2]
    @property
    def e(self):
        return self.vec[..., 3]

    @property
    def pt(self):
        return np.sqrt(self.x * self.x + self.y * self.y)

    # Invariant mass. If mass is negative then -sqrt(-mass) is returned
    @property
    def recalculated_mass(self):
        mass = np.sum(np.square(self.vec) * [1, 1, 1, -1], 1)
        return np.sqrt(np.abs(mass)) #* np.sign(mass)

    @property
    def angle_phi(self):
        return np.arctan2(self.y, self.x) + (self.y < 0) * 2 * np.pi

    @property
    def angle_theta(self):
        return - np.arctan2(self.x, self.z) * (2 * (self.x < 0) - 1)

    @property
    def length(self):
        lgth2 = np.square(self.vec[:,:3])
        lgth2 = np.sum(lgth2, 1)
        lgth = np.sqrt(lgth2)
        return lgth

    
    def scale_to_versor(self):
        p_len = np.sqrt(self.e*self.e - self.x * self.x - self.y * self.y - self.z * self.z)

        return Particle([
            self.x/p_len, self.y/p_len, self.z/p_len, self.e/p_len])

    def set(self, other):
        if isinstance(other, Particle):
            self.vec = other.vec
        elif isinstance(other, np.ndarray) and len(other.shape) == 2 and other.shape[1] == 4:
            self.vec = other

    def rotate_xz(self, theta):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        return Particle([
            cos_theta * self.x + sin_theta * self.z, self.y,
            -sin_theta * self.x + cos_theta * self.z, self.e])

    def rotate_xy(self, theta):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        return Particle([
            cos_theta * self.x - sin_theta * self.y,
            sin_theta * self.x + cos_theta * self.y,
            self.z, self.e])

    def boost_along_z(self, p_pz, p_e):
        m = np.sqrt(p_e * p_e - p_pz * p_pz)

        return Particle([
            self.x, self.y,
            (p_e * self.z + p_pz * self.e) / m,
            (p_pz * self.z + p_e * self.e) / m])

    def boost(self, p):
        p_len = np.sqrt(p.x * p.x + p.y * p.y + p.z * p.z)
        phi = p.angle_phi
        theta = p.rotate_xy(-phi).angle_theta

        ret = self.rotate_xy(-phi).rotate_xz(-theta)
        ret = ret.boost_along_z(-p_len, p.e)
        return ret.rotate_xz(theta).rotate_xy(phi)


    def scale_lifetime(self, scale):

        mass2 = self.e*self.e - self.x*self.x - self.y*self.y - self.z*self.z
        energy = np.sqrt(  self.x*self.x*scale*scale + self.y*self.y*scale*scale + self.z*self.z*scale*scale + mass2)
        return Particle([self.x*scale, self.y*scale, self.z*scale,  energy])     
    

    def sum(self, dim):
        return np.sum(self.vec, dim)

    def __add__(self, other):
        return Particle(self.vec + other.vec)

    def __radd__(self, other):
        return Particle(self.vec + other)

    def __mul__(self, other):
        return Particle(self.vec * other.vec)

    def __rmul__(self, other):
        return Particle(self.vec * other)

    def __divide__(self, other):
        return Particle(self.vec / other.vec)

    def __rdivide__(self, other):
        return Particle(self.vec / other)

    def __sub__(self, other):
        return Particle(self.vec - other.vec)

    def __rsub__(self, other):
        return Particle(self.vec - other)

    def __neg__(self, other):
        return Particle(-self.vec)


