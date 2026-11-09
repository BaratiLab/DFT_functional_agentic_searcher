"""PyTorch building blocks for B97-type (range-separated hybrid) meta-GGA functionals.

Conventions (identical to SyFES / Q-Chem / libxc for wB97M-V):
  * Energy densities e_* are per unit volume (libxc's eps * rho).
  * tau is the positive-definite KE density WITH the 1/2 factor: tau = 1/2 sum_i |grad phi_i|^2.
  * All quantities are spin-resolved; closed-shell systems simply have rho_a == rho_b.

Ported from google-research/symbolic_functionals (syfes/xc/{lda,gga,mgga,rsh}.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

PI = math.pi
# UEG kinetic prefactor for a single spin channel (tau has the 1/2 factor):
#   tau_UEG,sigma = C_TAU * rho_sigma^(5/3)
C_TAU = 3.0 / 10.0 * (6.0 * PI ** 2) ** (2.0 / 3.0)
C_X_LDA = -3.0 / 4.0 * (3.0 / PI) ** (1.0 / 3.0)        # e_x^LDA(unpolarized) = C_X_LDA * rho^(4/3)
FZETA_DEN = 2.0 ** (4.0 / 3.0) - 2.0

# Density threshold below which a spin channel is treated as empty.
RHO_THR = 1e-10
TAU_THR = 1e-14
SIGMA_THR = 1e-20

# PW92 with the "modified" (PW_MOD) constants used by libxc for B97M-V/wB97M-V
PW92 = {
    'unpol': dict(a=0.0310907, alpha1=0.21370, beta1=7.5957, beta2=3.5876, beta3=1.6382, beta4=0.49294),
    'pol': dict(a=0.01554535, alpha1=0.20548, beta1=14.1189, beta2=6.1977, beta3=3.3662, beta4=0.62517),
    'alpha': dict(a=0.0168869, alpha1=0.11125, beta1=10.357, beta2=3.6231, beta3=0.88026, beta4=0.49671),
    'fpp0': 1.709920934161365617563962776245,
}

# wB97M-V reference parameters: (w power, u power, coefficient)
WB97MV = {
    'x': ((0, 0, 0.85), (1, 0, 0.259), (0, 1, 1.007)),
    'ss': ((0, 0, 0.443), (1, 0, -4.535), (2, 0, -3.39), (4, 3, 4.278), (0, 4, -1.437)),
    'os': ((0, 0, 1.000), (1, 0, 1.358), (2, 0, 2.924), (6, 0, -1.39), (2, 1, -8.812), (6, 1, 9.142)),
    'gamma_x': 0.004, 'gamma_ss': 0.2, 'gamma_os': 0.006,
    'omega': 0.3, 'cx_sr': 0.15,  # short-range exact exchange fraction; long-range = 1.0
}
B97MV = {
    'x': ((0, 0, 1.000), (0, 1, 1.308), (0, 2, 1.901), (1, 0, 0.416), (1, 1, 3.070)),
    'ss': ((0, 0, 1.000), (0, 2, -1.855), (1, 0, -5.668), (3, 2, -20.497), (4, 2, -20.364)),
    'os': ((0, 0, 1.000), (0, 1, 1.573), (0, 3, -6.298), (1, 0, 2.535), (3, 2, -6.427)),
    'gamma_x': 0.004, 'gamma_ss': 0.2, 'gamma_os': 0.006,
    'omega': 0.0, 'cx_sr': 0.0,
}


# ----------------------------------------------------------------------------- LDA pieces
def pw92_g(rs, a, alpha1, beta1, beta2, beta3, beta4):
    """PW92 G(rs) (PRB 45, 13244, Eq. 10)."""
    srs = torch.sqrt(rs)
    den = 2 * a * (beta1 * srs + beta2 * rs + beta3 * rs * srs + beta4 * rs * rs)
    return -2 * a * (1 + alpha1 * rs) * torch.log1p(1 / den)


def wigner_seitz_radius(rho):
    return (3.0 / (4.0 * PI)) ** (1.0 / 3.0) * rho.clamp_min(RHO_THR) ** (-1.0 / 3.0)


def f_zeta(zeta):
    """von Barth-Hedin spin interpolation f(zeta) in [0, 1]."""
    return ((1 + zeta).clamp_min(0) ** (4.0 / 3.0) + (1 - zeta).clamp_min(0) ** (4.0 / 3.0) - 2) / FZETA_DEN


def eps_c_pw92(rs, zeta):
    """PW92 correlation energy per particle."""
    ec0 = pw92_g(rs, **PW92['unpol'])
    ec1 = pw92_g(rs, **PW92['pol'])
    ac = -pw92_g(rs, **PW92['alpha'])
    fz = f_zeta(zeta)
    z4 = zeta ** 4
    return ec0 + ac * fz * (1 - z4) / PW92['fpp0'] + (ec1 - ec0) * fz * z4


def e_c_pw92(rho_a, rho_b):
    rho = rho_a + rho_b
    rs = wigner_seitz_radius(rho)
    zeta = ((rho_a - rho_b) / rho.clamp_min(RHO_THR)).clamp(-1, 1)
    return rho * eps_c_pw92(rs, zeta)


def e_c_pw92_fully_polarized(rho_s):
    """PW92 correlation energy density of a fully spin-polarized density rho_s (zeta = 1)."""
    rs = wigner_seitz_radius(rho_s)
    return rho_s * pw92_g(rs, **PW92['pol'])


def e_x_lda_spin(rho_s):
    """LDA exchange energy density of one spin channel: 1/2 e_x^LDA(2 rho_s)."""
    return 0.5 * C_X_LDA * (2 * rho_s) ** (4.0 / 3.0)


def f_rsh(rho_s, omega):
    """Short-range attenuation factor for LDA exchange of one spin channel (wB97M-V paper Eq. 11).

    a = omega / k_F,sigma with k_F,sigma = (6 pi^2 rho_s)^(1/3).
    Returns 1 when omega == 0.
    """
    if omega == 0:
        return torch.ones_like(rho_s)
    kf = (6 * PI ** 2 * rho_s.clamp_min(RHO_THR)) ** (1.0 / 3.0)
    a = omega / kf
    # For large a (very low density) use the asymptotic series to avoid cancellation:
    #   F(a) ~ 1/(36 a^2) - 1/(960 a^4) + ...
    ia2 = 1 / (a * a)
    exact = 1 - 2.0 / 3.0 * a * (2 * math.sqrt(PI) * torch.erf(1 / a) - 3 * a + a ** 3
                                 + (2 * a - a ** 3) * torch.exp(-ia2))
    series = ia2 / 36 - ia2 * ia2 / 960 + ia2 ** 3 / 26880
    return torch.where(a < 5.0, exact, series)


# ----------------------------------------------------------------------------- descriptors
def b97_u(x2, gamma):
    """B97 finite-domain transform u = gamma x^2 / (1 + gamma x^2) in [0, 1)."""
    gx = gamma * x2
    return gx / (1 + gx)


def b97_w(t):
    """w = (t - 1)/(t + 1) in [-1, 1], t = tau_UEG / tau."""
    return (t - 1) / (t + 1)


def power_series(w, u, terms, coeffs=None):
    """sum_k c_k w^i_k u^j_k.  `terms` is ((i, j, c), ...); if coeffs given, it overrides c."""
    out = torch.zeros_like(w)
    for k, (i, j, c) in enumerate(terms):
        ck = c if coeffs is None else coeffs[k]
        out = out + ck * (w ** i if i else 1.0) * (u ** j if j else 1.0)
    return out


@dataclass
class SpinChannel:
    """All per-spin-channel quantities a functional may use (one spin sigma)."""
    rho: torch.Tensor      # rho_sigma
    sigma: torch.Tensor    # |grad rho_sigma|^2
    tau: torch.Tensor      # tau_sigma (with 1/2)
    mask: torch.Tensor     # True where rho_sigma > RHO_THR

    @property
    def x2(self):
        """Squared reduced gradient x_sigma^2 = |grad rho_s|^2 / rho_s^(8/3)  (B97 'x', a.k.a. paper 's')."""
        return self.sigma / self.rho ** (8.0 / 3.0)

    @property
    def s2_pbe(self):
        """Squared PBE-style reduced gradient s^2 = x^2 / (4 (6 pi^2)^(2/3))."""
        return self.x2 / (4 * (6 * PI ** 2) ** (2.0 / 3.0))

    @property
    def tau_ueg(self):
        return C_TAU * self.rho ** (5.0 / 3.0)

    @property
    def tau_w(self):
        """von Weizsaecker KE density sigma / (8 rho)."""
        return self.sigma / (8 * self.rho)

    @property
    def t(self):
        """t = tau_UEG / tau (B97M-V convention)."""
        return self.tau_ueg / self.tau

    @property
    def w(self):
        return b97_w(self.t)

    @property
    def alpha(self):
        """Iso-orbital indicator alpha = (tau - tau_W)/tau_UEG >= 0."""
        return ((self.tau - self.tau_w) / self.tau_ueg).clamp_min(0)

    @property
    def rs(self):
        """Wigner-Seitz radius of this spin channel's density (as if total rho = rho_s)."""
        return wigner_seitz_radius(self.rho)

    @property
    def e_x_lda(self):
        return e_x_lda_spin(self.rho)

    @property
    def e_c_lda_ss(self):
        return e_c_pw92_fully_polarized(self.rho)


@dataclass
class Density:
    """Spin-resolved density on a set of grid points.

    Raw inputs (float tensors of shape [N]): rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb, tau_a, tau_b.
    """
    rho_a: torch.Tensor
    rho_b: torch.Tensor
    sigma_aa: torch.Tensor
    sigma_ab: torch.Tensor
    sigma_bb: torch.Tensor
    tau_a: torch.Tensor
    tau_b: torch.Tensor
    closed: bool = False   # True => rho_a == rho_b everywhere (closed shell); channel b aliases channel a
    _cache: dict = field(default_factory=dict, repr=False)

    @staticmethod
    def _channel(rho, sigma, tau):
        mask = rho > RHO_THR
        r = torch.where(mask, rho, torch.full_like(rho, 1.0))
        # keep tau >= tau_W and > 0, sigma >= 0 (bounds enforced as in libxc)
        s = torch.where(mask, sigma.clamp_min(SIGMA_THR), torch.zeros_like(sigma))
        tw = s / (8 * r)
        t = torch.where(mask, torch.maximum(tau.clamp_min(TAU_THR), tw), torch.full_like(tau, C_TAU))
        return SpinChannel(r, s, t, mask)

    @property
    def a(self) -> SpinChannel:
        if 'a' not in self._cache:
            self._cache['a'] = self._channel(self.rho_a, self.sigma_aa, self.tau_a)
        return self._cache['a']

    @property
    def b(self) -> SpinChannel:
        if self.closed:
            return self.a
        if 'b' not in self._cache:
            self._cache['b'] = self._channel(self.rho_b, self.sigma_bb, self.tau_b)
        return self._cache['b']

    def channels(self):
        return (self.a, self.b)

    @property
    def rho(self):
        return self.rho_a + self.rho_b

    @property
    def zeta(self):
        return ((self.rho_a - self.rho_b) / self.rho.clamp_min(RHO_THR)).clamp(-1, 1)

    @property
    def rs(self):
        return wigner_seitz_radius(self.rho)

    @property
    def sigma(self):
        """|grad rho|^2 of the total density."""
        return (self.sigma_aa + 2 * self.sigma_ab + self.sigma_bb).clamp_min(0)

    @property
    def tau(self):
        return self.tau_a + self.tau_b

    def e_c_lda_decomposed(self):
        """Stoll partition of PW92 correlation: (e_c_aa, e_c_bb, e_c_ab)."""
        if 'ec' not in self._cache:
            ra = torch.where(self.a.mask, self.rho_a, torch.zeros_like(self.rho_a))
            rb = torch.where(self.b.mask, self.rho_b, torch.zeros_like(self.rho_b))
            e_tot = e_c_pw92(ra, rb)
            e_aa = torch.where(self.a.mask, e_c_pw92_fully_polarized(self.a.rho), torch.zeros_like(ra))
            e_bb = torch.where(self.b.mask, e_c_pw92_fully_polarized(self.b.rho), torch.zeros_like(rb))
            self._cache['ec'] = (e_aa, e_bb, e_tot - e_aa - e_bb)
        return self._cache['ec']

    # Spin-averaged descriptors used by B97M-V-type opposite-spin correlation
    @property
    def x2_avg(self):
        return 0.5 * (torch.where(self.a.mask, self.a.x2, 0.0) + torch.where(self.b.mask, self.b.x2, 0.0))

    @property
    def t_avg(self):
        ta = torch.where(self.a.mask, self.a.t, 0.0)
        tb = torch.where(self.b.mask, self.b.t, 0.0)
        return 0.5 * (ta + tb)

    @property
    def w_avg(self):
        return b97_w(self.t_avg)

    def scaled(self, lam: float) -> 'Density':
        """Uniform coordinate scaling rho_lam(r) = lam^3 rho(lam r)."""
        return Density(self.rho_a * lam ** 3, self.rho_b * lam ** 3,
                       self.sigma_aa * lam ** 8, self.sigma_ab * lam ** 8, self.sigma_bb * lam ** 8,
                       self.tau_a * lam ** 5, self.tau_b * lam ** 5, closed=self.closed)

    def swapped(self) -> 'Density':
        return Density(self.rho_b, self.rho_a, self.sigma_bb, self.sigma_ab, self.sigma_aa,
                       self.tau_b, self.tau_a)


# ----------------------------------------------------------------------------- reference functional
def e_xc_b97m_family(d: Density, P=WB97MV, cx=None, css=None, cos=None):
    """Semilocal XC energy density of a wB97M-V / B97M-V form (exact exchange & VV10 excluded)."""
    e = torch.zeros_like(d.rho_a)
    eaa, ebb, eab = d.e_c_lda_decomposed()
    for ch, ec_ss in ((d.a, eaa), (d.b, ebb)):
        w = ch.w
        fx = power_series(w, b97_u(ch.x2, P['gamma_x']), P['x'], cx)
        fss = power_series(w, b97_u(ch.x2, P['gamma_ss']), P['ss'], css)
        ex = ch.e_x_lda * f_rsh(ch.rho, P['omega']) * fx
        e = e + torch.where(ch.mask, ex + ec_ss * fss, 0.0)
    fos = power_series(d.w_avg, b97_u(d.x2_avg, P['gamma_os']), P['os'], cos)
    return e + eab * fos
