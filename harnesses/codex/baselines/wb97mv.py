"""ωB97M-V (Mardirossian & Head-Gordon 2016) semilocal part — the baseline / seed functional.

g_x   = 0.85 + c_x10 w + c_x01 u_x                        (c_x00 = 0.85 fixed by the UEG constraint)
g_css = c00 + c10 w + c20 w^2 + c43 w^4 u^3 + c04 u^4      (per spin, u with gamma_ss = 0.2)
g_cos = c00 + c10 w + c20 w^2 + c60 w^6 + c21 w^2 u + c61 w^6 u   (spin-averaged w, u with gamma_os = 0.006)
12 linear parameters; gamma_x = 0.004, gamma_ss = 0.2, gamma_os = 0.006 fixed.
"""
from dftsr.api import *

X_TERMS = ((1, 0), (0, 1))
SS_TERMS = ((0, 0), (1, 0), (2, 0), (4, 3), (0, 4))
OS_TERMS = ((0, 0), (1, 0), (2, 0), (6, 0), (2, 1), (6, 1))


def series(w, u, terms, c):
    out = 0.0
    for k, (i, j) in enumerate(terms):
        out = out + c[k] * w ** i * u ** j
    return out


class WB97MV(Functional):
    family = 'rsh'

    def init_params(self):
        return dict(cx=[0.259, 1.007],
                    css=[0.443, -4.535, -3.39, 4.278, -1.437],
                    cos=[1.000, 1.358, 2.924, -1.39, -8.812, 9.142])

    def exchange(self, p, d):
        return exchange_from_fx(d, lambda ch: 0.85 + series(ch.w, b97_u(ch.x2, 0.004), X_TERMS, p['cx']), self.omega)

    def correlation(self, p, d):
        fss = lambda ch: series(ch.w, b97_u(ch.x2, 0.2), SS_TERMS, p['css'])
        fos = lambda d: series(d.w_avg, b97_u(d.x2_avg, 0.006), OS_TERMS, p['cos'])
        return correlation_from_factors(d, fss, fos)
