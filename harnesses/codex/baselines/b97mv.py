"""B97M-V (Mardirossian & Head-Gordon 2015) — pure meta-GGA + VV10 ('mgga' family: no exact exchange).

g_x   = 1 + 1.308 u + 1.901 u^2 + 0.416 w + 3.070 w u      (c00 = 1 fixed by UEG)
g_css = 1 - 1.855 u^2 - 5.668 w - 20.497 w^3 u^2 - 20.364 w^4 u^2
g_cos = 1 + 1.573 u - 6.298 u^3 + 2.535 w - 6.427 w^3 u^2
"""
from dftsr.api import *

X_TERMS = ((0, 1), (0, 2), (1, 0), (1, 1))
SS_TERMS = ((0, 0), (0, 2), (1, 0), (3, 2), (4, 2))
OS_TERMS = ((0, 0), (0, 1), (0, 3), (1, 0), (3, 2))


def series(w, u, terms, c):
    out = 0.0
    for k, (i, j) in enumerate(terms):
        out = out + c[k] * w ** i * u ** j
    return out


class B97MV(Functional):
    family = 'mgga'

    def init_params(self):
        return dict(cx=[1.308, 1.901, 0.416, 3.070],
                    css=[1.000, -1.855, -5.668, -20.497, -20.364],
                    cos=[1.000, 1.573, -6.298, 2.535, -6.427])

    def exchange(self, p, d):
        return exchange_from_fx(d, lambda ch: 1.0 + series(ch.w, b97_u(ch.x2, 0.004), X_TERMS, p['cx']), self.omega)

    def correlation(self, p, d):
        fss = lambda ch: series(ch.w, b97_u(ch.x2, 0.2), SS_TERMS, p['css'])
        fos = lambda d: series(d.w_avg, b97_u(d.x2_avg, 0.006), OS_TERMS, p['cos'])
        return correlation_from_factors(d, fss, fos)
