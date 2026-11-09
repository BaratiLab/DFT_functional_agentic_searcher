"""GAS22 (Ma et al., Sci. Adv. 2022) — evolved from ωB97M-V by SyFES; same RSH frame (ω=0.3, 0.15 SR-HF, VV10).

g_x   = c0 + c1 u_x + c2 w,                       u_x gamma = 0.003840616724010807
g_css = u - c1 w - c2 w^2 - c3 u^6 + c4 u^6 w^4,   u gamma = 0.46914023462026644  (coefficient of u fixed to 1)
g_cos = c0 + c1 w^2 + c2 w^2 x^{2/3} + c3 w^6 + c4 w^6 x^{2/3}   (x^2 = spin-averaged squared reduced gradient)
Initial values = published GAS22 parameters. Note g_x(UEG) = c0 = 0.862 (not exactly 0.85).
"""
import os
from dftsr.api import *
EPS = float(os.environ.get("GAS22_EPS", "1e-30"))


class GAS22(Functional):
    family = 'rsh'

    def init_params(self):
        return dict(cx=[0.862139736374172, 0.936993691972698, 0.317533683085033],
                    css=[-4.10753796482853, -5.24218990333846, -1.76643208454076, 7.5380689617542],
                    cos=[0.805124374375355, 7.98909430970845, -1.76098915061634, -7.54815900595292,
                         2.00093961824784])

    def exchange(self, p, d):
        c = p['cx']
        return exchange_from_fx(d, lambda ch: c[0] + c[1] * b97_u(ch.x2, 0.003840616724010807) + c[2] * ch.w,
                                self.omega)

    def correlation(self, p, d):
        cs, co = p['css'], p['cos']

        def fss(ch):
            u, w = b97_u(ch.x2, 0.46914023462026644), ch.w
            return u + cs[0] * w + cs[1] * w ** 2 + cs[2] * u ** 6 + cs[3] * u ** 6 * w ** 4

        def fos(d):
            w = d.w_avg
            x23 = torch.pow(d.x2_avg + EPS, 1.0 / 3.0)
            return co[0] + co[1] * w ** 2 + co[2] * w ** 2 * x23 + co[3] * w ** 6 + co[4] * w ** 6 * x23

        return correlation_from_factors(d, fss, fos)
