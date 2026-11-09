from dftsr.api import *


class WB97ssTauPartOSlin1Pub(Functional):
    family = 'rsh'

    def init_params(self):
        # published wB97M-V coefficients as INITIAL values only (all refit); tied ss kinetic = mean of
        # published c10/c20; os u-pair tied at published c21; window small
        return dict(cx=[0.259, 1.007],
                    css=[0.443, -3.9625, 4.278, -1.437],
                    cos=[2.924, -1.39, -8.812],
                    brs=[-0.1])

    def exchange(self, p, d):
        return exchange_from_fx(d, lambda ch: 0.85 + p['cx'][0] * ch.w + p['cx'][1] * b97_u(ch.x2, 0.004), self.omega)

    def correlation(self, p, d):
        b, co, cs = p['brs'], p['cos'], p['css']

        def fss(ch):
            rs, w = ch.rs, ch.w
            u = b97_u(ch.x2, 0.2)
            h = rs / (1 + rs ** 4)
            w2 = w * w
            tau_part = cs[1] * w * (1 + w) + cs[2] * w2 * w2 * u ** 3
            return cs[0] + cs[3] * u ** 4 + tau_part * (1 + b[0] * (1 + w2) * h)

        def fos(d):
            w, u = d.w_avg, b97_u(d.x2_avg, 0.006)
            w2 = w * w
            return (1.0 + w) + co[0] * w2 + co[1] * w2 ** 3 + co[2] * w2 * u * (1 - w2 * w2)
        return correlation_from_factors(d, fss, fos)
