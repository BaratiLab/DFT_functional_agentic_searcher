from dftsr.api import *

GSS = 0.46914023462026644
X_TERMS = ((1, 0), (0, 1))


def series(w, u, terms, c):
    out = 0.0
    for k, (i, j) in enumerate(terms):
        out = out + c[k] * w ** i * u ** j
    return out


class Ablation_GASosFree(Functional):
    family = 'rsh'

    def init_params(self):
        return dict(cx=[0.2561, 1.0218],
                    css=[-4.9695, -5.2652, -2.0334, 7.8336],
                    cos=[0.6053, 6.6259, -1.6902, -3.1688, 1.8788], brs=[-0.3172, -0.2449])

    def exchange(self, p, d):
        return exchange_from_fx(d, lambda ch: 0.85 + series(ch.w, b97_u(ch.x2, 0.004), X_TERMS, p['cx']), self.omega)

    def correlation(self, p, d):
        cs, b, co = p['css'], p['brs'], p['cos']

        def fss(ch):
            u, w = b97_u(ch.x2, GSS), ch.w
            rs = ch.rs
            h = rs / (1 + rs ** 3)
            g = u + cs[0] * w + cs[1] * w ** 2 + cs[2] * u ** 6 + cs[3] * u ** 6 * w ** 4
            return g * (1 + (b[0] + b[1] * w ** 2) * h)

        def fos(d):
            w = d.w_avg
            q = torch.pow(d.x2_avg + 1.0, 1.0 / 3.0) - 1.0
            return co[0] + co[1] * w ** 2 + co[2] * w ** 2 * q + co[3] * w ** 6 + co[4] * w ** 6 * q
        return correlation_from_factors(d, fss, fos)
