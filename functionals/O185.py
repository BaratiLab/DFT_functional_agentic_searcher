from dftsr.api import *

GSS = 0.46914023462026644
X_TERMS = ((1, 0), (0, 1))
OS_TERMS = ((1, 0), (2, 0), (6, 0), (2, 1), (6, 1))


def series(w, u, terms, c):
    out = 0.0
    for k, (i, j) in enumerate(terms):
        out = out + c[k] * w ** i * u ** j
    return out


class Ablation_F145_OSwindow(Functional):
    family = 'rsh'

    def init_params(self):
        return dict(cx=[0.2707, 0.9761],
                    css=[-4.2424, -4.0976, -1.8109, 7.0580],
                    cos=[0.9739, 3.6648, -3.1708, -10.8518, 11.0724], brs=[-0.05, -0.05])

    def exchange(self, p, d):
        return exchange_from_fx(d, lambda ch: 0.85 + series(ch.w, b97_u(ch.x2, 0.004), X_TERMS, p['cx']), self.omega)

    def correlation(self, p, d):
        cs, b = p['css'], p['brs']

        def fss(ch):
            u, w = b97_u(ch.x2, GSS), ch.w
            return u + cs[0] * w + cs[1] * w ** 2 + cs[2] * u ** 6 + cs[3] * u ** 6 * w ** 4

        def fos(d):
            rs, w = d.rs, d.w_avg
            g = 1.0 + series(w, b97_u(d.x2_avg, 0.006), OS_TERMS, p['cos'])
            return g * (1 + (b[0] + b[1] * w ** 2) * rs / (1 + rs ** 3))
        return correlation_from_factors(d, fss, fos)
