from dftsr.api import *

GSS = 0.46914023462026644
X_TERMS = ((1, 0), (0, 1))
OS_TERMS = ((1, 0), (2, 0), (6, 0), (2, 1), (6, 1))


def series(w, u, terms, c):
    out = 0.0
    for k, (i, j) in enumerate(terms):
        out = out + c[k] * w ** i * u ** j
    return out


class Ablation_F145_BroadWin(Functional):
    family = 'rsh'

    def init_params(self):
        return dict(cx=[0.2561, 1.0218],
                    css=[-4.9695, -5.2652, -2.0334, 7.8336],
                    cos=[1.1698, 3.7874, -3.3527, -10.5976, 10.3822], brs=[-0.3172, -0.2449])

    def exchange(self, p, d):
        return exchange_from_fx(d, lambda ch: 0.85 + series(ch.w, b97_u(ch.x2, 0.004), X_TERMS, p['cx']), self.omega)

    def correlation(self, p, d):
        cs, b = p['css'], p['brs']

        def fss(ch):
            u, w = b97_u(ch.x2, GSS), ch.w
            rs = ch.rs
            h = rs / (1 + rs ** 2)
            g = u + cs[0] * w + cs[1] * w ** 2 + cs[2] * u ** 6 + cs[3] * u ** 6 * w ** 4
            return g * (1 + (b[0] + b[1] * w ** 2) * h)
        fos = lambda d: 1.0 + series(d.w_avg, b97_u(d.x2_avg, 0.006), OS_TERMS, p['cos'])
        return correlation_from_factors(d, fss, fos)
