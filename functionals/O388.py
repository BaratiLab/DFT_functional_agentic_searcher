from dftsr.api import *

GSS = 0.46914023462026644


class H_ShellELF_BetaTie9(Functional):
    family = 'rsh'

    def init_params(self):
        return dict(cx=[0.259, 1.007],
                    css=[-4.675, -1.766, 7.538],
                    cos=[2.924, -8.977], brs=[-0.25], sb=[1.0])

    def exchange(self, p, d):
        return exchange_from_fx(d, lambda ch: 0.85 + p['cx'][0] * ch.w + p['cx'][1] * b97_u(ch.x2, 0.004), self.omega)

    def correlation(self, p, d):
        cs, b, co = p['css'], p['brs'], p['cos']

        def fss(ch):
            u, w = b97_u(ch.x2, GSS), ch.w
            rs = ch.rs
            h = rs / (1 + rs ** 4)
            g = u + cs[0] * w * (1 + w) + cs[1] * u ** 6 + cs[2] * u ** 6 * w ** 4
            # shell-boundary term gated by the electron-localisation function: 1 - ELF = alpha^2/(1+alpha^2)
            al = torch.clamp(ch.alpha, min=0.0)
            a2 = al * al
            shell = 100.0 * p['sb'][0] * h * (1 - 1 / (1 + a2)) ** 3 * (1 - u) ** 8
            return g * (1 + b[0] * (1 + w ** 2) * h) + shell

        def fos(d):
            w, u = d.w_avg, b97_u(d.x2_avg, 0.006)
            w2 = w ** 2
            return (1.0 + w) + w2 * (1 - w2 ** 2) * (co[0] + co[1] * u)
        return correlation_from_factors(d, fss, fos)
