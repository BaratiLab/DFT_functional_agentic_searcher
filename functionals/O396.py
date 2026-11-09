from dftsr.api import *

GSS = 0.46914023462026644


def elf_gate(ch):
    al = torch.clamp(ch.alpha, min=0.0)
    a2 = al * al
    return 1 - 1 / (1 + a2)


class OSShellELF10(Functional):
    family = 'rsh'

    def init_params(self):
        return dict(cx=[0.2348, 0.9969],
                    css=[-4.4335, -1.9591, 7.6410],
                    cos=[3.5123, -10.5609], brs=[-0.2990], sb=[0.8154], so=[0.05])

    def exchange(self, p, d):
        return exchange_from_fx(d, lambda ch: 0.85 + p['cx'][0] * ch.w + p['cx'][1] * b97_u(ch.x2, 0.004), self.omega)

    def correlation(self, p, d):
        cs, b, co = p['css'], p['brs'], p['cos']

        def fss(ch):
            u, w = b97_u(ch.x2, GSS), ch.w
            rs = ch.rs
            h = rs / (1 + rs ** 4)
            g = u + cs[0] * w * (1 + w) + cs[1] * u ** 6 + cs[2] * u ** 6 * w ** 4
            shell = 100.0 * p['sb'][0] * h * elf_gate(ch) ** 3 * (1 - u) ** 8
            return g * (1 + b[0] * (1 + w ** 2) * h) + shell

        def fos(d):
            w, u = d.w_avg, b97_u(d.x2_avg, 0.006)
            w2 = w ** 2
            base = (1.0 + w) + w2 * (1 - w2 ** 2) * (co[0] + co[1] * u)
            # opposite-spin counterpart of the shell term (spin-averaged ELF gate and gradient)
            ga = torch.where(d.a.mask, elf_gate(d.a), 0.0)
            gb = torch.where(d.b.mask, elf_gate(d.b), 0.0)
            gate = 0.5 * (ga + gb)
            rs2 = d.rs * 1.2599210498948732
            hos = rs2 / (1 + rs2 ** 4)
            uavg = b97_u(d.x2_avg, GSS)
            return base + 100.0 * p['so'][0] * hos * gate ** 3 * (1 - uavg) ** 8
        return correlation_from_factors(d, fss, fos)
