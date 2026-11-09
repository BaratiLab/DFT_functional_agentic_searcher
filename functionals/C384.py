from dftsr.api import *
class IslandA093(Functional):
    family='rsh'
    def init_params(self):
        return {'cx': [1.0329949226750874, 0.179046859336696], 'css': [0.9696623184827495, -4.32829416949567, -2.0666797358280524, 7.344032675399885], 'cos': [0.8928583705174385, 5.472615612958179, -0.2911908510202552, -6.222729290982356], 'smooth': [1.1889392582843643], 'kinetic_os': [3.078614384967419], 'sswindow': [-0.26790401394943353], 'density_os': [0.47798181803853845]}
    def exchange(self,p,d):
        def gx(ch):
            u,w=b97_u(ch.x2,0.003840616724010807),ch.w
            return 0.85+p['cx'][0]*u+p['cx'][1]*w
        return exchange_from_fx(d,gx,self.omega)
    def correlation(self,p,d):
        cs,co=p['css'],p['cos']
        def fss(ch):
            u,w=b97_u(ch.x2,0.46914023462026644),ch.w
            z=(ch.tau_w/(ch.tau+1e-30)).clamp(0,1)
            return cs[0]*u+cs[1]*w*(1+w)+cs[2]*u**6+cs[3]*u**6*w**4+p['sswindow'][0]*z*(1-z)
        def fos(d):
            w=(1+d.zeta)/2*d.a.w+(1-d.zeta)/2*d.b.w
            s=softplus(p['smooth'][0])+1e-8
            q=(d.x2_avg+s)**(1.0/3.0)-s**(1.0/3.0)
            v=q/(1+0.006*q)
            return co[0]+(co[1]*w**2+co[3]*w**6)*(1+co[2]*v)+p['kinetic_os'][0]*w*(1+w)/torch.sqrt(1+(softplus(p['density_os'][0])*d.rs)**2)
        return correlation_from_factors(d,fss,fos)
