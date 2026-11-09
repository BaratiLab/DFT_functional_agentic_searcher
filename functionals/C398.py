from dftsr.api import *
class BFunctional(Functional):
    family='rsh'
    def init_params(self):
        return {'cx': [1.0402385496988753, 0.23633524855573618], 'css': [0.9400267515797734, -2.8398814125642473, -4.612414385720957, -1.5931886568004954, 7.6095881777056995, -0.1934287298114782], 'cos': [0.8944152384581956, 6.729407484752352, -1.5884815938736891, 1.7817538062663416], 'root_shape': [0.573997574093411], 'kinetic_os': [0.05]}
    def exchange(self,p,d):
        return exchange_from_fx(d,lambda ch: 0.85+p['cx'][0]*b97_u(ch.x2,0.003840616724010807)+p['cx'][1]*ch.w,self.omega)
    def correlation(self,p,d):
        def fss(ch):
            u,w=b97_u(ch.x2,0.46914023462026644),ch.w
            z=(ch.tau_w/(ch.tau+1e-30)).clamp(0,1)
            c=p['css']
            poly=c[0]*u+c[1]*w+c[2]*w**2+c[3]*u**6+c[4]*u**6*w**4+c[5]*z*(1-z)
            return poly
        def fos(d):
            w=(1+d.zeta)/2*d.a.w+(1-d.zeta)/2*d.b.w
            s=softplus(p['root_shape'][0])+1e-12
            r=(s+d.x2_avg)**(1/3)-s**(1/3)
            v=r/(1+0.006*r)
            c=p['cos']
            return c[0]+c[1]*w**2*(1-w**4)+c[2]*w**2*v+c[3]*w**6*v+p['kinetic_os'][0]*w*(1-b97_u(d.x2_avg,0.006))
        return correlation_from_factors(d,fss,fos)
