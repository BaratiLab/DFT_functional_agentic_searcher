from dftsr.api import *
class BFunctional(Functional):
    family='rsh'
    def init_params(self):
        return {'cx': [1.0099482333014884, 0.19886585955154787], 'css': [1.0960702831631148, -2.859011828441663, -4.616148732859335, -1.612989284574745, 5.95613948445327, -0.1807008106895367], 'cos': [0.8035925594049149, 5.765897330858385, -1.597654559078649, -4.742395901395269, 1.7789791228065948], 'root_shape': [1.119056597025386], 'cr': [3.078614384967419, 0.47798181803853845]}
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
            rden=softplus(p['cr'][1])*d.rs
            return c[0]+c[1]*w**2+c[2]*w**2*v+c[3]*w**6+c[4]*w**6*v+p['cr'][0]*w*(1+w)/torch.sqrt(1+rden**2)
        return correlation_from_factors(d,fss,fos)
