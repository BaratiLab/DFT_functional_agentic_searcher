from dftsr.api import *
class BFunctional(Functional):
    family='rsh'
    def init_params(self):
        return {'cx': [1.0394984008334365, 0.19790933468451527], 'css': [0.644301878760264, -4.836515664736259, -1.4224346225262894, 6.710142476464712, -0.20344874544667227], 'cos': [1.0800109354775256, 3.6216376048231993, -1.7149709407106475, -5.001649689858358, 1.8894251229524799], 'root_shape': [0.4183507078535885], 'q': [-0.47130693782172395], 'kinetic_os': [3.3012799037960656]}
    def exchange(self,p,d):
        return exchange_from_fx(d,lambda ch: 0.85+p['cx'][0]*b97_u(ch.x2,0.003840616724010807)+p['cx'][1]*ch.w,self.omega)
    def correlation(self,p,d):
        def fss(ch):
            u,w=b97_u(ch.x2,0.46914023462026644),ch.w
            z=(ch.tau_w/(ch.tau+1e-30)).clamp(0,1)
            c=p['css']
            poly=c[0]*u+c[1]*w*(1+w)+c[2]*u**6+c[3]*u**6*w**4+c[4]*z*(1-z)
            return poly
        def fos(d):
            w=(1+d.zeta)/2*d.a.w+(1-d.zeta)/2*d.b.w
            s=softplus(p['root_shape'][0])+1e-12
            r=(s+d.x2_avg)**(1/3)-s**(1/3)
            v=r/(1+0.006*r)
            c=p['cos']
            return (c[0]+c[1]*w**2+c[2]*w**2*v+c[3]*w**6+c[4]*w**6*v)+p['kinetic_os'][0]*w*(1+w)/torch.sqrt(1+(softplus(p['q'][0])*d.rs)**2)
        return correlation_from_factors(d,fss,fos)
