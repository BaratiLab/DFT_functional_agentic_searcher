from dftsr.api import *
class BFunctional(Functional):
    family='rsh'
    def init_params(self):
        return {'cx': [1.026595289618704, 0.18548933238816442], 'css': [0.7209853587343611, -4.341883996438071, -1.4336764799378352, 6.7154025681485185, -0.24681862672549493], 'cos': [1.0339888081905861, 3.9800015581234045, -1.6238384845307006, -5.590958185259198, 1.798100537604004], 'root_shape': [0.5693257037760272], 'q': [-0.16518024213142027], 'kinetic_os': [2.6600819445490407]}
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
            return (c[0]+c[1]*w**2+c[2]*w**2*v+c[3]*w**6+c[4]*w**6*v)+p['kinetic_os'][0]*w*(1+w)/(1+softplus(p['q'][0])*d.rs)
        return correlation_from_factors(d,fss,fos)
