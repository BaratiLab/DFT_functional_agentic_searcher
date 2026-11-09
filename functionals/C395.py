from dftsr.api import *
class BFunctional(Functional):
    family='rsh'
    def init_params(self):
        return {'cx': [1.0435360428200626, 0.20321155891619436], 'css': [0.7274211661294759, -4.41901851061404, -1.4556670356057029, 5.735990658405081, -0.20458796539440488], 'cos': [1.0538471711342332, 3.2745582398901556, -1.6451114527204427, 1.7698932792720867], 'root_shape': [0.21548992167904887], 'q': [0.332126814694281], 'kinetic_os': [2.477994532537782]}
    def exchange(self,p,d):
        return exchange_from_fx(d,lambda ch: 0.85+p['cx'][0]*b97_u(ch.x2,0.003840616724010807)+p['cx'][1]*ch.w,self.omega)
    def correlation(self,p,d):
        def fss(ch):
            u,w=b97_u(ch.x2,0.46914023462026644),ch.w
            z=(ch.tau_w/(ch.tau+1e-30)).clamp(0,1)
            c=p['css']
            poly=c[0]*u+c[1]*w*(1+w)+c[2]*u**6+c[3]*u**6*w**4+c[4]*z*(1-z)*(1-u)
            return poly
        def fos(d):
            w=(1+d.zeta)/2*d.a.w+(1-d.zeta)/2*d.b.w
            s=softplus(p['root_shape'][0])+1e-12
            r=(s+d.x2_avg)**(1/3)-s**(1/3)
            v=r/(1+0.006*r)
            c=p['cos']
            return (c[0]+c[1]*w**2*(1-w**4)+c[2]*w**2*v+c[3]*w**6*v)+p['kinetic_os'][0]*w*(1+w)/torch.sqrt(1+(softplus(p['q'][0])*d.rs)**2)
        return correlation_from_factors(d,fss,fos)
