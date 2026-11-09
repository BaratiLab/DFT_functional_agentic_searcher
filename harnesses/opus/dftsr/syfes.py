"""SyFES port (Ma, Narayanaswamy, Riley, Li, Sci. Adv. 8, eabq0279 (2022)): symbolic functional forms as instruction
lists, the SyFES mutation rules, functional-equivalence fingerprints and the GAS22 seed form.

Faithful to google-research/symbolic_functionals/syfes (symbolic/instructions.py, enhancement_factors.py,
mutators.py, xc_functionals.py), re-implemented without JAX so it runs in this project's .venv:
  * a form = 3 enhancement factors (f_x, f_css, f_cos); each = features ('x2', 'w'), shared parameters, variables
    (incl. 'enhancement_factor') and an instruction list. Same JSON layout as SyFES `XCFunctional.to_dict()`.
  * e_xc = e_x^LDA-SR * F_x(x2_s, w_s) [per spin] + e_css^LDA * F_css(x2_s, w_s) [per spin]
           + e_cos^LDA * F_cos(x2_avg, w_avg)
  * instruction pool probabilities = SI Section 3 (arithmetic 0.06 x5, power 0.05 x6, UTransform 0.1,
    PBEX/RPBEX/B88X/PBEC 0.075 each); mutation pool = 4 rules x 0.25; component probabilities 1/3 each;
    max 2 bound parameters per enhancement factor (example/config.py).
  * the same code evaluates a form on numpy arrays (fingerprints, CPU checks) and on torch tensors (GPU).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np

# ============================================================================ instructions
# name -> (number of inputs, bound parameter names)
INSTRUCTIONS = {
    'AdditionBy1Instruction': (1, []),
    'Power2Instruction': (1, []),
    'Power3Instruction': (1, []),
    'Power4Instruction': (1, []),
    'Power6Instruction': (1, []),
    'SquareRootInstruction': (1, []),
    'CubeRootInstruction': (1, []),
    'Log1PInstruction': (1, []),
    'ExpInstruction': (1, []),
    'UTransformInstruction': (1, ['gamma_utransform']),
    'PBEXInstruction': (1, ['kappa_pbex', 'mu_pbex']),
    'RPBEXInstruction': (1, ['kappa_rpbex', 'mu_rpbex']),
    'B88XInstruction': (1, ['beta_b88x']),
    'AdditionInstruction': (2, []),
    'SubtractionInstruction': (2, []),
    'MultiplicationInstruction': (2, []),
    'MultiplicationAdditionInstruction': (2, []),
    'DivisionInstruction': (2, []),
    'PBECInstruction': (2, ['beta_pbec', 'gamma_pbec']),
}
BOUND_PARAMETER_ASSOCIATION = {b: name for name, (_, bs) in INSTRUCTIONS.items() for b in bs}

# SyFES SI, Section 3: probabilities of the instruction pool used for the evolutions starting from (ω)B97M-V
INSTRUCTION_POOL = {
    **{n: 0.06 for n in ('AdditionInstruction', 'SubtractionInstruction', 'MultiplicationInstruction',
                         'DivisionInstruction', 'MultiplicationAdditionInstruction')},
    **{n: 0.05 for n in ('Power2Instruction', 'Power3Instruction', 'Power4Instruction', 'Power6Instruction',
                         'SquareRootInstruction', 'CubeRootInstruction')},
    'UTransformInstruction': 0.1,
    **{n: 0.075 for n in ('PBEXInstruction', 'RPBEXInstruction', 'B88XInstruction', 'PBECInstruction')},
}
MUTATION_POOL = {'insert_instruction': 0.25, 'remove_instruction': 0.25, 'replace_instruction': 0.25,
                 'change_argument': 0.25}
COMPONENTS = ('f_x', 'f_css', 'f_cos')
MAX_NUM_BOUND_PARAMETERS = 2

PW_PBE = {  # syfes/xc/lda.py _LDA_PW_PBE_PARAMETERS['eps_c_unpolarized'] (used by PBECInstruction)
    'a': 0.0310907, 'alpha1': 0.21370, 'beta1': 7.5957, 'beta2': 3.5876, 'beta3': 1.6382, 'beta4': 0.49294}
EPSILON = 1e-50


class _NP:
    """numpy backend for instruction semantics (float64)."""
    pi = math.pi
    sqrt = staticmethod(np.sqrt)
    cbrt = staticmethod(np.cbrt)
    log1p = staticmethod(np.log1p)
    log = staticmethod(np.log)
    exp = staticmethod(np.exp)
    arcsinh = staticmethod(np.arcsinh)


def _torch_backend():
    import torch

    class _T:
        pi = math.pi
        sqrt = staticmethod(torch.sqrt)
        log1p = staticmethod(torch.log1p)
        log = staticmethod(torch.log)
        exp = staticmethod(torch.exp)
        arcsinh = staticmethod(torch.asinh)

        @staticmethod
        def cbrt(x):  # real cube root (numpy/jax cbrt semantics, defined for negative x)
            return torch.sign(x) * torch.abs(x) ** (1.0 / 3.0)
    return _T


def _sqrt_any(xp, x):
    return xp.sqrt(x)


def _f_x_pbe(xp, x, kappa, mu):
    s = (1 / (2 * (3 * math.pi ** 2) ** (1 / 3))) * x
    return 1 + kappa - kappa / (1 + mu * s ** 2 / kappa)


def _f_x_rpbe(xp, x, kappa, mu):
    s = (1 / (2 * (3 * math.pi ** 2) ** (1 / 3))) * x
    return 1 + kappa * (1 - xp.exp(- mu * s ** 2 / kappa))


def _f_x_b88(xp, x, beta):
    c1 = 8 * (1 / 2) ** (2 / 3) * (math.pi / 3) ** (1 / 3) / 3 * beta
    c2 = 6 * 2 ** (1 / 3) * beta
    return 1 + c1 * x ** 2 / (1 + c2 * x * xp.arcsinh(2 ** (1 / 3) * x))


def _e_c_pbe_unpolarized(xp, rho, sigma, beta, gamma):
    p = PW_PBE
    rs = (3 / (4 * math.pi)) ** (1 / 3) * (rho + EPSILON) ** (-1 / 3)
    den = 2 * p['a'] * (p['beta1'] * rs ** 0.5 + p['beta2'] * rs + p['beta3'] * rs ** 1.5 + p['beta4'] * rs ** 2)
    e_c_lda = rho * (-2 * p['a'] * (1 + p['alpha1'] * rs) * xp.log(1 + 1 / den))
    eps_c_lda = e_c_lda / (rho + EPSILON)
    t = (math.pi ** 0.5 / (4 * (3 * math.pi ** 2) ** (1 / 6))) * _sqrt_any(xp, sigma) / rho ** (7 / 6)
    a = beta / (gamma * (xp.exp(- eps_c_lda / gamma) - 1))
    at2 = a * t ** 2
    h = gamma * xp.log(1 + beta / gamma * t ** 2 * (1 + at2) / (1 + at2 + at2 ** 2))
    return e_c_lda + rho * h


def apply_instruction(ins, ws, xp):
    """Apply one instruction [name, out, in1, (in2)] to workspace dict `ws` (SyFES semantics)."""
    name, out, args = ins[0], ins[1], ins[2:]
    a = ws[args[0]]
    if name == 'AdditionBy1Instruction':
        ws[out] = a + 1.
    elif name == 'Power2Instruction':
        ws[out] = a ** 2
    elif name == 'Power3Instruction':
        ws[out] = a ** 3
    elif name == 'Power4Instruction':
        ws[out] = a ** 4
    elif name == 'Power6Instruction':
        ws[out] = a ** 6
    elif name == 'SquareRootInstruction':
        ws[out] = _sqrt_any(xp, a)
    elif name == 'CubeRootInstruction':
        ws[out] = xp.cbrt(a)
    elif name == 'Log1PInstruction':
        ws[out] = xp.log1p(a)
    elif name == 'ExpInstruction':
        ws[out] = xp.exp(a)
    elif name == 'UTransformInstruction':
        tmp = ws['gamma_utransform'] * a
        ws[out] = tmp / (1. + tmp)
    elif name == 'PBEXInstruction':
        ws[out] = _f_x_pbe(xp, a, ws['kappa_pbex'], ws['mu_pbex'])
    elif name == 'RPBEXInstruction':
        ws[out] = _f_x_rpbe(xp, a, ws['kappa_rpbex'], ws['mu_rpbex'])
    elif name == 'B88XInstruction':
        ws[out] = _f_x_b88(xp, a, ws['beta_b88x'])
    else:
        b = ws[args[1]]
        if name == 'AdditionInstruction':
            ws[out] = a + b
        elif name == 'SubtractionInstruction':
            ws[out] = a - b
        elif name == 'MultiplicationInstruction':
            ws[out] = a * b
        elif name == 'MultiplicationAdditionInstruction':
            ws[out] = ws[out] + a * b
        elif name == 'DivisionInstruction':
            ws[out] = a / b
        elif name == 'PBECInstruction':
            ws[out] = _e_c_pbe_unpolarized(xp, a, b, ws['beta_pbec'], ws['gamma_pbec'])
        else:
            raise ValueError(f'unknown instruction {name}')


# ============================================================================ enhancement factors / forms
def ef_bound_parameters(ef):
    return sorted({b for ins in ef['instructions'] for b in INSTRUCTIONS[ins[0]][1]})


def ef_parameter_names(ef):
    return list(ef['shared_parameter_names']) + ef_bound_parameters(ef)


def ef_used_parameter_names(ef):
    used = {a for ins in ef['instructions'] for a in ins[2:]}
    return [p for p in ef['shared_parameter_names'] if p in used] + ef_bound_parameters(ef)


def ef_allowed_inputs(ef):
    return list(ef['feature_names']) + list(ef['shared_parameter_names']) + list(ef['variable_names'])


def validate_ef(ef):
    if 'enhancement_factor' not in ef['variable_names']:
        raise ValueError('"enhancement_factor" not in variable_names')
    allowed = set(ef_allowed_inputs(ef))
    if len(allowed) != len(ef_allowed_inputs(ef)):
        raise ValueError('repeated names')
    for ins in ef['instructions']:
        n_in, _ = INSTRUCTIONS[ins[0]]
        if len(ins) != 2 + n_in:
            raise ValueError(f'wrong number of arguments: {ins}')
        if ins[1] not in ef['variable_names']:
            raise ValueError(f'invalid output {ins}')
        for a in ins[2:]:
            if a not in allowed:
                raise ValueError(f'invalid input {ins}')
    if len(ef_bound_parameters(ef)) > MAX_NUM_BOUND_PARAMETERS:
        raise ValueError('too many bound parameters')


def eval_ef(ef, features: dict, params: dict, xp=_NP, zero=None):
    """Enhancement factor on grids. `params` maps parameter name -> value (scalar or broadcastable array).
    Variables start at `zero` (numpy float64 0 by default, a 0-d tensor for torch), as in SyFES (np.array(0.))."""
    if zero is None:
        zero = np.float64(0.0)
        params = {k: np.float64(v) if np.ndim(v) == 0 else v for k, v in params.items()}
    ws = {**features, **params, **{v: zero for v in ef['variable_names']}}
    for ins in ef['instructions']:
        apply_instruction(ins, ws, xp)
    return ws['enhancement_factor']


def parameter_layout(form, used_only=True):
    """Ordered list of (component, parameter name) = the CMA-ES parameter vector of a form."""
    out = []
    for c in COMPONENTS:
        names = ef_used_parameter_names(form[c]) if used_only else sorted(ef_parameter_names(form[c]))
        out += [(c, n) for n in names]
    return out


def split_params(form, theta, layout=None):
    """theta[..., k] -> {component: {name: theta[..., k]}}."""
    layout = layout or parameter_layout(form)
    out = {c: {} for c in COMPONENTS}
    for k, (c, n) in enumerate(layout):
        out[c][n] = theta[..., k]
    return out


def form_key(form):
    return json.dumps({c: form[c] for c in COMPONENTS}, sort_keys=True)


def form_to_text(form):
    """Readable listing of the instruction lists."""
    lines = []
    for c in COMPONENTS:
        lines.append(f'{c}:')
        for ins in form[c]['instructions']:
            lines.append('  ' + _ins_str(ins))
    return '\n'.join(lines)


def _ins_str(ins):
    n, o, a = ins[0], ins[1], ins[2:]
    sym = {'AdditionInstruction': '+', 'SubtractionInstruction': '-', 'MultiplicationInstruction': '*',
           'DivisionInstruction': '/'}
    if n in sym:
        return f'{o} = {a[0]} {sym[n]} {a[1]}'
    if n == 'MultiplicationAdditionInstruction':
        return f'{o} += {a[0]} * {a[1]}'
    if n.startswith('Power'):
        return f'{o} = {a[0]} ** {n[5]}'
    short = {'AdditionBy1Instruction': '{a} + 1', 'SquareRootInstruction': 'sqrt({a})',
             'CubeRootInstruction': 'cbrt({a})', 'Log1PInstruction': 'log(1 + {a})', 'ExpInstruction': 'exp({a})',
             'UTransformInstruction': 'UTransform({a}; gamma_utransform)', 'PBEXInstruction': 'PBEX({a})',
             'RPBEXInstruction': 'RPBEX({a})', 'B88XInstruction': 'B88X({a})'}
    if n in short:
        return f'{o} = ' + short[n].format(a=a[0])
    return f'{o} = {n.replace("Instruction", "")}({", ".join(a)})'


def symbolic_expression(form):
    """sympy expressions of the three enhancement factors (u = UTransform(x2), etc. kept as functions)."""
    import sympy

    class _S:
        pi = sympy.pi
        sqrt = staticmethod(sympy.sqrt)
        cbrt = staticmethod(lambda x: x ** sympy.Rational(1, 3))
        log1p = staticmethod(lambda x: sympy.log(1 + x))
        log = staticmethod(sympy.log)
        exp = staticmethod(sympy.exp)
        arcsinh = staticmethod(sympy.asinh)
    out = {}
    for c in COMPONENTS:
        ef = form[c]
        ws = {f: sympy.Symbol(f) for f in ef['feature_names']}
        ws.update({p: sympy.Symbol(p) for p in ef_parameter_names(ef)})
        ws.update({v: sympy.Integer(0) for v in ef['variable_names']})
        try:
            for ins in ef['instructions']:
                apply_instruction(ins, ws, _S)
            out[c] = str(ws['enhancement_factor'])
        except Exception as e:  # pragma: no cover
            out[c] = f'<{e!r}>'
    return out


# ============================================================================ fingerprints (functional equivalence)
_SIG = {'rho': np.array([10., 1., 1.e-1, 1.e-2, 1.e-3, 1.e-4, 1.e-5]), 'x2': np.array([10., 100., 1000.]),
        'w': np.array([-1., -0.5, 0., 0.5]), 'u': np.array([0.0, 0.5, 0.9])}


def _pw_g(rs, a, alpha1, beta1, beta2, beta3, beta4):
    den = 2 * a * (beta1 * rs ** 0.5 + beta2 * rs + beta3 * rs ** 1.5 + beta4 * rs ** 2)
    return -2 * a * (1 + alpha1 * rs) * np.log(1 + 1 / den)


def _lda_signature_prefactors():
    """SyFES: e_x^LDA(2 rho), and the Stoll split of PW92 e_c(2 rho) (rho = rho_sigma)."""
    from .xc_torch import PW92
    rho = 2 * _SIG['rho']
    e_x = -3 / 4 * (3 / np.pi) ** (1 / 3) * rho ** (4 / 3)
    rs = (3 / (4 * np.pi)) ** (1 / 3) * rho ** (-1 / 3)
    e_c = rho * _pw_g(rs, **PW92['unpol'])
    rs_s = (3 / (4 * np.pi)) ** (1 / 3) * (rho / 2) ** (-1 / 3)
    e_ss = 2 * (rho / 2) * _pw_g(rs_s, **PW92['pol'])
    return e_x, e_ss, e_c - e_ss


_SIG_E = None


def fingerprint(form, num_feature_samples=10, num_parameter_samples=10, num_decimals=5):
    """MD5 of e_xc on random (features, parameters) samples; identical for forms that evaluate identically
    (SyFES XCFunctional.get_fingerprint)."""
    global _SIG_E
    if _SIG_E is None:
        _SIG_E = _lda_signature_prefactors()
    e_x, e_ss, e_os = _SIG_E
    layout = parameter_layout(form, used_only=False)
    rs = np.random.RandomState(0)
    psamples = rs.rand(num_parameter_samples, len(layout))
    feats = sorted({f for c in COMPONENTS for f in form[c]['feature_names']})
    sig = []
    with np.errstate(all='ignore'):
        for ps in psamples:
            P = split_params(form, ps, layout)
            idx = rs.choice(len(_SIG['rho']), size=num_feature_samples)
            fs = {'rho': _SIG['rho'][idx]}
            for f in feats:
                if f != 'rho':
                    fs[f] = rs.choice(_SIG[f], size=num_feature_samples)
            v = (e_x[idx] * _bcast(eval_ef(form['f_x'], fs, P['f_x']), num_feature_samples)
                 + e_ss[idx] * _bcast(eval_ef(form['f_css'], fs, P['f_css']), num_feature_samples)
                 + e_os[idx] * _bcast(eval_ef(form['f_cos'], fs, P['f_cos']), num_feature_samples))
            sig.extend(v.tolist())
    fmt = f'{{:.{num_decimals}f}}'
    return hashlib.md5(','.join(fmt.format(x) for x in sig).encode()).hexdigest()


def _bcast(v, n):
    return np.broadcast_to(np.asarray(v, dtype=float), (n,))


# ============================================================================ mutator (syfes/symbolic/mutators.py)
class Mutator:
    def __init__(self, seed=0, instruction_pool=None, mutation_pool=None,
                 component_probabilities=(1 / 3, 1 / 3, 1 / 3), max_num_bound_parameters=MAX_NUM_BOUND_PARAMETERS):
        self.ipool = dict(instruction_pool or INSTRUCTION_POOL)
        self.mpool = dict(mutation_pool or MUTATION_POOL)
        assert abs(sum(self.ipool.values()) - 1) < 1e-8 and abs(sum(self.mpool.values()) - 1) < 1e-8
        self.cprob = list(component_probabilities)
        self.max_bound = max_num_bound_parameters
        self.rs = np.random.RandomState(seed)

    # -- helpers
    def _instruction_name(self, existing_bound, num_inputs=None):
        cands = [n for n in self.ipool
                 if (num_inputs is None or INSTRUCTIONS[n][0] == num_inputs)
                 and len(INSTRUCTIONS[n][1]) <= self.max_bound - len(existing_bound)]
        for b in existing_bound:
            n = BOUND_PARAMETER_ASSOCIATION[b]
            if n in self.ipool and (num_inputs is None or INSTRUCTIONS[n][0] == num_inputs):
                cands.append(n)
        cands = sorted(set(cands))
        if not cands:
            raise ValueError('no instruction satisfies the conditions')
        p = np.array([self.ipool[n] for n in cands])
        return str(self.rs.choice(cands, p=p / p.sum()))

    def _random_instruction(self, ef):
        name = self._instruction_name(ef_bound_parameters(ef))
        out = str(self.rs.choice(ef['variable_names']))
        allowed = ef_allowed_inputs(ef)
        return [name, out] + [str(self.rs.choice(allowed)) for _ in range(INSTRUCTIONS[name][0])]

    def _index(self, ef, allow_last=False):
        return int(self.rs.randint(0, len(ef['instructions']) + (1 if allow_last else 0)))

    # -- the four mutation rules
    def insert_instruction(self, ef):
        L = copy.deepcopy(ef['instructions'])
        i = self._index(ef, allow_last=True)
        new = self._random_instruction(ef)
        L.insert(i, new)
        return L, f'insert@{i}: {_ins_str(new)}'

    def remove_instruction(self, ef):
        L = copy.deepcopy(ef['instructions'])
        i = self._index(ef)
        old = L.pop(i)
        return L, f'remove@{i}: {_ins_str(old)}'

    def replace_instruction(self, ef):
        L = copy.deepcopy(ef['instructions'])
        i = self._index(ef)
        old = L[i]
        name = self._instruction_name(ef_bound_parameters(ef), num_inputs=INSTRUCTIONS[old[0]][0])
        L[i] = [name] + old[1:]
        return L, f'replace@{i}: {_ins_str(old)} -> {_ins_str(L[i])}'

    def change_argument(self, ef):
        L = copy.deepcopy(ef['instructions'])
        i = self._index(ef)
        old = L[i]
        new = list(old)
        k = int(self.rs.randint(1 + INSTRUCTIONS[old[0]][0]))
        if k == 0:
            new[1] = str(self.rs.choice(list(ef['variable_names'])))
        else:
            new[1 + k] = str(self.rs.choice(ef_allowed_inputs(ef)))
        L[i] = new
        return L, f'change_arg{k}@{i}: {_ins_str(old)} -> {_ins_str(new)}'

    def mutate_ef(self, ef):
        if not ef['instructions']:
            mt = 'insert_instruction'
        else:
            mt = str(self.rs.choice(list(self.mpool), p=list(self.mpool.values())))
        L, msg = getattr(self, mt)(ef)
        new = dict(ef, instructions=L)
        validate_ef(new)
        return new, mt, msg

    def mutate(self, form):
        c = str(self.rs.choice(list(COMPONENTS), p=self.cprob))
        new_ef, mt, msg = self.mutate_ef(form[c])
        child = {k: copy.deepcopy(v) for k, v in form.items() if k in COMPONENTS}
        child[c] = new_ef
        return child, dict(component=c, mutation=mt, change=msg)


# ============================================================================ seed forms
def _ef(features, n_shared, n_vars, instructions):
    return dict(feature_names=list(features), shared_parameter_names=[f'c{i}' for i in range(n_shared)],
                variable_names=[f'v{i}' for i in range(n_vars - 1)] + ['enhancement_factor'],
                instructions=[list(i) for i in instructions])


F = 'enhancement_factor'
FEATURES = ('x2', 'w')   # syfes example/config.py feature_names_{x,css,cos} = 'x2,w'

# GAS22 (SI Eq. / Table S1; colab run_GAS22.ipynb), written with the shared-parameter / variable counts of the
# SyFES ωB97M-V "short" representation it evolved from (x: 3 params / 3 variables, css: 5/5, cos: 6/5).
GAS22_FORM = {
    'f_x': _ef(FEATURES, 3, 3, [
        ['UTransformInstruction', 'v0', 'x2'],
        ['AdditionInstruction', F, 'c0', F],
        ['MultiplicationAdditionInstruction', F, 'c1', 'w'],
        ['MultiplicationAdditionInstruction', F, 'c2', 'v0'],
    ]),
    'f_css': _ef(FEATURES, 5, 5, [
        ['UTransformInstruction', 'v0', 'x2'],
        ['AdditionInstruction', F, F, 'v0'],                      # + u   (coefficient 1)
        ['MultiplicationAdditionInstruction', F, 'c0', 'w'],
        ['Power2Instruction', 'v1', 'w'],
        ['MultiplicationAdditionInstruction', F, 'c1', 'v1'],
        ['Power6Instruction', 'v2', 'v0'],
        ['MultiplicationAdditionInstruction', F, 'c2', 'v2'],
        ['Power4Instruction', 'v1', 'w'],
        ['MultiplicationInstruction', 'v3', 'v1', 'v2'],
        ['MultiplicationAdditionInstruction', F, 'c3', 'v3'],
    ]),
    'f_cos': _ef(FEATURES, 6, 5, [
        ['AdditionInstruction', F, 'c0', F],
        ['Power2Instruction', 'v0', 'w'],
        ['MultiplicationAdditionInstruction', F, 'c1', 'v0'],
        ['CubeRootInstruction', 'v1', 'x2'],
        ['MultiplicationInstruction', 'v2', 'v0', 'v1'],
        ['MultiplicationAdditionInstruction', F, 'c2', 'v2'],
        ['Power6Instruction', 'v3', 'w'],
        ['MultiplicationAdditionInstruction', F, 'c3', 'v3'],
        ['MultiplicationInstruction', 'v2', 'v3', 'v1'],
        ['MultiplicationAdditionInstruction', F, 'c4', 'v2'],
    ]),
}
GAS22_PARAMS = {
    'f_x': {'c0': 0.862139736374172, 'c1': 0.317533683085033, 'c2': 0.936993691972698,
            'gamma_utransform': 0.003840616724010807},
    'f_css': {'c0': -4.10753796482853, 'c1': -5.24218990333846, 'c2': -1.76643208454076, 'c3': 7.5380689617542,
              'gamma_utransform': 0.46914023462026644},
    'f_cos': {'c0': 0.805124374375355, 'c1': 7.98909430970845, 'c2': -1.76098915061634, 'c3': -7.54815900595292,
              'c4': 2.00093961824784},
}
for _c in COMPONENTS:
    validate_ef(GAS22_FORM[_c])


def params_to_theta(form, params: dict, layout=None):
    layout = layout or parameter_layout(form)
    return np.array([params[c][n] for c, n in layout], dtype=float)


def gas22_reference_check():
    """F_x, F_css, F_cos of the encoded GAS22 vs the colab closed forms on random points (max abs diff)."""
    rs = np.random.RandomState(1)
    x2 = 10 ** rs.uniform(-3, 4, 1000)
    w = rs.uniform(-1, 1, 1000)
    u = lambda g: g * x2 / (1 + g * x2)
    ref = {
        'f_x': 0.862139736374172 + 0.936993691972698 * u(0.003840616724010807) + 0.317533683085033 * w,
        'f_css': (u(0.46914023462026644) - 4.10753796482853 * w - 5.24218990333846 * w ** 2
                  - 1.76643208454076 * u(0.46914023462026644) ** 6
                  + 7.5380689617542 * u(0.46914023462026644) ** 6 * w ** 4),
        'f_cos': (0.805124374375355 + 7.98909430970845 * w ** 2 - 1.76098915061634 * w ** 2 * np.cbrt(x2)
                  - 7.54815900595292 * w ** 6 + 2.00093961824784 * w ** 6 * np.cbrt(x2)),
    }
    return {c: float(np.abs(eval_ef(GAS22_FORM[c], {'x2': x2, 'w': w}, GAS22_PARAMS[c]) - ref[c]).max())
            for c in COMPONENTS}
