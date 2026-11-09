# Prior functional forms: B97 → ωB97M-V → GAS22 → SAFS26

Reference sheet for the agentic search. It collects every functional form that appears in the two source papers:

* **[SyFES]** H. Ma, A. Narayanaswamy, P. Riley, L. Li, *Evolving symbolic density functionals*, Sci. Adv. 8, eabq0279 (2022). Code: google-research/symbolic_functionals.
* **[SAFS]** T. Duston et al., *Agentic Discovery of Exchange–Correlation Density Functionals*, arXiv:2605.05460 (2026).

Both papers use the same data (MGCDB84) and the same splits, and both start from ωB97M-V. In the extracted PDF text of [SyFES] the "ω" is lost, so it reads "B97M-V". The released GAS22 code, however, uses the ωB97M-V range-separated hybrid frame, which confirms that the starting point is ωB97M-V.

Exact coefficients are taken from the authors' code where it exists (SyFES `xc/mgga.py`, `xc/gga.py`, `colab/run_GAS22.ipynb`). Each form is also implemented in PyTorch in `baselines/`. `dftsr/xc_torch.py` reproduces libxc ωB97M-V to about 1e-7 Ha per molecule.

---

## 0. Common framework (B97-type, spin-resolved)

$$E_{xc} = \underbrace{\sum_\sigma \int e^{\rm LDA}_{x,\sigma}\,F_{\rm rsh}(a_\sigma)\,g_x\,d\mathbf r}_{\text{SR semilocal exchange}} + \underbrace{\sum_\sigma\int e^{\rm LDA}_{c,\sigma\sigma}\,g_{c,ss}\,d\mathbf r + \int e^{\rm LDA}_{c,\alpha\beta}\,g_{c,os}\,d\mathbf r}_{\text{semilocal correlation}} + c_x E_x^{\rm HF,SR} + E_x^{\rm HF,LR} + E_c^{\rm VV10}$$

| piece | definition (code: `dftsr/xc_torch.py`) |
|---|---|
| LDA exchange, one spin | $e^{\rm LDA}_{x,\sigma} = \tfrac12 e_x^{\rm LDA}(2\rho_\sigma) = -\tfrac34(3/\pi)^{1/3}\,2^{1/3}\rho_\sigma^{4/3}$ |
| RSH attenuation | $F_{\rm rsh}(a)=1-\tfrac23 a\left[2\sqrt\pi\,{\rm erf}(1/a)-3a+a^3+(2a-a^3)e^{-1/a^2}\right]$, $a=\omega/k_{F\sigma}$, $k_{F\sigma}=(6\pi^2\rho_\sigma)^{1/3}$ |
| PW92 correlation | $e_c^{\rm PW92}(\rho_\alpha,\rho_\beta)$ with the PW_MOD constants (as in libxc) |
| Stoll partition | $e_{c,\sigma\sigma}=e_c^{\rm PW92}(\rho_\sigma,0)$, $e_{c,\alpha\beta}=e_c^{\rm PW92}(\rho_\alpha,\rho_\beta)-e_{c,\alpha\alpha}-e_{c,\beta\beta}$ |
| reduced gradient (B97 "x", [SAFS] "s") | $x_\sigma^2 = \lvert\nabla\rho_\sigma\rvert^2/\rho_\sigma^{8/3}$ |
| gradient variable | $u = \gamma x^2/(1+\gamma x^2)\in[0,1)$ |
| kinetic variable | $t_\sigma = \tau^{\rm UEG}_\sigma/\tau_\sigma$, $\tau^{\rm UEG}_\sigma=\tfrac{3}{10}(6\pi^2)^{2/3}\rho_\sigma^{5/3}$ ($\tau$ includes the ½) |
| kinetic variable | $w = (t-1)/(t+1)\in[-1,1]$ (UEG: $w=0$; single orbital $\tau=\tau_W$: $w\to$ positive) |
| opposite-spin averages | $x_{\rm avg}^2=\tfrac12(x_\alpha^2+x_\beta^2)$, $t_{\rm avg}=\tfrac12(t_\alpha+t_\beta)$, $w_{\rm avg}=(t_{\rm avg}-1)/(t_{\rm avg}+1)$ |
| fixed (ωB97M-V frame) | $\omega=0.3$, $c_x=0.15$ (SR HF), LR HF = 1, VV10 $b=6.0$, $C=0.01$ |
| γ values | $\gamma_x=0.004$, $\gamma_{ss}=0.2$, $\gamma_{os}=0.006$ |

[SAFS] writes the same variables as $s=\lvert\nabla\rho\rvert/\rho^{4/3}$, $t=\tau/\rho^{5/3}$ and $w=(k_\sigma-t)/(k_\sigma+t)$ with $k_\sigma=\tfrac{3}{10}(6\pi^2)^{2/3}$. These are identical to the definitions above.

**Power series notation.** $g=\sum_{ij} c_{ij}\,w^i u^j$. The B97M-V / ωB97M-V term sets were chosen by best-subset selection over train+validation.

---

## 1. B97 (Becke 1997), GGA: SyFES proof-of-principle target

$g = c_0 + c_1 u + c_2 u^2$ (u only, no τ)

| channel | γ | c0 | c1 | c2 |
|---|---|---|---|---|
| x | 0.004 | 0.8094 | 0.5073 | 0.7481 |
| c,ss | 0.2 | 0.1737 | 2.3487 | −2.4868 |
| c,os | 0.006 | 0.9454 | 0.7471 | −4.5961 |

SyFES rediscovered the B97 exchange form from scratch, with B97 total energies as targets, as
$F = c_0^2 + c_1\left(c_2 + \frac{x^2}{1+\gamma x^2}\right)^2$ ($c_0$=0.8504, $c_1$=0.7480, $c_2$=0.3394, γ=0.0040). Expanded, this is $c_0+c_1u+c_2u^2$ with 0.8094 / 0.5073 / 0.7481.

## 2. ωB97X-V, range-separated hybrid GGA (relative of the target; SyFES code)

$g_x = 0.833 + 0.603u + 1.194u^2$, $g_{ss}=0.556-0.257u$, $g_{os}=1.219-1.850u$, with ω = 0.3 and the same γ values. It has 10 parameters.

## 3. B97M-V (2015), pure meta-GGA + VV10 (`baselines/b97mv.py`, family `mgga`)

| channel | terms $c_{ij}\,w^iu^j$ |
|---|---|
| x | $1.000 + 1.308u + 1.901u^2 + 0.416w + 3.070wu$ |
| c,ss | $1.000 - 1.855u^2 - 5.668w - 20.497w^3u^2 - 20.364w^4u^2$ |
| c,os | $1.000 + 1.573u - 6.298u^3 + 2.535w - 6.427w^3u^2$ |

## 4. ωB97M-V (2016): the baseline and seed (`baselines/wb97mv.py`, family `rsh`)

| channel | terms | parameters |
|---|---|---|
| x (SR) | $g_x = 0.85 + 0.259\,w + 1.007\,u_x$ | $c_{x,00}=1-c_x=0.85$ fixed by the UEG limit |
| c,ss | $0.443 - 4.535\,w - 3.39\,w^2 + 4.278\,w^4u^3 - 1.437\,u^4$ | γ = 0.2 |
| c,os | $1.000 + 1.358\,w + 2.924\,w^2 - 1.39\,w^6 - 8.812\,w^2u + 9.142\,w^6u$ | γ = 0.006, spin-averaged w, u |

That gives 12 free linear coefficients ([SAFS] counts 12; our implementation trains 13, because $c_{os,00}$ is also refit). [SAFS] writes the exchange as $g_x = c_{x,0} + c_1 w + c_2 u$.

Reported WRMSD (kcal/mol, MGCDB84, non-SCF on the ωB97M-V density, as quoted by [SAFS]): train 3.32 / val 4.02 / test 3.64; val+test excluding RG10 ("tot") 4.07. [SyFES] quotes a test WRMSD of 4.237 for ωB97M-V.

## 5. GAS22, from SyFES regularized evolution (`baselines/gas22.py`)

The search used 12 evolutions, 29,628 forms in total, CMA-ES fitting with parameters in [−10, 10], and an instruction set {+, −, ×, /, s += p×q, pⁿ with n ∈ {2, 3, 4, 6, 1/2, 1/3}, p/(1+γp)}. The ωB97M-V RSH frame is kept.

$$g_x = 0.862139736 + 0.936993692\,u_x + 0.317533683\,w,\qquad \gamma_x = 0.003840617$$
$$g_{c,ss} = u - 4.107537965\,w - 5.242189903\,w^2 - 1.766432085\,u^6 + 7.538068962\,u^6w^4,\qquad \gamma_{ss}=0.469140235$$
$$g_{c,os} = 0.805124374 + 7.989094310\,w^2 - 1.760989151\,w^2(x^2)^{1/3} - 7.548159006\,w^6 + 2.000939618\,w^6(x^2)^{1/3}$$

* The exchange is symbolically the ωB97M-V form; the search did not change exchange.
* Same-spin correlation: the power series stays, but with new exponents; the constant term disappears and the coefficient of u is 1.
* Opposite-spin correlation contains $x^{2/3}=(x^2)^{1/3}$ (i.e. $|\nabla\rho|^{2/3}/\rho^{8/9}$-type), which lies outside the {w, u} power-series space. This was the main novelty.
* Note: $g_x(\text{UEG}) = 0.862 \ne 0.85$, a 1.4 % violation of the [SAFS] UEG constraint (threshold 0.5 %).
* WRMSD: non-SCF train 3.27 / val 3.74 / test 3.19 ([SAFS] Table 3), tot 3.77. SyFES reports test 3.585 vs 4.237 for ωB97M-V (−15 %). SCF results are very close to non-SCF. GAS22 does worse than ωB97M-V only on TCD.

## 6. SAFS26-a, agentic search (Seed 2.0), physical constraints enforced

* **Exchange:** $g_x = P_x(w,u) + c_{v,0}\,v + c_{v,1}\,v^2$, with $v = st/(1+st+\epsilon)$ ($s=x$, $t=\tau/\rho^{5/3}$). This adds a bounded gradient–kinetic cross term; $v\to0$ at the UEG ($s=0$), so the UEG limit is preserved.
* **Correlation (ss and os share the form):** $g = N/D$ with
  * $N = P_c(w,u) + c_v[0]v + c_v[1]v^2 + \tanh(c_z[0]z + c_z[1]z^2) + c_x[0]\chi + c_x[1]\chi^2$
  * $D = 1 + \sum_i \mathrm{softplus}(d_i)\,\phi_i$, with basis $\phi_i$ = monomials of $P_c(w,u)$ ∪ $\{v, v^2, z, z^2, \chi, \chi^2\}$. Softplus coefficients give $D \ge 1$ and no poles.
  * $z = f_\zeta(\zeta)\,s\,t/(1+f_\zeta s t+\epsilon)$ and $\chi = z s t/(1+z s t+\epsilon)$, where $f_\zeta=[(1+\zeta)^{4/3}+(1-\zeta)^{4/3}-2]/(2^{4/3}-2)$. Same-spin uses per-spin s, t; opposite-spin uses spin-averaged values.
* It has 50 trainable parameters (4 exchange, 46 correlation).
* WRMSD: train 2.76 / val 3.65 / test 3.41 / tot **3.70** (−9.1 % vs ωB97M-V). It passes spin symmetry, UEG and scaling, but fails AE18 grid convergence.
* Caveat noted by the authors: the exchange "spin" descriptor $z_x$ collapses to $v$, because exchange is evaluated per spin channel, so ζ is not available there.
* Per category (val+test): TCD 1.76, TCE 3.63, ID 10.30 (best). It is worse than ωB97M-V on NCED and NCD.

## 7. SAFS26-b, agentic search, physical + grid constraints

* **Exchange:** $g_x = P_x(w,u,v)\,(1 + c_{\rm sig}\,\sigma(a_x(s^2-b_x)))$. Here $v = 1/(1+\alpha^2)$, with iso-orbital indicator $\alpha=(\tau-\tau_W)/\tau_{\rm UEG}$ clipped to [0.01, 1]; cross terms are wv and uv. The sigmoid factor slightly breaks UEG but stays under the threshold.
* **Same-spin:** $g_{\alpha\alpha} = [P_{css}(w_\alpha,u_\alpha)(1+c_{\rm sig}\sigma(a_{ss}(t_\alpha-k_c/2))) + c_1u_\alpha w_\alpha + c_2u_\alpha^2w_\alpha^2]\,\big(1+\frac{\beta_{ss}r_{s,\alpha}+c_{v}r_{s,\alpha}v_\alpha+c_\zeta\zeta_\alpha}{1+\Gamma_{ss}r_{s,\alpha}^2}\big)$, with $\zeta_\alpha=\zeta(1-\zeta)$ and $\zeta_\beta=-\zeta(1+\zeta)$ (so the result is spin-symmetric) and $\Gamma=0.1$ fixed.
* **Opposite-spin:** $g_{\alpha\beta} = P_{cos}(w_{\rm avg},u_{\rm avg},z)\,\big(1+\frac{\beta_{os}+c_v v_{\rm avg}+c_\zeta|\zeta|}{1+\Gamma_{os}r_{s,\rm avg}^2}\big)$, with $z=f_\zeta(\zeta)$ and an 11-term polynomial with cross powers in z up to quadratic. Here $s_{\rm avg}=\sqrt{(s_\alpha^2+s_\beta^2)/2}$, $t_{\rm avg}=2t_\alpha t_\beta/(t_\alpha+t_\beta)$ (harmonic mean) and $r_{s,\rm avg}=(3/4\pi\rho)^{1/3}$.
* It has 19 trainable parameters (5 same-spin polynomial coefficients frozen from the parent).
* WRMSD: train 2.75 / val 3.78 / test 3.44 / tot **3.83** (−5.9 %). It passes all 4 constraints.

## 8. Other functionals reported in [SAFS]

| functional | model | constraints | train | val | test | tot |
|---|---|---|---|---|---|---|
| ωB97M-V | — | — | 3.32 | 4.02 | 3.64 | 4.07 |
| GAS22 | — | — | 3.27 | 3.74 | 3.19 | 3.77 |
| SAFS26-x | Seed 2.0 | none (fails all 4) | 2.91 | 3.95 | 3.49 | 3.81 |
| SAFS26-a | Seed 2.0 | physical | 2.76 | 3.65 | 3.41 | 3.70 |
| SAFS26-b | Seed 2.0 | physical + grid | 2.75 | 3.78 | 3.44 | 3.83 |
| AFS26-a | GPT 5.4 | physical | 2.81 | 3.74 | 3.78 | 3.90 |
| AFS26-b | GPT 5.4 | physical + grid | 2.65 | 3.65 | 3.96 | 3.90 |
| WB97MV_CrossTerm…ActivatedCSS_v1 | Seed 2.0, no islands | — | — | 3.64 | 4.09 | — |

The AFS26 forms are not given in the paper. Their validation–test gap indicates validation overfitting.

## 9. Intermediate ideas from the SAFS26 trajectories (useful stepping stones)

| iter | name | idea | Δ WRMSD_val |
|---|---|---|---|
| 14 | ExtendedWB97MV | bounded cross descriptor $v=st/(1+st)$ in all 3 channels | 4.40→4.09 |
| 30 | SpinEnhancedExtendedWB97MV | spin descriptor $z=f_\zeta st/(1+f_\zeta st)$ in correlation | →4.05 |
| 97 | SpinExEnhancedWB97MV | same in exchange (no effect: ζ collapses per spin channel) | ≈0 |
| 18 | WB97MV_Rational | P/Q rational for x and c, **failed UEG** (trainable denominator constant) | 4.40→4.17 |
| 82 | TanhBoundedSpinCross… | tanh-bounded spin descriptors + $x=zv/(1+zv)$ in correlation | →4.08 |
| 106 | RationalBoundedCrossEnhanced | rational correlation only, softplus denominator coefficients (D ≥ 1) | →3.81 |
| 197 | Fusion… (SAFS26-a) | spin-enhanced exchange lineage + rational correlation lineage | →3.65 |
| 8 | WB97MV_AlphaDescriptor | $v=1/(1+\alpha^2)$ in all channels, **failed UEG** in exchange | 4.40→4.34 |
| 80 | CorrelationOnlyVDescriptor | v only in correlation, all constraints pass | 4.54 |
| 141/192 | SigmoidRegimeTunableSwitch | sigmoid switches in exchange + $r_s$ rational correction $(1+\beta r_s/(1+\gamma r_s^2))$ in correlation | 4.54→4.13 |
| 332 | spin-polarization corrections | $\zeta(1-\zeta)$ (ss), $\lvert\zeta\rvert$ (os) terms | 4.12→4.04 |
| 384 | ZetaPolynomial (SAFS26-b) | $z=f_\zeta$ as third polynomial variable in os | →3.78 |

(The 4.40 starting value is [SAFS]'s baseline under their scoring and setup; their ωB97M-V val is 4.02.)

## 10. Anti-patterns: how unconstrained or over-exploited searches cheated ([SAFS] App. F, G)

1. **Antisymmetric spin term:** $g_{ab} \mathrel{+}= \alpha\,\zeta$ breaks $E_{xc}[\rho_\alpha,\rho_\beta]=E_{xc}[\rho_\beta,\rho_\alpha]$ (3.5 mHa).
2. **Neural/sigmoid layer in exchange with a nonzero bias:** $\sigma(b)\neq0$ at the UEG, so $g_x \neq 0.85$ (1.67 %).
3. **Laplacian feature** $\nabla^2\rho/(k_F^2\rho)$ in exchange: not invariant under coordinate scaling, and a finite-difference Laplacian on the grid is grid-dependent. (The Laplacian is deliberately not exposed by our API.)
4. **Steep switch** $1/(1+e^{10(r_s-0.5)})$: a grid-sensitivity blow-up on AE18 (0.100 vs 0.015 kcal/mol).
5. **Frozen-parameter grafting:** freezing 61 of 65 parent parameters and training 4 new ones gives a record validation score but test 4.09 > ωB97M-V. This leaks the validation set through selection pressure.

---

## 11. Data, splits and metric (both papers)

* MGCDB84: 4986 data points, 5931 single-point calculations (our ACCDB copy: 4985 after dropping XB51_15; 5930 molecules used).
* Splits (SyFES = ωB97M-V "training / primary test / secondary test"): **train 870, validation 2963, test 1152**. The data types are NCED, NCEC, NCD, IE, ID, TCE, TCD, BH, AE18 and RG10.
* $\mathrm{WRMSD}=\sqrt{\frac1N\sum_i w_i(E_i-E_i^{\rm ref})^2}$ (kcal/mol). Group weights: TCD 0.1; TCE, AE18 1; NCD, ID, BH 10; NCED, NCEC, IE 100; RG10 10⁴ (positive-reference RG10 points get 1). Each subset weight is normalised within its group to [1, 2], and validation weights are ×2. See `dftsr/dataset.py`.
* [SAFS] variant: two C20C24 molecules moved from test to validation (`load_reactions('agentic2026')` moves the whole C20C24 subset). The final "tot" metric is val + test excluding RG10 (3547 points).
* Densities: ωB97M-V/def2-QZVPPD, PySCF default grid (level 3), SG-1 grid for VV10 ([SyFES] protocol). Non-SCF evaluation is done on these fixed densities.
