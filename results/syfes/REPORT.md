# SyFES baseline run (non-LLM regularized evolution from GAS22)

Run directory: `/home/ash/Documents/dft_functional_sr/runs/syfes/job0001_gas22_400`

Trained forms: 400 (finite 368, non-finite/error 32); equivalent mutations skipped by fingerprint: 1079; GPU time 61.1 GPU-h, mean 550.0 s and 119297 CMA-ES evaluations per form.

## Best forms, float64 re-check (stored parameters, no refit; SyFES split)

| form | params | train | validation | violations | failed constraints |
|---|---|---|---|---|---|
| GAS22 (seed, published) | 14 | 3.2702 | 3.4623 | 1 | ueg_exchange |
| 174 | 14 | 3.4441 | 3.5057 | 0 |  |
| 55 | 14 | 3.2690 | 3.5542 | 1 | ueg_exchange |
| 699 | 14 | 3.2072 | 3.5554 | 1 | ueg_exchange |
| 30 | 14 | 3.1934 | 3.5683 | 1 | ueg_exchange |
| 1244 | 14 | 3.1896 | 3.5655 | 1 | ueg_exchange |
| 11 | 14 | 3.1891 | 3.5739 | 1 | ueg_exchange |
| 57 | 14 | 3.1888 | 3.5895 | 1 | ueg_exchange |
| 784 | 15 | 3.1883 | 3.5945 | 1 | ueg_exchange |
| 1266 | 14 | 3.1883 | 3.5944 | 1 | ueg_exchange |
| 542 | 14 | 3.1883 | 3.5949 | 1 | ueg_exchange |

## Best-validation trajectory (search numbers)

| trained forms | form | train | validation |
|---|---|---|---|
| 1 | 2 | 3.3075 | 3.9695 |
| 3 | 1 | 3.3497 | 3.7958 |
| 10 | 11 | 3.189 | 3.5748 |
| 28 | 30 | 3.1933 | 3.5682 |
| 37 | 55 | 3.2688 | 3.5473 |
| 95 | 174 | 3.4441 | 3.5039 |

## Expressions of the re-checked forms

### form 0

```
f_x:
  v0 = UTransform(x2; gamma_utransform)
  enhancement_factor = c0 + enhancement_factor
  enhancement_factor += c1 * w
  enhancement_factor += c2 * v0
f_css:
  v0 = UTransform(x2; gamma_utransform)
  enhancement_factor = enhancement_factor + v0
  enhancement_factor += c0 * w
  v1 = w ** 2
  enhancement_factor += c1 * v1
  v2 = v0 ** 6
  enhancement_factor += c2 * v2
  v1 = w ** 4
  v3 = v1 * v2
  enhancement_factor += c3 * v3
f_cos:
  enhancement_factor = c0 + enhancement_factor
  v0 = w ** 2
  enhancement_factor += c1 * v0
  v1 = cbrt(x2)
  v2 = v0 * v1
  enhancement_factor += c2 * v2
  v3 = w ** 6
  enhancement_factor += c3 * v3
  v2 = v3 * v1
  enhancement_factor += c4 * v2
```

- f_x: `c0 + c1*w + c2*gamma_utransform*x2/(gamma_utransform*x2 + 1.0)`
- f_css: `c0*w + c1*w**2 + c2*gamma_utransform**6*x2**6/(gamma_utransform*x2 + 1.0)**6 + c3*gamma_utransform**6*w**4*x2**6/(gamma_utransform*x2 + 1.0)**6 + gamma_utransform*x2/(gamma_utransform*x2 + 1.0)`
- f_cos: `c0 + c1*w**2 + c2*w**2*x2**(1/3) + c3*w**6 + c4*w**6*x2**(1/3)`
- params: f_x.c0=0.86214, f_x.c1=0.317534, f_x.c2=0.936994, f_x.gamma_utransform=0.00384062, f_css.c0=-4.10754, f_css.c1=-5.24219, f_css.c2=-1.76643, f_css.c3=7.53807, f_css.gamma_utransform=0.46914, f_cos.c0=0.805124, f_cos.c1=7.98909, f_cos.c2=-1.76099, f_cos.c3=-7.54816, f_cos.c4=2.00094

### form 174

```
f_x:
  v0 = UTransform(x2; gamma_utransform)
  enhancement_factor = c0 + enhancement_factor
  enhancement_factor += c1 * w
  enhancement_factor += c2 * v0
f_css:
  v0 = UTransform(x2; gamma_utransform)
  enhancement_factor = enhancement_factor + v0
  enhancement_factor += v0 * v0
  enhancement_factor += c0 * w
  v1 = w ** 2
  enhancement_factor += c1 * v1
  v2 = v0 ** 6
  enhancement_factor += c2 * v2
  v1 = w ** 4
  v3 += v1 * v2
  enhancement_factor += c3 * v3
f_cos:
  enhancement_factor = c0 - enhancement_factor
  v0 = w ** 2
  enhancement_factor += c1 * v0
  v1 = cbrt(x2)
  v2 = v0 * v1
  enhancement_factor += c2 * v2
  v3 = w ** 6
  enhancement_factor += c3 * v3
  v2 = v3 * v1
  enhancement_factor += c4 * v2
```

- f_x: `c0 + c1*w + c2*gamma_utransform*x2/(gamma_utransform*x2 + 1.0)`
- f_css: `c0*w + c1*w**2 + c2*gamma_utransform**6*x2**6/(gamma_utransform*x2 + 1.0)**6 + c3*gamma_utransform**6*w**4*x2**6/(gamma_utransform*x2 + 1.0)**6 + gamma_utransform**2*x2**2/(gamma_utransform*x2 + 1.0)**2 + gamma_utransform*x2/(gamma_utransform*x2 + 1.0)`
- f_cos: `c0 + c1*w**2 + c2*w**2*x2**(1/3) + c3*w**6 + c4*w**6*x2**(1/3)`
- params: f_x.c0=0.846842, f_x.c1=0.296961, f_x.c2=0.682496, f_x.gamma_utransform=0.00666651, f_css.c0=-3.80104, f_css.c1=-5.92389, f_css.c2=-2.46188, f_css.c3=6.32983, f_css.gamma_utransform=0.682774, f_cos.c0=0.62517, f_cos.c1=8.06507, f_cos.c2=-1.637, f_cos.c3=-5.31495, f_cos.c4=1.88518

### form 55

```
f_x:
  v0 = UTransform(x2; gamma_utransform)
  enhancement_factor = c0 + enhancement_factor
  enhancement_factor += c1 * w
  v0 = v0 - c0
  enhancement_factor += c2 * v0
f_css:
  v0 = UTransform(x2; gamma_utransform)
  enhancement_factor = enhancement_factor + v0
  enhancement_factor += c0 * w
  v1 = w ** 2
  enhancement_factor += c1 * v1
  v2 = v0 ** 6
  enhancement_factor += c2 * v2
  v1 = w ** 4
  v3 = v1 * v2
  enhancement_factor += c3 * v3
f_cos:
  enhancement_factor = c0 + enhancement_factor
  v0 = w ** 2
  enhancement_factor += c1 * v0
  v1 = cbrt(x2)
  v2 = v0 * v1
  enhancement_factor += c2 * v2
  v3 = w ** 6
  enhancement_factor += c3 * v3
  v2 = v3 * v1
  enhancement_factor += c4 * v2
```

- f_x: `c0 + c1*w + c2*(-c0 + gamma_utransform*x2/(gamma_utransform*x2 + 1.0))`
- f_css: `c0*w + c1*w**2 + c2*gamma_utransform**6*x2**6/(gamma_utransform*x2 + 1.0)**6 + c3*gamma_utransform**6*w**4*x2**6/(gamma_utransform*x2 + 1.0)**6 + gamma_utransform*x2/(gamma_utransform*x2 + 1.0)`
- f_cos: `c0 + c1*w**2 + c2*w**2*x2**(1/3) + c3*w**6 + c4*w**6*x2**(1/3)`
- params: f_x.c0=4.66723, f_x.c1=0.346831, f_x.c2=0.815314, f_x.gamma_utransform=0.00461165, f_css.c0=-4.38767, f_css.c1=-4.3331, f_css.c2=-1.92363, f_css.c3=5.25525, f_css.gamma_utransform=0.438445, f_cos.c0=0.824597, f_cos.c1=6.28157, f_cos.c2=-1.43294, f_cos.c3=-5.15305, f_cos.c4=1.63409

### form 699

```
f_x:
  enhancement_factor = c2 ** 6
  v1 = v0 ** 2
  v0 = UTransform(x2; gamma_utransform)
  enhancement_factor = c0 - enhancement_factor
  v1 = sqrt(x2)
  enhancement_factor += c1 * w
  enhancement_factor += c2 * v0
  v1 = c1 / c1
f_css:
  v0 = UTransform(x2; gamma_utransform)
  enhancement_factor = enhancement_factor + v0
  enhancement_factor += c0 * w
  v1 = w ** 2
  enhancement_factor += c1 * v1
  v2 = v0 ** 6
  enhancement_factor += c2 * v2
  v1 = w ** 4
  v0 += c3 * enhancement_factor
  v3 = v1 * v2
  enhancement_factor += c3 * v3
f_cos:
  enhancement_factor = c0 + enhancement_factor
  v0 = w ** 2
  enhancement_factor += c1 * v0
  v1 = cbrt(x2)
  v2 = v0 * v1
  enhancement_factor += c2 * v2
  v3 = w ** 6
  v0 = cbrt(c3)
  enhancement_factor += c3 * v3
  v2 = v3 * v1
  enhancement_factor += c4 * v2
```

- f_x: `c0 + c1*w - c2**6 + c2*gamma_utransform*x2/(gamma_utransform*x2 + 1.0)`
- f_css: `c0*w + c1*w**2 + c2*gamma_utransform**6*x2**6/(gamma_utransform*x2 + 1.0)**6 + c3*gamma_utransform**6*w**4*x2**6/(gamma_utransform*x2 + 1.0)**6 + gamma_utransform*x2/(gamma_utransform*x2 + 1.0)`
- f_cos: `c0 + c1*w**2 + c2*w**2*x2**(1/3) + c3*w**6 + c4*w**6*x2**(1/3)`
- params: f_x.c0=2.97272, f_x.c1=0.282398, f_x.c2=1.13249, f_x.gamma_utransform=0.00293943, f_css.c0=-3.79622, f_css.c1=-4.34865, f_css.c2=-1.66765, f_css.c3=5.20415, f_css.gamma_utransform=0.46426, f_cos.c0=0.823897, f_cos.c1=6.61877, f_cos.c2=-1.50701, f_cos.c3=-5.44503, f_cos.c4=1.70034

### form 30

```
f_x:
  v0 = UTransform(x2; gamma_utransform)
  enhancement_factor = c2 ** 3
  enhancement_factor = c0 + enhancement_factor
  enhancement_factor += c1 * w
  enhancement_factor += c2 * v0
f_css:
  v0 = UTransform(x2; gamma_utransform)
  enhancement_factor = enhancement_factor + v0
  enhancement_factor += c0 * w
  v1 = w ** 2
  enhancement_factor += c1 * v1
  v2 = v0 ** 6
  enhancement_factor += c2 * v2
  v1 = w ** 4
  v3 = v1 * v2
  enhancement_factor += c3 * v3
f_cos:
  enhancement_factor = c0 + enhancement_factor
  v0 = w ** 2
  enhancement_factor += c1 * v0
  v1 = cbrt(x2)
  v2 = v0 * v1
  enhancement_factor += c2 * v2
  v3 = w ** 6
  enhancement_factor += c3 * v3
  v2 = v3 * v1
  enhancement_factor += c4 * v2
```

- f_x: `c0 + c1*w + c2**3 + c2*gamma_utransform*x2/(gamma_utransform*x2 + 1.0)`
- f_css: `c0*w + c1*w**2 + c2*gamma_utransform**6*x2**6/(gamma_utransform*x2 + 1.0)**6 + c3*gamma_utransform**6*w**4*x2**6/(gamma_utransform*x2 + 1.0)**6 + gamma_utransform*x2/(gamma_utransform*x2 + 1.0)`
- f_cos: `c0 + c1*w**2 + c2*w**2*x2**(1/3) + c3*w**6 + c4*w**6*x2**(1/3)`
- params: f_x.c0=-1.6102, f_x.c1=0.255096, f_x.c2=1.35249, f_x.gamma_utransform=0.00232501, f_css.c0=-3.5617, f_css.c1=-4.33821, f_css.c2=-1.55835, f_css.c3=5.25796, f_css.gamma_utransform=0.480573, f_cos.c0=0.827039, f_cos.c1=6.76372, f_cos.c2=-1.53755, f_cos.c3=-5.64497, f_cos.c4=1.72501

### form 1244

```
f_x:
  v1 = cbrt(enhancement_factor)
  v0 = sqrt(v1)
  v0 = UTransform(x2; gamma_utransform)
  v1 = v1 + v0
  enhancement_factor = c0 + enhancement_factor
  enhancement_factor += c1 * w
  enhancement_factor += c2 * v0
  enhancement_factor = enhancement_factor ** 6
f_css:
  v0 = UTransform(x2; gamma_utransform)
  enhancement_factor = v2 ** 2
  enhancement_factor = enhancement_factor + v0
  enhancement_factor += c0 * w
  v3 = c3 ** 6
  v2 = UTransform(enhancement_factor; gamma_utransform)
  v1 = w ** 2
  enhancement_factor += c1 * v1
  v2 = v0 ** 6
  enhancement_factor += c2 * v2
  v1 = w ** 4
  v3 = v1 * v2
  enhancement_factor += c3 * v3
  v2 = w ** 4
f_cos:
  v3 = c2 * c1
  enhancement_factor = c0 + enhancement_factor
  v0 = w ** 2
  enhancement_factor += c1 * v0
  v1 = cbrt(x2)
  v2 = v0 * v1
  enhancement_factor += c2 * v2
  v3 = w ** 6
  enhancement_factor += c3 * v3
  v2 = v3 * v1
  enhancement_factor += c4 * v2
```

- f_x: `(c0 + c1*w + c2*gamma_utransform*x2/(gamma_utransform*x2 + 1.0))**6`
- f_css: `c0*w + c1*w**2 + c2*gamma_utransform**6*x2**6/(gamma_utransform*x2 + 1.0)**6 + c3*gamma_utransform**6*w**4*x2**6/(gamma_utransform*x2 + 1.0)**6 + gamma_utransform*x2/(gamma_utransform*x2 + 1.0)`
- f_cos: `c0 + c1*w**2 + c2*w**2*x2**(1/3) + c3*w**6 + c4*w**6*x2**(1/3)`
- params: f_x.c0=0.975831, f_x.c1=0.0423102, f_x.c2=0.213746, f_x.gamma_utransform=0.00262957, f_css.c0=-3.38571, f_css.c1=-4.56859, f_css.c2=-1.46247, f_css.c3=5.82855, f_css.gamma_utransform=0.512869, f_cos.c0=0.833004, f_cos.c1=6.97167, f_cos.c2=-1.5532, f_cos.c3=-6.10816, f_cos.c4=1.72194

### form 11

```
f_x:
  v0 = UTransform(x2; gamma_utransform)
  enhancement_factor = c0 + enhancement_factor
  enhancement_factor += c1 * w
  enhancement_factor += c2 * v0
  enhancement_factor = enhancement_factor ** 3
f_css:
  v0 = UTransform(x2; gamma_utransform)
  enhancement_factor = enhancement_factor + v0
  enhancement_factor += c0 * w
  v1 = w ** 2
  enhancement_factor += c1 * v1
  v2 = v0 ** 6
  enhancement_factor += c2 * v2
  v1 = w ** 4
  v3 = v1 * v2
  enhancement_factor += c3 * v3
f_cos:
  enhancement_factor = c0 + enhancement_factor
  v0 = w ** 2
  enhancement_factor += c1 * v0
  v1 = cbrt(x2)
  v2 = v0 * v1
  enhancement_factor += c2 * v2
  v3 = w ** 6
  enhancement_factor += c3 * v3
  v2 = v3 * v1
  enhancement_factor += c4 * v2
```

- f_x: `(c0 + c1*w + c2*gamma_utransform*x2/(gamma_utransform*x2 + 1.0))**3`
- f_css: `c0*w + c1*w**2 + c2*gamma_utransform**6*x2**6/(gamma_utransform*x2 + 1.0)**6 + c3*gamma_utransform**6*w**4*x2**6/(gamma_utransform*x2 + 1.0)**6 + gamma_utransform*x2/(gamma_utransform*x2 + 1.0)`
- f_cos: `c0 + c1*w**2 + c2*w**2*x2**(1/3) + c3*w**6 + c4*w**6*x2**(1/3)`
- params: f_x.c0=0.952356, f_x.c1=0.0824295, f_x.c2=0.451388, f_x.gamma_utransform=0.00241613, f_css.c0=-3.36591, f_css.c1=-4.5151, f_css.c2=-1.45293, f_css.c3=5.73853, f_css.gamma_utransform=0.511714, f_cos.c0=0.833269, f_cos.c1=6.97351, f_cos.c2=-1.5604, f_cos.c3=-6.08619, f_cos.c4=1.73172

### form 57

```
f_x:
  v0 = UTransform(x2; gamma_utransform)
  enhancement_factor = c0 + enhancement_factor
  v0 += c2 * c0
  enhancement_factor += c1 * w
  enhancement_factor += c2 * v0
f_css:
  v0 = UTransform(x2; gamma_utransform)
  enhancement_factor = enhancement_factor + v0
  enhancement_factor += c0 * w
  v1 = w ** 2
  enhancement_factor += c1 * v1
  v2 = v0 ** 6
  enhancement_factor += c2 * v2
  v1 = w ** 4
  v3 = v1 * v2
  enhancement_factor += c3 * v3
f_cos:
  v3 = v2 - c1
  enhancement_factor = c0 + enhancement_factor
  v0 = w ** 2
  enhancement_factor += c1 * v0
  v1 = cbrt(x2)
  v2 = v0 * v1
  enhancement_factor += c2 * v2
  v3 = w ** 6
  enhancement_factor += c3 * v3
  v2 = v3 * v1
  enhancement_factor += c4 * v2
```

- f_x: `c0 + c1*w + c2*(c0*c2 + gamma_utransform*x2/(gamma_utransform*x2 + 1.0))`
- f_css: `c0*w + c1*w**2 + c2*gamma_utransform**6*x2**6/(gamma_utransform*x2 + 1.0)**6 + c3*gamma_utransform**6*w**4*x2**6/(gamma_utransform*x2 + 1.0)**6 + gamma_utransform*x2/(gamma_utransform*x2 + 1.0)`
- f_cos: `c0 + c1*w**2 + c2*w**2*x2**(1/3) + c3*w**6 + c4*w**6*x2**(1/3)`
- params: f_x.c0=0.248948, f_x.c1=0.235588, f_x.c2=1.57221, f_x.gamma_utransform=0.00191754, f_css.c0=-3.40159, f_css.c1=-4.32995, f_css.c2=-1.48245, f_css.c3=5.3167, f_css.gamma_utransform=0.492272, f_cos.c0=0.829011, f_cos.c1=6.87522, f_cos.c2=-1.56013, f_cos.c3=-5.79122, f_cos.c4=1.74193

### form 784

```
f_x:
  v1 = cbrt(enhancement_factor)
  v0 = UTransform(x2; gamma_utransform)
  v1 = v1 + v0
  enhancement_factor = c0 + enhancement_factor
  enhancement_factor += c1 * w
  enhancement_factor += c2 * v0
f_css:
  v0 = UTransform(x2; gamma_utransform)
  enhancement_factor = v2 ** 4
  v1 = sqrt(w)
  enhancement_factor = enhancement_factor + v0
  enhancement_factor += c0 * w
  v2 = UTransform(enhancement_factor; gamma_utransform)
  v1 = w ** 2
  enhancement_factor += c1 * v1
  v2 = v0 ** 6
  enhancement_factor += c2 * v2
  v1 = w ** 4
  v3 = v1 * v2
  enhancement_factor += c3 * v3
  v2 = w ** 4
f_cos:
  v3 = v2 * c1
  enhancement_factor = c0 + enhancement_factor
  v0 = w ** 2
  enhancement_factor += c1 * v0
  enhancement_factor += c5 * c1
  v1 = cbrt(x2)
  v2 = v0 * v1
  enhancement_factor += c2 * v2
  v3 = w ** 6
  enhancement_factor += c3 * v3
  v2 = v3 * v1
  enhancement_factor += c4 * v2
```

- f_x: `c0 + c1*w + c2*gamma_utransform*x2/(gamma_utransform*x2 + 1.0)`
- f_css: `c0*w + c1*w**2 + c2*gamma_utransform**6*x2**6/(gamma_utransform*x2 + 1.0)**6 + c3*gamma_utransform**6*w**4*x2**6/(gamma_utransform*x2 + 1.0)**6 + gamma_utransform*x2/(gamma_utransform*x2 + 1.0)`
- f_cos: `c0 + c1*c5 + c1*w**2 + c2*w**2*x2**(1/3) + c3*w**6 + c4*w**6*x2**(1/3)`
- params: f_x.c0=0.864491, f_x.c1=0.228675, f_x.c2=1.66278, f_x.gamma_utransform=0.00178619, f_css.c0=-3.34039, f_css.c1=-4.32072, f_css.c2=-1.45425, f_css.c3=5.35061, f_css.gamma_utransform=0.498591, f_cos.c0=-0.549064, f_cos.c1=6.90382, f_cos.c2=-1.56748, f_cos.c3=-5.84395, f_cos.c4=1.74651, f_cos.c5=0.199867

### form 1266

```
f_x:
  v0 = UTransform(x2; gamma_utransform)
  enhancement_factor = c0 + enhancement_factor
  enhancement_factor += c1 * w
  enhancement_factor += c2 * v0
  v1 += c1 * c1
  enhancement_factor += enhancement_factor * c0
f_css:
  v0 = UTransform(x2; gamma_utransform)
  enhancement_factor = enhancement_factor + v0
  enhancement_factor += c0 * w
  v1 = w ** 2
  enhancement_factor += c1 * v1
  v2 = v0 ** 6
  enhancement_factor += c2 * v2
  v1 = c2 ** 6
  v1 = w ** 4
  v3 = v1 * v2
  enhancement_factor += c3 * v3
  v2 = sqrt(enhancement_factor)
f_cos:
  enhancement_factor = c0 + enhancement_factor
  v0 = w ** 2
  v3 = c4 / c1
  enhancement_factor += c1 * v0
  v1 = cbrt(x2)
  v2 = v0 * v1
  enhancement_factor += c2 * v2
  v3 = v2 ** 3
  v3 = w ** 6
  enhancement_factor += c3 * v3
  v2 = v3 * v1
  enhancement_factor += c4 * v2
  v3 = w ** 6
```

- f_x: `c0*(c0 + c1*w + c2*gamma_utransform*x2/(gamma_utransform*x2 + 1.0)) + c0 + c1*w + c2*gamma_utransform*x2/(gamma_utransform*x2 + 1.0)`
- f_css: `c0*w + c1*w**2 + c2*gamma_utransform**6*x2**6/(gamma_utransform*x2 + 1.0)**6 + c3*gamma_utransform**6*w**4*x2**6/(gamma_utransform*x2 + 1.0)**6 + gamma_utransform*x2/(gamma_utransform*x2 + 1.0)`
- f_cos: `c0 + c1*w**2 + c2*w**2*x2**(1/3) + c3*w**6 + c4*w**6*x2**(1/3)`
- params: f_x.c0=0.555667, f_x.c1=0.146972, f_x.c2=1.07163, f_x.gamma_utransform=0.0017831, f_css.c0=-3.33799, f_css.c1=-4.32481, f_css.c2=-1.45396, f_css.c3=5.3514, f_css.gamma_utransform=0.498523, f_cos.c0=0.830965, f_cos.c1=6.90894, f_cos.c2=-1.56891, f_cos.c3=-5.8463, f_cos.c4=1.74798

### form 542

```
f_x:
  v1 = v0 ** 2
  v0 = UTransform(x2; gamma_utransform)
  enhancement_factor = c0 + enhancement_factor
  v1 = sqrt(x2)
  enhancement_factor = sqrt(c0)
  enhancement_factor += c1 * w
  enhancement_factor += c2 * v0
  v1 = c1 / c1
f_css:
  v0 = UTransform(x2; gamma_utransform)
  enhancement_factor = enhancement_factor + v0
  enhancement_factor += c0 * w
  v1 = w ** 2
  enhancement_factor += c1 * v1
  v2 = v0 ** 6
  enhancement_factor += c2 * v2
  v1 = c2 ** 6
  v1 = w ** 4
  v3 = v1 * v2
  enhancement_factor += c3 * v3
f_cos:
  enhancement_factor = c0 + enhancement_factor
  v0 = w ** 2
  v3 = c4 + c1
  enhancement_factor += c1 * v0
  v1 = cbrt(x2)
  v2 = v0 * v1
  enhancement_factor += c2 * v2
  v3 = v2 ** 3
  v3 = w ** 6
  enhancement_factor += c3 * v3
  v2 = v3 * v1
  enhancement_factor += c4 * v2
```

- f_x: `sqrt(c0) + c1*w + c2*gamma_utransform*x2/(gamma_utransform*x2 + 1.0)`
- f_css: `c0*w + c1*w**2 + c2*gamma_utransform**6*x2**6/(gamma_utransform*x2 + 1.0)**6 + c3*gamma_utransform**6*w**4*x2**6/(gamma_utransform*x2 + 1.0)**6 + gamma_utransform*x2/(gamma_utransform*x2 + 1.0)`
- f_cos: `c0 + c1*w**2 + c2*w**2*x2**(1/3) + c3*w**6 + c4*w**6*x2**(1/3)`
- params: f_x.c0=0.747333, f_x.c1=0.227712, f_x.c2=1.67851, f_x.gamma_utransform=0.0017667, f_css.c0=-3.33205, f_css.c1=-4.32254, f_css.c2=-1.45037, f_css.c3=5.35117, f_css.gamma_utransform=0.499562, f_cos.c0=0.831092, f_cos.c1=6.90962, f_cos.c2=-1.56855, f_cos.c3=-5.8507, f_cos.c4=1.74726
