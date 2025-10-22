
def fuse(scores, weights):
    return float(max(0.0, min(1.0, sum(weights.get(k,0)*scores.get(k,0.5) for k in weights))))
DEFAULT_WEIGHTS={'zt_quality':0.25,'leader':0.15,'lhb':0.15,'news':0.10,'chip':0.10,'chan':0.08,'elliott':0.07,'fib':0.05,'mood':0.03,'risk':0.02}
