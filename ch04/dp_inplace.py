V = {"L1": 0.0, "L2": 0.0}

cnt = 0
while True:
    t = 0.5 * ( -1 + .9 * V["L1"]) + 0.5 * (1 + .9 * V["L2"])
    delta = abs(t - V["L1"])
    V["L1"] = t

    t = 0.5 * ( 0 + .9 * V["L2"]) + 0.5 * (-1 + .9 * V["L1"])
    delta = max(delta, abs(t - V["L2"]))
    V["L2"] = t

    cnt += 1
    if delta < 1e-4:
        print(V)
        print(f"Converged after {cnt} iterations")
        break

"""
{'L1': -2.2493782177156936, 'L2': -2.7494201578106514}
Converged after 60 iterations
"""