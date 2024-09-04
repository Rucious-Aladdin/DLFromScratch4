V = {"L1": 0.0, "L2": 0.0}
new_V = V.copy()

cnt = 0
while True:
    new_V["L1"] =.5 * ( -1 + .9 * V["L1"]) + .5 * (1 + .9 * V["L2"])
    new_V["L2"] =.5 * ( 0 + .9 * V["L2"]) + .5 * (-1 + .9 * V["L1"])

    delta = abs(new_V["L1"] - V["L1"])
    delta = max(delta, abs(new_V["L2"] - V["L2"]))

    V = new_V.copy()

    cnt += 1
    if delta < 1e-4:
        print(V)
        print(f"Converged after {cnt} iterations")
        break

"""
{'L1': -2.249167525908671, 'L2': -2.749167525908671}
Converged after 76 iterations
"""