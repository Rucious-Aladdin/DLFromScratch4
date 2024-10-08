
import sys
if ".." not in sys.path: sys.path.append("..")
import numpy as np
from common.grid_world import GridWorld
from collections import defaultdict

def eval_onestep(pi, V, env, gamma=.9):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue
            
        action_probs = pi[state]
        new_V = 0

        for action, action_prob in action_probs.items():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            new_V += action_prob * (r + gamma * V[next_state])
        
        V[state] = new_V
    return V

def policy_eval(pi, V, env, gamma=.9, theta=1e-3):
    while True:
        old_V = V.copy()
        V = eval_onestep(pi, V, env, gamma)

        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t
        
        if delta < theta:
            break
    return V

if __name__ == "__main__":
    env = GridWorld()
    gamma = .9
    pi = defaultdict(lambda: {0:.25, 1:.25, 2:.25, 3:.25})
    V = defaultdict(lambda: 0)
    V = policy_eval(pi, V, env, gamma)
    env.render_v(V, pi, savefig=True, filename='random_policy.png')