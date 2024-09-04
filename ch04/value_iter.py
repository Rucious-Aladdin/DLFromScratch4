import sys
if ".." not in sys.path: sys.path.append("..")
import numpy as np
from collections import defaultdict
from common.grid_world import GridWorld
from ch04.policy_iter import greedy_policy

def value_iter_onestep(
        V: dict, 
        env: GridWorld, 
        gamma: float
    ) -> dict:

    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue

        action_values = []
        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]
            action_values.append(value)
        
        V[state] = max(action_values)
    return V

def value_iter(
        V: dict,
        env: GridWorld,
        gamma: float,
        threshold: float = 1e-3,
        is_render: bool = True,
        save_last_fig: bool = False
):  
    while True:
        if is_render:
            env.render_v(V)
        
        old_V = V.copy()
        V = value_iter_onestep(V, env, gamma)

        delta = max(abs(old_V[state] - V[state]) for state in env.states())
        for state in V.keys():
            t = abs(old_V[state] - V[state])
            if delta < t:
                delta = t

        if delta < threshold:
            if save_last_fig:
                env.render_v(V, savefig=True, filename='value_iter.png')
            break
    return V

if __name__ == "__main__":
    V = defaultdict(lambda: 0)
    env = GridWorld()
    gamma = .9

    V = value_iter(V, env, gamma, is_render=True, save_last_fig=True)

    pi = greedy_policy(V, env, gamma)
    env.render_v(V, pi, savefig=True, filename='value_iter_with_policy.png')