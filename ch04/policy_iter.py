import sys
if ".." not in sys.path: sys.path.append("..")
from collections import defaultdict
from common.grid_world import GridWorld
from policy_eval import policy_eval

def argmax(d):
    max_value = max(d.values())
    max_key = 0
    for key, value in d.items():
        if value == max_value:
            max_key = key
    return max_key

def greedy_policy(V, env, gamma=.9):
    pi = {}

    for state in env.states():
        action_values = {}
        
        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]
            action_values[action] = value
        max_action = argmax(action_values)
        action_probs = {0: 0, 1: 0, 2: 0, 3: 0}
        action_probs[max_action] = 1.0
        pi[state] = action_probs
    return pi

def policy_iter(env, gamma=.9, threshold=.001, is_render=False, savefig=False, filename='policy_iter.png'):
    pi = defaultdict(lambda: {0:.25, 1:.25, 2:.25, 3:.25})
    V = defaultdict(lambda: 0)

    while True:
        V = policy_eval(pi, V, env, gamma, threshold)
        new_pi = greedy_policy(V, env, gamma)

        if is_render:
            env.render_v(V, pi)

        if new_pi == pi:
            if savefig:
                env.render_v(V, pi, savefig=True, filename=filename)
            break
        pi = new_pi
    return pi

if __name__ == "__main__":
    env = GridWorld()
    gamma = 0.9
    pi = policy_iter(env, gamma, is_render=True, savefig=True)