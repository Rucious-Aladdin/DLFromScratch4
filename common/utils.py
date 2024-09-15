import numpy as np

def greedy_probs(Q, state, epsilon=0.1, action_size=4):
    qs = [Q[(state, a)] for a in range(action_size)]
    max_action = np.argmax(qs)

    base_prob = epsilon / action_size
    action_probs = {action: base_prob for action in range(action_size)}
    action_probs[max_action] += (1 - epsilon)
    return action_probs