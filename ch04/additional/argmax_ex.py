def argmax(d):
    max_value = max(d.values())
    max_key = 0
    for key, value in d.items():
        if value == max_value:
            max_key = key
    return max_key

action_values = {0: .1, 1: -.3, 2: 9.9, 3: -1.3}

max_action = argmax(action_values)
print(max_action)