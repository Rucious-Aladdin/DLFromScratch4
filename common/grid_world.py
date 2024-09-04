import numpy as np
import common.grid_world_renderer as render_helper


class GridWorld:
    def __init__(self):
        self.action_space = [0, 1, 2, 3]
        self.action_meaning = {
            0: 'UP',
            1: 'RIGHT',
            2: 'DOWN',
            3: 'LEFT'
        }

        self.reward_map = np.array([
            [0, 0, 0, 1],
            [0, None, 0, -1],
            [0, 0, 0, 0]
        ])
        self.goal_state = (0, 3)
        self.wall_state = (1, 1)
        self.start_state = (2, 0)
        self.agent_space = self.start_state

    def next_state(self, state, action):
        action_move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move = action_move_map[action]
        next_state = (state[0] + move[0], state[1] + move[1])
        ny, nx = next_state


        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            next_state = state
        elif next_state == self.wall_state:
            next_state = state

        return next_state

    def reward(self, state, action, next_state):
        return self.reward_map[next_state]
    
    def render_v(self, v=None, policy=None, print_value=True, savefig=True, filename='image.png'):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state,
                                          self.wall_state)
        renderer.render_v(v, policy, print_value, savefig=savefig, filename=filename)

    def render_q(self, q=None, print_value=True, savefig=True, filename='image.png'):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state,
                                          self.wall_state)
        renderer.render_q(q, print_value, savefig=savefig, filename='image.png')

    @property
    def height(self):
        return self.reward_map.shape[0]
    
    @property
    def width(self):
        return self.reward_map.shape[1]
    
    @property
    def shape(self):
        return self.reward_map.shape
    
    def actions(self):
        return self.action_space
    
    def states(self):
        for h in range(self.height):
            for w in range(self.width):
                yield (h, w)

if __name__ == '__main__':
    env = GridWorld()
    print(env.height)
    print(env.width)
    print(env.shape)

    for action in env.actions():
        print(env.action_meaning[action])

    print("===")

    for state in env.states():
        print(state)

    env.render_v(savefig=True, filename='init_gridworld.png')
    """
    3
    4
    (3, 4)
    UP
    RIGHT
    DOWN
    LEFT
    ===
    (0, 0)
    (0, 1)
    (0, 2)
    (0, 3)
    (1, 0)
    (1, 1)
    (1, 2)
    (1, 3)
    (2, 0)
    (2, 1)
    (2, 2)
    (2, 3)
    """