import numpy as np
import matplotlib.pyplot as plt

class Grid(object):
    
    def __init__(self, discount=0.9, start_state=None):
        # -1: wall
        # 0: empty, episode continues
        # other: number indicates reward, episode will terminate
        self._layout = np.zeros((10, 10))
        self._start_state = start_state
        self._goal_state = (8, 2)
        self._layout[(2, 8)] = 10
        self._state = self._start_state
        self._number_of_states = np.prod(np.shape(self._layout))
        self._discount = discount

    @property
    def number_of_states(self):
        return self._number_of_states
    
    def plot_grid(self):
        plt.figure(figsize=(3, 3))
        plt.imshow(self._layout > -1, interpolation="nearest")     
        ax = plt.gca()
        ax.grid(0)
        plt.xticks([])
        plt.yticks([])
        plt.title("The grid")
        plt.text(self._start_state[0], self._start_state[1], 
            r"$\mathbf{S}$", ha='center', va='center', color='white')
        plt.text(self._goal_state[0], self._goal_state[1], 
            r"$\mathbf{G}$", ha='center', va='center', color='white')
        h, w = self._layout.shape
        for y in range(h-1):
            plt.plot([-0.5, w-0.5], [y+0.5, y+0.5], '-k', lw=2)
        for x in range(w-1):
            plt.plot([x+0.5, x+0.5], [-0.5, h-0.5], '-k', lw=2)

    def plot_windy_grid(self, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.imshow(self._layout > -1, interpolation="nearest")     
        # ax = plt.gca()
        ax.grid(0)
        ax.set_xticks([])
        ax.set_yticks([])

        for i in range(10):
            ax.annotate(r'$\downarrow$', xy=(0.1*i, 1.1), xycoords='axes fraction', size=15)
            ax.annotate('2', xy=(0.1*i+0.035, 1.02), xycoords='axes fraction', size=10)
        if self._start_state is not None:
            ax.text(self._start_state[0], self._start_state[1], 
                r"$\mathbf{S}$", ha='center', va='center', color='white')
        ax.text(self._goal_state[0], self._goal_state[1], 
            r"$\mathbf{G}$", ha='center', va='center', color='white')
        h, w = self._layout.shape
        for y in range(h-1):
            ax.plot([-0.5, w-0.5], [y+0.5, y+0.5], '-k', lw=2)
        for x in range(w-1):
            ax.plot([x+0.5, x+0.5], [-0.5, h-0.5], '-k', lw=2)
  
    def get_obs(self):
        y, x = self._state
        return y*self._layout.shape[1] + x
  
    def int_to_state(self, int_obs):
        x = int_obs % self._layout.shape[1]
        y = int_obs // self._layout.shape[1]
        return y, x

    def step(self, action):
        y, x = self._state

        if action == 0:  # up
            new_state = (y - 1, x)
        elif action == 1:  # right
            new_state = (y, x + 1)
        elif action == 2:  # down
            new_state = (y + 1, x)
        elif action == 3:  # left
            new_state = (y, x - 1)
        else:
            raise ValueError("Invalid action: {} is not 0, 1, 2, or 3.".format(action))

        new_y, new_x = new_state
        new_y += 2
        new_state = (np.mod(new_y, 10), np.mod(new_x, 10))
        
        new_y, new_x = new_state
        if self._layout[new_y, new_x] == -1:  # wall
            reward = -5.
            discount = self._discount
            new_state = (y, x)
        elif self._layout[new_y, new_x] == 0:  # empty cell
            reward = -.5
            discount = self._discount
        else:  # a goal
            reward = self._layout[new_y, new_x]
            discount = 1.
            new_state = self._start_state
        
        self._state = new_state
        return reward, discount, self.get_obs()
  
class boundedGrid(Grid):
    def __init__(self, discount=0.9, l=None, w=None, start_state=None):
        self._layout = np.zeros((2*l, 2*w))
        self._layout[int(.5*l):int(1.5*l), int(.5*w):int(1.5*w)] = 1
        self._layout[int(.5*l-1):int(1.5*l+1), int(.5*w-1):int(1.5*w+1)] += 1
        if start_state is not None:
            self._start_state = start_state
        else:
            self._start_state=(int(.7*l), int(.7*w))
        self._layout[self._layout == 1] = -1.5
        self._goal_state=(int(1.3*l), int(.7*w))
        self._state=self._start_state
        self._number_of_states=np.prod(np.shape(self._layout))
        self._discount = discount
    
    def plot_grid(self, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.imshow(self._layout, interpolation="nearest")   
        
        ax.grid(0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Bounded grid")
        ax.text(self._start_state[0], self._start_state[1], 
            r"$\mathbf{S}$", ha='center', va='center', color='black')
        ax.text(self._goal_state[0], self._goal_state[1], 
            r"$\mathbf{G}$", ha='center', va='center', color='black')
        h, w = self._layout.shape
        for y in range(h-1):
            ax.plot([-0.5, w-0.5], [y+0.5, y+0.5], '-k', lw=2, alpha=.5)
        for x in range(w-1):
            ax.plot([x+0.5, x+0.5], [-0.5, h-0.5], '-k', lw=2, alpha=.5)
