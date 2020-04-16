
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# world height
WORLD_HEIGHT = 7

# world width
WORLD_WIDTH = 10

# possible actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

# probability for exploration
EPSILON = 0.1

# Sarsa step size
ALPHA = 0.5

START = [6, 0]
GOAL = [6, 9]
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]

def step(state, action):
    i, j = state
    if action == ACTION_UP:
        return [max(i - 1, 0), j]
    elif action == ACTION_DOWN:
        return [max(min(i + 1, WORLD_HEIGHT - 1), 0), j]
    elif action == ACTION_LEFT:
        return [i, max(j - 1, 0)]
    elif action == ACTION_RIGHT:
        return [i, min(j + 1, WORLD_WIDTH - 1)]
    else:
        assert False

def reward(state, action):
    next_state = step(state, action)
    if next_state[0] == 6 and next_state[1] > 0 and next_state[1] < 10 - 1:
        return -10000
    return -1

# play for an episode
def episode(q_value):
    # track the total time steps in this episode
    time = 0

    # initialize state
    state = START

    while state != GOAL:
    # choose an action based on epsilon-greedy algorithm
        if np.random.binomial(1, EPSILON) == 1:
            action = np.random.choice(ACTIONS)
        else:
            values_ = q_value[state[0], state[1], :]
            action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

    # keep going until get to the goal state

        next_state = step(state, action)
        #if np.random.binomial(1, EPSILON) == 1:
        #    next_action = np.random.choice(ACTIONS)
        #else:
        values_ = q_value[next_state[0], next_state[1], :]
        next_action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

        # Sarsa update
        q_value[state[0], state[1], action] += \
            ALPHA * (reward(state, action) + q_value[next_state[0], next_state[1], next_action] -
                     q_value[state[0], state[1], action])

        if next_state[0] == 6 and next_state[1] > 0 and next_state[1] < 10 - 1:
            break

        state = next_state
        #action = next_action
        time += 1
    return time

def q_learning():
    q_value = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
    episode_limit = 500

    steps = []
    ep = 0
    while ep < episode_limit:
        steps.append(episode(q_value))
        # time = episode(q_value)
        # episodes.extend([ep] * time)
        ep += 1

    steps = np.add.accumulate(steps)

    plt.plot(steps, np.arange(1, len(steps) + 1))
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')

    plt.savefig('./q-learning.png')
    plt.close()

    # display the optimal policy
    optimal_policy = []
    for i in range(0, WORLD_HEIGHT):
        optimal_policy.append([])
        for j in range(0, WORLD_WIDTH):
            if [i, j] == GOAL:
                optimal_policy[-1].append('G')
                continue
            bestAction = np.argmax(q_value[i, j, :])
            if bestAction == ACTION_UP:
                optimal_policy[-1].append('U')
            elif bestAction == ACTION_DOWN:
                optimal_policy[-1].append('D')
            elif bestAction == ACTION_LEFT:
                optimal_policy[-1].append('L')
            elif bestAction == ACTION_RIGHT:
                optimal_policy[-1].append('R')
    print('Optimal policy is:')
    for row in optimal_policy:
        print(row)

if __name__ == '__main__':
    q_learning()