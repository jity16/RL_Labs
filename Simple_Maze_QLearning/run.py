from maze_env import Maze
from agent import QTable


def explore():
    for episode in range(100):
        # initial state
        state = env.reset()
        # repeat
        while True:
            env.render()
            # choose action 
            action = agent.choose_best_action(str(state))

            # take action and get next state and reward
            state_, reward, done = env.step(action)

            # update Q_table
            agent.update_value(str(state), action, reward, str(state_))

            # update state
            state = state_

            # break tile terminal
            if done:
                break

    # end of game
    print('game over')
    env.destroy()

env = Maze()
agent = QTable(actions=list(range(env.n_actions)))
env.after(100,explore)
env.mainloop()