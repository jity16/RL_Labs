import pandas as pd
import numpy as np
class QTable:
    def __init__(self, actions, learning_rate=0.01, gamma=0.9, e_greedy=0.95):
        self.actions = actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.e_greedy = e_greedy
        self.q_table = pd.DataFrame(columns=actions, dtype=float)
    
    def choose_best_action(self, state):
        self.check_state_exist(state)
        all_actions = self.q_table.loc[state]
        if np.random.uniform() < self.e_greedy:
            # * choose best action: a = argmax(Q(s,a))
            best_actions = all_actions[all_actions == all_actions.max()].index
            action = np.random.choice(best_actions)
        else:
            action = np.random.choice(all_actions.index)
        return action

    def update_value(self, s1, a, r, s2):
        self.check_state_exist(s2)
        q_predict = self.q_table.loc[s1,a]
        q_real = r + self.gamma * self.q_table.loc[s1].max()
        self.q_table.loc[s1,a] += self.learning_rate * (q_real - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.actions,
                    name=state,
                )
            )


