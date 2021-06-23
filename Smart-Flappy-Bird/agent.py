from game.flappy_bird import GameState

from random import randint



def rand_action():
    if randint(1, 100) <= 10:
        return [0, 1]
    else:
        return [1, 0]

def agent1():
    game_state = GameState()
    
    while True:
        action = rand_action()
        image_data, reward, terminal = game_state.frame_step(action)

        

if __name__ == "__main__":
    agent1()