from game.flappy_bird import GameState

from random import randint
import cv2

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

def convert_to_gray(image_data):
    image_data = image_data[0:288, 0:400]
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    image_data[image_data > 0] = 255
    return image_data

bird_leftmost_X = 55
bird_X = 75
bird_size = 22
min_Y = 1
max_X = 285
max_Y = 395

def check_tube(image_data, tube_X):
    if image_data[tube_X, min_Y] == 0:
        return False, min_Y, max_Y
    if image_data[tube_X, max_Y] == 0:
        return False, min_Y, max_Y
    
    up_Y = min_Y
    down_Y = max_Y
    
    for Y in range(min_Y, max_Y):
        if image_data[tube_X, Y] == 0:
            up_Y = Y
            break
    
    for Y in range(max_Y, min_Y, -1):
        if image_data[tube_X, Y] == 0:
            down_Y = Y
            break
    
    return True, up_Y, down_Y

def get_bird_Y(image_data, bird_X):
    has_tube, tube_up_Y, tube_down_Y = check_tube(image_data, bird_X)
    
    bird_search_up_Y = tube_up_Y + 5
    bird_search_down_Y = tube_down_Y - 5
    
    last_value = 0
    bird_top_Y = bird_search_up_Y
    bird_bottom_Y = bird_search_down_Y
    
    for Y in range(bird_search_up_Y, bird_search_down_Y):
        if last_value == 0:
            if image_data[bird_X, Y] != 0:
                last_value = 1
                bird_top_Y = Y
        else:
            if image_data[bird_X, Y] == 0:
                bird_bottom_Y = Y
                break
    
    return (bird_top_Y + bird_bottom_Y) // 2

def check_need_flap(image_data):
    for tube_X in range(bird_leftmost_X, max_X,30):
        has_tube, tube_up_Y, tube_down_Y = check_tube(image_data, tube_X)
        if has_tube:
            break
    
    if not has_tube:
        tube_up_Y = min_Y
        tube_down_Y = max_Y
    
    bird_Y = get_bird_Y(image_data, bird_X)
    bird_bottom_safe_Y = bird_Y + bird_size / 2 + 15
    
    if bird_bottom_safe_Y >= tube_down_Y:
        return True
    elif bird_bottom_safe_Y >= max_Y:
        return True
    else:
        return False

def agent2():
    game_state = GameState()
    
    # Init
    need_flap = True

    before_score = 0
    current_score = 0
    # Loop
    while True:
        before_score = game_state.score
        action = [0, 1] if need_flap else [1, 0]
        image_data, reward, terminal = game_state.frame_step(action)
        image_data = convert_to_gray(image_data)
        need_flap = check_need_flap(image_data)
        current_score = game_state.score
        if (before_score != 0) and (current_score == 0):
            print("best score = ",before_score) 


if __name__ == "__main__":
    agent2()
