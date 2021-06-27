import argparse

def get_arguments():
    parser = argparse.ArgumentParser()

    # workspace:
    parser.add_argument('--not_cuda', action='store_true', help='disables cuda', default=0)

    # environments settings:
    parser.add_argument('--env', default='CartPole-v1', help="Gym environments")

    # networks hyper parameters:
    
