import argparse

def get_arguments():
    parser = argparse.ArgumentParser()

    # workspace:
    parser.add_argument('--not_cuda', action='store_true', help='disables cuda', default=0)

    # load, input, save configurations:
    parser.add_argument('--load_dir', default='', help='load trained network parameters')

    # environments settings:
    parser.add_argument('--env', default='CartPole-v1', help="Gym environments")

    
    # networks hyper parameters:
    parser.add_argument('--rm_size', type=int, help='replay memory capacity', default=100000)


    # optimization hyper parameters:
    parser.add_argument('--niters', type=int, help='number of iterations', default=2)

    return parser