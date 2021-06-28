import argparse

def get_arguments():
    parser = argparse.ArgumentParser()

    # workspace:
    parser.add_argument('--not_cuda', action='store_true', help='disables cuda', default=1)

    # load, input, save configurations:
    parser.add_argument('--load_dir', default='', help='load trained network parameters')

    # environments settings:
    parser.add_argument('--env', default='CartPole-v1', help="Gym environments")

    
    # networks hyper parameters:
    parser.add_argument('--rm_size', type=int, help='replay memory capacity', default=100000)


    # optimization hyper parameters:
    parser.add_argument('--niters', type=int, help='number of iterations', default=20)
    parser.add_argument('--eps_start', type = float, help='initial epsilon', default=0.9)
    parser.add_argument('--eps_end', type=float, help='final epsilon', default=0.05)
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--gamma',type=float,help='discount factor', default=0.99)
    return parser