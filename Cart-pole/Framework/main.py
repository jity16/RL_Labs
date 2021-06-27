from typing import DefaultDict
from config import get_arguments
from agent import train
import utils as utils

# import time

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--mode', default='train', help='train / test mode')

    opt = parser.parse_args()
    opt = utils.post_config(opt)

    if opt.mode == "train":
        train(opt)
    # else:
    #     test(opt)
