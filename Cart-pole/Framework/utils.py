import os
def generate_models_models(opt):
    save_dir = "TrainedModels/%s" % (opt.env)

    if opt.mode == 'train':
        try:
            os.makedirs(opt.model_dir)
        except OSError:
            pass
    return save_dir

def post_config(opt):
    # init fixed parameters
    opt.device = torch.device("cpu" if opt.not_cuda else "cuda:0")
    opt.model_dir = generate_models_models(opt)

    if torch.cuda.is_available() and opt.not_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    return opt