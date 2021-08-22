import os
import argparse


class BaseOptions(object):

    def __init__(self):
        self.initialized = False
        self.parser = None
        self.opt = None

    def initialize(self, parser):

        # specify folder
        parser.add_argument('--data_folder', type=str, default='./data',
                            help='path to data')
        parser.add_argument('--model_path', type=str, default='./save',
                            help='path to save model')
        parser.add_argument('--tb_path', type=str, default='./tb',
                            help='path to tensorboard')
        parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet'])
        parser.add_argument('--exp_iter', type=int, default=-1,
                            help='when not -1, will be written into the dir name')
        
        # basics
        parser.add_argument('--print_freq', type=int, default=10,
                            help='print frequency')
        parser.add_argument('--save_freq', type=int, default=20,
                            help='save frequency')
        parser.add_argument('--batch_size', type=int, default=16,
                            help='batch_size')
        parser.add_argument('-j', '--num_workers', type=int, default=0,
                            help='num of workers to use')

        # optimization
        parser.add_argument('--epochs', type=int, default=200,
                            help='number of training epochs')
        parser.add_argument('--optm', type=str, default='ADAM', choices=['ADAM', 'SGD'])

        parser.add_argument('--learning_rate', type=float, default=0.003,
                            help='learning rate')
        parser.add_argument('--lr_decay_epochs', type=str, default='120,160',
                            help='where to decay lr, can be a list')
        parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                            help='decay rate for learning rate')
        parser.add_argument('--weight_decay', type=float, default=1e-4,
                            help='weight decay')
        parser.add_argument('--momentum', type=float, default=0.9,
                            help='momentum for SGD')
        parser.add_argument('--cosine', action='store_true',
                            help='using cosine annealing')
        parser.add_argument('--warm', action='store_true',
                            help='add warm-up setting')
        parser.add_argument('--amp', action='store_true',
                            help='using mixed precision')
        parser.add_argument('--opt_level', type=str, default='O2',
                            choices=['O1', 'O2'])
        parser.add_argument('--use_ema', action='store_true')
        parser.add_argument('--alpha', default=0.999, type=float,
                            help='momentum coefficients for model update')

        # resume
        parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')

        # Parallel setting
        parser.add_argument('--world-size', default=-1, type=int,
                            help='number of nodes for distributed training')
        parser.add_argument('--rank', default=-1, type=int,
                            help='node rank for distributed training')
        parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                            help='url used to set up distributed training')
        parser.add_argument('--dist-backend', default='nccl', type=str,
                            help='distributed backend, [nccl, mpi, gloo]')
        parser.add_argument('--seed', default=None, type=int,
                            help='seed for initializing training. ')
        parser.add_argument('--gpu', default=None, type=int,
                            help='GPU id to use.')
        parser.add_argument('--multiprocessing-distributed', action='store_true',
                            help='Use multi-processing distributed training to launch '
                                 'N processes per node, which has N GPUs. This is the '
                                 'fastest way to use PyTorch for either single node or '
                                 'multi node data parallel training')

        return parser

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>35}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def modify_options(self, opt):
        raise NotImplementedError

    def parse(self):
        if not self.initialized:
            parser = argparse.ArgumentParser('arguments options')
            parser = self.initialize(parser)
            self.parser = parser
            self.initialized = True
        else:
            parser = self.parser

        opt = parser.parse_args()
        opt = self.modify_options(opt)
        self.opt = opt

        self.print_options(opt)

        return opt
