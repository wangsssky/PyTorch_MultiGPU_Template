import os
import math
from .base_options import BaseOptions


class CustomOptions(BaseOptions):

	def initialize(self, parser):
		parser = BaseOptions.initialize(self, parser)

		# model setup
		parser.add_argument('--arch', default='resnet50', type=str,
							help='e.g., resnet18, resnet50')

		# method selection
		parser.add_argument('--method', default='Customize', type=str,
							choices=['InsDis', 'CMC', 'Customize'],
							help='Choose predefined method. Configs will be override '
								 'for all methods except for `Customize`, which allows '
								 'for user-defined combination of methods')

		parser.add_argument('--aug', default='A', type=str,
							help='data augmentation for training')
		parser.add_argument('--input_res', type=int, default=224)

		parser.add_argument('--crop', type=float, default=0.2,
							help='crop threshold for RandomResizedCrop')

		parser.add_argument('--save_score', action='store_true')

		# parser.set_defaults(epochs=60)
		# parser.set_defaults(learning_rate=30)
		# parser.set_defaults(lr_decay_epochs='60,80')
		# parser.set_defaults(lr_decay_rate=0.1)
		# parser.set_defaults(weight_decay=0)

		return parser

	def modify_options(self, opt):
		"""
		some settings based basic options
		:param opt: options
		:return: modified options
		"""
		iterations = opt.lr_decay_epochs.split(',')
		opt.lr_decay_epochs = list([])
		for it in iterations:
			opt.lr_decay_epochs.append(int(it))

		# set up saving name
		opt.model_name = '{}_{}_aug_{}'.format(
			opt.method, opt.arch, opt.aug
		)
		if opt.amp:
			opt.model_name = '{}_amp_{}'.format(opt.model_name, opt.opt_level)
		if opt.cosine:
			opt.model_name = '{}_cosine'.format(opt.model_name)
		if opt.use_ema:
			opt.model_name = '{}_ema'.format(opt.model_name)

		# warm-up for large-batch training, e.g. 1024 with multiple nodes
		if opt.batch_size > 256:
			opt.warm = True
		if opt.warm:
			opt.model_name = '{}_warm'.format(opt.model_name)
			opt.warmup_from = 0.01
			if opt.epochs > 500:
				opt.warm_epochs = 10
			else:
				opt.warm_epochs = 5
			if opt.cosine:
				eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
				opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
						1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
			else:
				opt.warmup_to = opt.learning_rate

		opt.model_name = '{}_epoch{}'.format(opt.model_name, opt.epochs)
		opt.model_name = '{}_bs{}'.format(opt.model_name, opt.batch_size)
		opt.model_name = '{}_lr{}'.format(opt.model_name, opt.learning_rate)
		opt.model_name = '{}_{}'.format(opt.model_name, opt.optm)

		if opt.exp_iter != -1:
			opt.model_name = '{}_iter{}'.format(opt.model_name, opt.exp_iter)

		# create folders
		opt.model_folder = os.path.join(opt.model_path, opt.model_name)
		if not os.path.isdir(opt.model_folder):
			os.makedirs(opt.model_folder)
		opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
		if not os.path.isdir(opt.tb_folder):
			os.makedirs(opt.tb_folder)

		return opt
