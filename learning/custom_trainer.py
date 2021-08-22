from __future__ import print_function

import os
import sys
import time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict

from .util import AverageMeter, accuracy
from .base_trainer import BaseTrainer

try:
    from apex import amp, optimizers
except ImportError:
    pass


class CustomTrainer(BaseTrainer):
    """trainer for Linear evaluation"""
    def __init__(self, args):
        super(CustomTrainer, self).__init__(args)
        self.log_step = 1

    def logging(self, epoch, logs, lr=None, train=True):
        """ logging to tensorboard

        Args:
          epoch: training epoch
          logs: loss and accuracy
          lr: learning rate
          train: True of False
        """
        args = self.args
        if args.rank == 0:
            pre = 'train_' if train else 'test_'
            if args.learning_rate != 30:
                pre = f'lr_{args.learning_rate:.3f}/' + pre
            self.logger.log_value(pre+'acc', logs[0], epoch)
            self.logger.log_value(pre+'acc5', logs[1], epoch)
            self.logger.log_value(pre+'loss', logs[2], epoch)
            if train and (lr is not None):
                self.logger.log_value('learning_rate', lr, epoch)

    def wrap_up(self, model, model_ema, optimizer):
        """Wrap up models with apex and DDP

        Args:
          model: model
          model_ema: momentum encoder
          optimizer: optimizer
        """
        args = self.args

        model.cuda(args.gpu)
        if isinstance(model_ema, torch.nn.Module):
            model_ema.cuda(args.gpu)

        # to amp model if needed
        if args.amp:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level=args.opt_level
            )
            if isinstance(model_ema, torch.nn.Module):
                model_ema = amp.initialize(
                    model_ema, opt_level=args.opt_level
                )
        # to distributed data parallel
        model = DDP(model, device_ids=[args.gpu])

        if isinstance(model_ema, torch.nn.Module):
            self.momentum_update(model.module, model_ema, 0)

        return model, model_ema, optimizer

    def resume_model(self,  model, model_ema, optimizer):
        """load checkpoint"""
        args = self.args
        start_epoch = 1
        if args.resume:
            if os.path.isfile(args.resume):
                checkpoint = torch.load(args.resume, map_location='cpu')
                args.start_epoch = checkpoint['epoch'] + 1
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                if isinstance(model_ema, torch.nn.Module):
                    model_ema.load_state_dict(checkpoint['model_ema'])
                if args.amp:
                    amp.load_state_dict(checkpoint['amp'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
                del checkpoint
                torch.cuda.empty_cache()
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        return start_epoch

    def save(self, model, model_ema, optimizer, epoch):
        """save model to checkpoint"""
        args = self.args
        if args.local_rank == 0 and args.rank == 0:
            # saving the model to each instance
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            if isinstance(model_ema, torch.nn.Module):
                state['model_ema'] = model_ema.state_dict()
            if args.amp:
                state['amp'] = amp.state_dict()
            save_file = os.path.join(args.model_folder, 'current.pth')
            torch.save(state, save_file)
            if epoch % args.save_freq == 0:
                save_file = os.path.join(
                    args.model_folder, 'ckpt_epoch_{}.pth'.format(epoch))
                torch.save(state, save_file)
            # help release GPU memory
            del state

    def train(self, epoch, train_loader, model, model_ema,
              criterion, optimizer):
        time1 = time.time()
        args = self.args

        model.train()
        model_ema.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        end = time.time()
        for idx, (input, target) in enumerate(train_loader):
            data_time.update(time.time() - end)

            input = input.float()
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # forward
            output = model(x=input, mode=2)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update momentum model
            self.momentum_update(model.module, model_ema, args.alpha)

            batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if args.local_rank == 0 and idx % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, idx, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1, top5=top5))
                sys.stdout.flush()

        time2 = time.time()
        print('train epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        return top1.avg, top5.avg, losses.avg

    def validate(self, epoch, val_loader, model, criterion):
        time1 = time.time()
        args = self.args

        model.eval()

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        with torch.no_grad():
            end = time.time()
            for idx, (input, target) in enumerate(val_loader):
                input = input.float()
                input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(x=input, mode=2)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(acc1[0], input.size(0))
                top5.update(acc5[0], input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if args.local_rank == 0 and idx % args.print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                           idx, len(val_loader), batch_time=batch_time,
                           loss=losses, top1=top1, top5=top5))

            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))

        time2 = time.time()
        print('eval epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        return top1.avg, top5.avg, losses.avg

    @staticmethod
    def momentum_update(model, model_ema, m):
        """ model_ema = m * model_ema + (1 - m) model """
        for p1, p2 in zip(model.parameters(), model_ema.parameters()):
            p2.data.mul_(m).add_(p1.detach().data, alpha=1 - m)
            p1.data = p2.data
