import os.path as osp
import time
from distutils import dist

from tensorboardX.writer import SummaryWriter

from reid.evaluation.evaluators_t import Evaluator as Evaluator_t2i
from reid.utils.meters import AverageMeter
from reid.utils.serialization import save_checkpoint


class BaseTrainer(object):
    def __init__(self, model, args, num_classes, is_distributed=False):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.args = args
        self.num_classes = num_classes
        self.is_distributed = is_distributed
        self.fp16 = args.fp16
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()

        # 仅在主进程（rank 0）或单进程模式下初始化 TensorBoard writer
        if not self.is_distributed or self.is_distributed and dist.get_rank() == 0:
            writer_dir = osp.join(self.args.logs_dir, 'data')
            self.writer = SummaryWriter(log_dir=writer_dir)

    def _logging(self, cur_iter):
        # 仅在主进程或单进程模式下打印日志
        if self.is_distributed and dist.get_rank() != 0:
            return
        if not (cur_iter % self.args.print_freq == 0):
            return
        print('Iter: [{}/{}]\t'
              'Time {:.3f} ({:.3f}) (ETA: {:.2f}h)\t'
              'Data {:.3f} ({:.3f})\t'
              .format(cur_iter, self.args.iters,
                      self.batch_time.val, self.batch_time.avg,
                      (self.args.iters - cur_iter) * self.batch_time.avg / 3600,
                      self.data_time.val, self.data_time.avg))

    def _refresh_information(self, cur_iter, lr):
        if not (cur_iter % self.args.refresh_freq == 0 or cur_iter == 1):
            return
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        # 仅在主进程或单进程模式下打印学习率
        if self.is_distributed and dist.get_rank() != 0:
            return
        print("lr = {} \t".format(lr))

    def _tensorboard_writer(self, current_iter, data):
        # 仅在主进程或单进程模式下记录 TensorBoard
        if self.is_distributed and dist.get_rank() != 0:
            return
        for key, value in data.items():
            self.writer.add_scalar(key, value, current_iter)

    def _do_valid(self, test_loader, query, gallery, validate_feat):
        assert query is not None and gallery is not None
        print('=' * 80)
        print("Validating....")
        self.model.eval()
        # 假设当前任务是 t2i
        evaluator = Evaluator_t2i(self.model, validate_feat)
        mAP = evaluator.evaluate(test_loader, query, gallery)
        self.model.train()
        print('=' * 80)
        return mAP

    def _parse_data(self, inputs):
        imgs, _, pids, _, indices = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets

    def run(self, inputs):
        raise NotImplementedError

    def train(self, data_loader, optimizer, lr_scheduler, test_loader=None, query=None, gallery=None):
        self.model.train()

        end = time.time()
        best_mAP, best_iter = 0, 0

        for i, inputs in enumerate(data_loader):
            current_iter = i + 1

            self._refresh_information(current_iter, lr=lr_scheduler.get_lr()[0])
            self.data_time.update(time.time() - end)

            loss = self.run(inputs)

            optimizer.zero_grad()
            if self.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            self.batch_time.update(time.time() - end)
            end = time.time()

            self._logging(current_iter)

            # 仅在主进程或单进程模式下保存模型和验证
            if self.is_distributed and dist.get_rank() != 0:
                continue

            if current_iter % self.args.save_freq == 0:
                if test_loader is not None:
                    mAP = self._do_valid(test_loader, query, gallery, self.args.validate_feat)
                    if best_mAP < mAP:
                        best_mAP = mAP
                        best_iter = current_iter
                    end = time.time()

                save_checkpoint({'state_dict': self.model.state_dict()},
                                fpath=osp.join(self.args.logs_dir,
                                               'checkpoints',
                                               'checkpoint_{}.pth.tar'.format(current_iter)))

                print('\n * Finished iterations {:3d}. Best iter {:3d}, Best mAP {:4.1%}.\n'
                      .format(current_iter, best_iter, best_mAP))

            lr_scheduler.step()