"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import tempfile
from caffe.proto import caffe_pb2
import caffe
import second.data.kitti_common as kitti
# from tqdm import tqdm
from visualdl import LogWriter
import numpy as np

def get_prototxt(solver_proto, save_path=None):
    if save_path:
        f = open(save_path, mode='w+')
    else:
        f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.write(str(solver_proto))
    f.close()

    return f.name


def standard_solver(train_net,
                    test_net,
                    prefix,
                    solver_type='ADAM',
                    weight_decay=0.001,
                    base_lr=0.01,
                    gamma=0.1,
                    stepsize=100,
                    test_iter=100,
                    test_interval=1000,
                    max_iter=1e5,
                    iter_size=1,
                    snapshot=1000,
                    display=1,
                    random_seed=0,
                    debug_info=False,
                    create_prototxt=True,
                    save_path=None):

    solver = caffe_pb2.SolverParameter()
    solver.train_net = train_net
    solver.test_net.extend([test_net])

    solver.test_iter.extend([test_iter])
    solver.test_interval = test_interval

    solver.base_lr = base_lr
    solver.lr_policy = 'step'  # "fixed"
    solver.gamma = gamma
    solver.stepsize = stepsize

    solver.display = display
    solver.max_iter = max_iter
    solver.iter_size = iter_size
    solver.snapshot = snapshot
    solver.snapshot_prefix = prefix
    solver.random_seed = random_seed

    solver.solver_mode = caffe_pb2.SolverParameter.GPU
    if solver_type is 'SGD':
        solver.solver_type = caffe_pb2.SolverParameter.SGD
    elif solver_type is 'ADAM':
        solver.solver_type = caffe_pb2.SolverParameter.ADAM
    solver.momentum = 0.9
    solver.momentum2 = 0.999

    solver.weight_decay = weight_decay

    solver.debug_info = debug_info

    if create_prototxt:
        solver = get_prototxt(solver, save_path)

    return solver


class SolverWrapper:
    def __init__(self,  train_net,
                        test_net,
                        prefix,
                        solver_type='ADAM',
                        weight_decay=0.001,
                        base_lr=0.002,
                        gamma=0.8, #0.1 for lr_policy
                        stepsize=100,
                        test_iter=3768,
                        test_interval=1000,
                        max_iter=1e5,
                        iter_size=1,
                        snapshot=1000,
                        display=1,
                        random_seed=0,
                        debug_info=False,
                        create_prototxt=True,
                        save_path=None):
        """Initialize the SolverWrapper."""
        self.solver_param = caffe_pb2.SolverParameter()
        self.solver_param.train_net = train_net
        self.solver_param.test_net.extend([test_net])

        self.solver_param.test_iter.extend([test_iter])
        self.solver_param.test_interval = test_interval
        self.solver_param.test_initialization = False

        self.solver_param.base_lr = base_lr
        self.solver_param.lr_policy = 'step'  # "fixed" #exp
        self.solver_param.gamma = gamma
        self.solver_param.stepsize = stepsize

        self.solver_param.display = display
        self.solver_param.max_iter = max_iter
        self.solver_param.iter_size = iter_size
        self.solver_param.snapshot = snapshot
        self.solver_param.snapshot_prefix = prefix
        self.solver_param.random_seed = random_seed

        self.solver_param.solver_mode = caffe_pb2.SolverParameter.GPU
        if solver_type is 'SGD':
            self.solver_param.solver_type = caffe_pb2.SolverParameter.SGD
        elif solver_type is 'ADAM':
            self.solver_param.solver_type = caffe_pb2.SolverParameter.ADAM
        self.solver_param.momentum = 0.9
        self.solver_param.momentum2 = 0.999

        self.solver_param.weight_decay = weight_decay

        self.solver_param.debug_info = debug_info

        if create_prototxt:
            solver_prototxt = get_prototxt(self.solver_param, save_path)

        self.solver = caffe.get_solver(solver_prototxt)

        self.cur_epoch = 0
        self.test_interval = 50 #1856  #replace self.solver_param.test_interval #9280

        self.logw = LogWriter("permutohedral_log", sync_cycle=100)
        with self.logw.mode('train') as logger:
            self.sc_train_reg_loss = logger.scalar("reg_loss")
            self.sc_train_cls_loss = logger.scalar("cls_loss")
        with self.logw.mode('val') as logger:
            self.sc_val_3d_easy_7 = logger.scalar("mAP3D_easy_7")
            self.sc_val_3d_moder_7 = logger.scalar("mAP3D_moderate_7")
            self.sc_val_3d_hard_7 = logger.scalar("mAP3D__hard_7")
            self.sc_val_3d_easy_5 = logger.scalar("mAP3D_easy_5")
            self.sc_val_3d_moder_5 = logger.scalar("mAP3D_moderate_5")
            self.sc_val_3d_hard_5 = logger.scalar("mAP3D__hard_5")

    def load_solver(self):
        return self.solver

    def train_model(self):
        #"""执行训练的整个流程，穿插了validation"""
        cur_iter = 0
        while cur_iter < self.solver_param.max_iter:

            for i in range(self.test_interval):

                self.solver.step(1)
                reg_loss = self.solver.net.blobs['reg_loss'].data
                cls_loss = self.solver.net.blobs['cls_loss'].data
                step = self.solver.iter

                self.sc_train_reg_loss.add_record(step, reg_loss)
                self.sc_train_cls_loss.add_record(step, cls_loss) # for logger

            self.eval_on_val()
            cur_iter += self.test_interval

    def eval_on_val(self):
        #"""在整个验证集上执行inference和evaluation"""
        self.solver.test_nets[0].share_with(self.solver.net)
        self.cur_epoch += 1
        for t in range(self.solver_param.test_iter[0]):

            self.solver.test_nets[0].forward()

            map3d_easy_7 = self.solver.test_nets[0].blobs['e7'].data
            map3d_moder_7 = self.solver.test_nets[0].blobs['m7'].data
            map3d_hard_7 = self.solver.test_nets[0].blobs['h7'].data
            map3d_easy_5 = self.solver.test_nets[0].blobs['e5'].data
            map3d_moder_5 = self.solver.test_nets[0].blobs['m5'].data
            map3d_hard_5 = self.solver.test_nets[0].blobs['h5'].data

        # ap, acc = perfeval.cls_eval(scores, gt_labels)
        # print('====================================================================\n')
        print('\nDo validation after the {:d}-th training epoch\n'.format(self.cur_epoch))
        # print('>>>>', end='\t')  #设定标记，方便于解析日志获取出数据
        # for i in range(num_classes):
        #     print('AP[{:d}]={:.2f}'.format(i, ap[i]), end=', ')
        # mAP = np.average(ap)
        # print('mAP={:.2f}, Accuracy={:.2f}'.format(mAP, acc))
        # print('\n====================================================================\n')
        step = self.solver.iter
        self.sc_val_3d_easy_7.add_record(step, map3d_easy_7)
        self.sc_val_3d_moder_7.add_record(step, map3d_moder_7)
        self.sc_val_3d_easy_7.add_record(step, map3d_hard_7)
        self.sc_val_3d_easy_5.add_record(step, map3d_easy_5)
        self.sc_val_3d_moder_5.add_record(step, map3d_moder_5)
        self.sc_val_3d_hard_5.add_record(step, map3d_hard_5)
