import os
import pathlib
import pickle
import shutil

import fire
from second.protos import pipeline_pb2

import caffe
from caffe import layers as L, params as P
from models import caffe_model
from solver import solver_function

#caffe model prepare
def caf_model(exp_dir, args):
    # if args.cpu:
    #     caffe.set_mode_cpu()
    # else:
    caffe.set_mode_gpu()
    caffe.set_device(0)

    trian_proto_path = os.path.join(exp_dir, 'train.prototxt')
    eval_proto_path = os.path.join(exp_dir, 'eval.prototxt')
    deploy_proto_path = os.path.join(exp_dir, 'deploy.prototxt')

    train_net = caffe_model.test_v1(phase='train', dataset_params=args)
    eval_net = caffe_model.test_v1(phase='eval', dataset_params=args)

    with open(trian_proto_path, 'w') as f:
        print(train_net, file=f)

    with open(eval_proto_path, 'w') as f:
        print(eval_net, file=f)

    # # deploy model use for prediction
    # caffe_model.test_v1(deploy=True, dataset_params=args,)

    random_seed = 0
    debug_info = False

    solver = solver_function.SolverWrapper(trian_proto_path,
                                           eval_proto_path,
                                           os.path.join(exp_dir, 'snapshot'),
                                           base_lr= 0.0002,
                                           gamma= 0.8,
                                           stepsize= 27840, #learning rate decay
                                           test_iter= 50, # 10 #number of iterations to use at each testing phase 3769
                                           test_interval= 50, # 'test every such iterations' 1856
                                           max_iter= 185600, # 296960 = 160*1856
                                           snapshot=9280, # how many steps save a model 9280
                                           solver_type='ADAM',
                                           weight_decay= 0.0001, # 0.0001,
                                           iter_size=2, #'number of mini-batches per iteration', batchsize*itersize = real_batch size
                                           display = 1,
                                           debug_info=debug_info,
                                           random_seed=random_seed,
                                           save_path=os.path.join(exp_dir, 'solver.prototxt'))

    # solver.train_model() #for debug purpose
    solver.train_solver().solve()

    print("[CAFF SOLVER INIT ] ")

    # if args.init_model:
    #     if args.init_model.endswith('.caffemodel'):
    #         solver.net.copy_from(args.init_model)
    #     else:
    #         solver.net.copy_from(os.path.join(exp_dir, 'snapshot_iter_{}.caffemodel'.format(args.init_model)))
    #
    # if args.init_state:
    #     if args.init_state.endswith('.solverstate'):
    #         solver.restore(args.init_state)
    #     else:
    #         solver.restore(os.path.join(exp_dir, 'snapshot_iter_{}.solverstate'.format(args.init_state)))
    #solver.solve()
    print("[CAFF SOLVER DONE DONE DONE ] ")

def train(config_path,
          model_dir,
          result_path=None,
          create_folder=False,
          display_step=50,
          summary_step=5,
          pickle_result=True):

    args = {}
    args['config_path'] = config_path
    args['model_dir'] = model_dir
    caf_model('/home/ubuntu/kitti_bcl/second/bcl_caffe', args)

if __name__ == '__main__':
    fire.Fire()
