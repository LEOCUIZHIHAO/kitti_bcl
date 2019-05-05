import os
import fire
from second.protos import pipeline_pb2

import caffe
from caffe import layers as L, params as P
from models import caffe_model, bcl_model
from solver import solver_function

import shutil
from second.protos import pipeline_pb2
from google.protobuf import text_format #make prototxt work
import pathlib

def load_model_config(model_dir, config_path):

    model_dir = pathlib.Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    config_file_bkp = "pipeline.config"
    config = pipeline_pb2.TrainEvalPipelineConfig()

    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    shutil.copyfile(config_path, str(model_dir / config_file_bkp))

    return config.model.second

def load_recent_model(exp_dir):
    maxit = 0
    file = os.listdir(exp_dir)
    caffemodel = [os.path.splitext(model)[0] for model in file if model.endswith('.caffemodel')]
    if len(caffemodel)==0:
        print("\n[Info] No model existing please retrain")
        exit()
    for idx, model in enumerate(caffemodel):
        ite=int(model.split('_')[-1])
        if ite>maxit:
            maxit=ite
            maxid=idx
    recent_model = caffemodel[maxid]
    if (str(recent_model) + '.solverstate'):
        print("\n[Info] Load existing model {}\n".format(str(recent_model)))
    return recent_model

def caf_model(exp_dir, model_cfg, args, restore, pretrained):
    # if args.cpu:
    #     caffe.set_mode_cpu()
    # else:
    caffe.set_mode_gpu()
    caffe.set_device(0)

    if not restore:
        #save prototxt path
        trian_proto_path = os.path.join(exp_dir, 'train.prototxt')
        eval_proto_path = os.path.join(exp_dir, 'eval.prototxt')
        deploy_proto_path = os.path.join(exp_dir, 'deploy.prototxt')

        train_net = bcl_model.test_v2(phase='train', dataset_params=args, model_cfg=model_cfg)
        eval_net = bcl_model.test_v2(phase='eval', dataset_params=args, model_cfg=model_cfg)

        # train_net = caffe_model.test_v1(phase='train', dataset_params=args, model_cfg=model_cfg)
        # eval_net = caffe_model.test_v1(phase='eval', dataset_params=args, model_cfg=model_cfg)

        with open(trian_proto_path, 'w') as f:
            print(train_net, file=f)

        with open(eval_proto_path, 'w') as f:
            print(eval_net, file=f)

    elif restore:

        try:
            trian_proto_path = os.path.join(exp_dir, 'train.prototxt')
            eval_proto_path = os.path.join(exp_dir, 'eval.prototxt')
            print("[info] Load prototxt from path :", trian_proto_path)
            print("[info] Load prototxt from path :", eval_proto_path)

        except Exception as e:
            print("\n[Info] No prototxt existing please train")
            raise

    # # deploy model use for prediction
    # caffe_model.test_v1(deploy=True, dataset_params=args,)

    solver = solver_function.SolverWrapper(trian_proto_path,
                                           eval_proto_path,
                                           os.path.join(exp_dir, 'pp'),
                                           pretrained=pretrained,
                                           base_lr= 0.0002,
                                           gamma= 0.8, #decay factor
                                           stepsize= 27840, #learning rate decay
                                           test_iter= 3769, # 10 #number of iterations to use at each testing phase 3769
                                           test_interval= 999999999, # 'test every such iterations' 1856 (test every 5 epoches) 9280
                                           max_iter= 296960, # 296960 = 160*1856 #185600
                                           snapshot=1856, # how many steps save a model 9280 (1856*2=3712) save 2 epoches
                                           solver_type='ADAM',
                                           weight_decay= 0.0001, # 0.0001,
                                           iter_size=2, #'number of mini-batches per iteration', batchsize*itersize = real_batch size
                                           display=50,
                                           debug_info=False,
                                           random_seed=19930416,
                                           save_path=os.path.join(exp_dir, 'solver.prototxt'))


    # solver = solver.load_solver()

    if restore:
        recent_model = load_recent_model(exp_dir)
        _solver = solver.load_solver()
        _solver.net.copy_from(os.path.join(exp_dir, "{}.caffemodel".format(str(recent_model))))
        _solver.restore(os.path.join(exp_dir, "{}.solverstate".format(str(recent_model))))

    # solver.solve()
    solver.train_model()

def train(config_path, model_dir, restore=True, pretrained=False):

    args = {}
    args['config_path'] = config_path
    args['model_dir'] = model_dir
    model_cfg = load_model_config(model_dir, config_path) #load layer configs

    caf_model(model_dir, model_cfg, args, restore, pretrained)

if __name__ == '__main__':
    fire.Fire()
