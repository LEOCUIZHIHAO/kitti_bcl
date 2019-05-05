import os
import tempfile
from caffe.proto import caffe_pb2
import caffe
from tqdm import tqdm
# from visualdl import LogWriter
import numpy as np

def get_prototxt(solver_proto, save_path=None):
    if save_path:
        f = open(save_path, mode='w+')
    else:
        f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.write(str(solver_proto))
    f.close()

    return f.name

class SolverWrapper:
    def __init__(self,  train_net,
                        test_net,
                        prefix,
                        pretrained,
                        solver_type='ADAM',
                        weight_decay=0.001,
                        base_lr=0.002,
                        gamma=0.8, #0.1 for lr_policy
                        stepsize=100,
                        test_iter=3769,
                        test_interval=50, #set test_interval to 999999999 if not it will auto run validation
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
            print(solver_prototxt)

        self.solver = caffe.get_solver(solver_prototxt)

        self.pretrained = pretrained
        self.cur_epoch = 0
        self.test_interval = 1856*2 #1856  #replace self.solver_param.test_interval #9280

        # self.logw = LogWriter("permutohedral_log", sync_cycle=100)
        # with self.logw.mode('train') as logger:
        #     self.sc_train_reg_loss = logger.scalar("reg_loss")
        #     self.sc_train_cls_loss = logger.scalar("cls_loss")
        # with self.logw.mode('val') as logger:
        #     self.sc_val_3d_easy_7 = logger.scalar("mAP3D_easy_7")
        #     self.sc_val_3d_moder_7 = logger.scalar("mAP3D_moderate_7")
        #     self.sc_val_3d_hard_7 = logger.scalar("mAP3D__hard_7")
        #     self.sc_val_3d_easy_5 = logger.scalar("mAP3D_easy_5")
        #     self.sc_val_3d_moder_5 = logger.scalar("mAP3D_moderate_5")
        #     self.sc_val_3d_hard_5 = logger.scalar("mAP3D__hard_5")

    def load_solver(self):
        return self.solver

    def train_model(self):

        if self.pretrained:
            print("\n[info] Load Pretrained Model\n")
            self.load_pretrained_weight()

        cur_iter = 0
        while cur_iter < self.solver_param.max_iter:

            for i in range(self.test_interval):

                self.solver.step(1)
                reg_loss = self.solver.net.blobs['reg_loss'].data
                cls_loss = self.solver.net.blobs['cls_loss'].data
                step = self.solver.iter

                # self.sc_train_reg_loss.add_record(step, reg_loss)
                # self.sc_train_cls_loss.add_record(step, cls_loss) # for logger

            self.eval_on_val()
            cur_iter += self.test_interval

    def eval_on_val(self):
        #"""在整个验证集上执行inference和evaluation"""
        self.solver.test_nets[0].share_with(self.solver.net)
        self.cur_epoch += 1

        for t in tqdm(range(self.solver_param.test_iter[0])):
            self.solver.test_nets[0].forward()

        map3d_easy_7 = self.solver.test_nets[0].blobs['e7'].data
        map3d_moder_7 = self.solver.test_nets[0].blobs['m7'].data
        map3d_hard_7 = self.solver.test_nets[0].blobs['h7'].data
        map3d_easy_5 = self.solver.test_nets[0].blobs['e5'].data
        map3d_moder_5 = self.solver.test_nets[0].blobs['m5'].data
        map3d_hard_5 = self.solver.test_nets[0].blobs['h5'].data

        step = self.solver.iter
        # self.sc_val_3d_easy_7.add_record(step, map3d_easy_7)
        # self.sc_val_3d_moder_7.add_record(step, map3d_moder_7)
        # self.sc_val_3d_easy_7.add_record(step, map3d_hard_7)
        # self.sc_val_3d_easy_5.add_record(step, map3d_easy_5)
        # self.sc_val_3d_moder_5.add_record(step, map3d_moder_5)
        # self.sc_val_3d_hard_5.add_record(step, map3d_hard_5)

        print('\nDo validation after the {:d}-th training epoch\n'.format(self.cur_epoch))

    def prepare_pretrained(self, weights_path, layer_name):
        weights = os.listdir(weights_path)

        graph = [w for w in weights if layer_name in w]
        layer_key = [int(m.split('_')[0]) for m in graph]
        layer_dict = dict(zip(layer_key, graph))

        keys = layer_dict.keys()
        keys = sorted(keys)
        layer_ordered = []
        for k in keys:
            layer_ordered.append(layer_dict[k])
        # print(layer_dict)
        # print(layer_ordered) #debug
        return layer_ordered

    def load_pretrained_weight(self):

        weights_path = './output/model_weights/'
        layer_name = 'voxel_feature'
        mlp_Layer = self.prepare_pretrained(weights_path, layer_name)

        self.solver.net.params['mlp_0'][0].data[...] = np.load(weights_path + mlp_Layer[0])[:,:4,:,:] #w
        self.solver.net.params['mlp_sc_0'][0].data[...] = np.load(weights_path + mlp_Layer[1]) #w
        self.solver.net.params['mlp_sc_0'][1].data[...] = np.load(weights_path + mlp_Layer[2]) #b
        self.solver.net.params['mlp_bn_0'][0].data[...] = np.load(weights_path + mlp_Layer[3]) #mean
        self.solver.net.params['mlp_bn_0'][1].data[...] = np.load(weights_path + mlp_Layer[4]) #var
        self.solver.net.params['mlp_bn_0'][2].data[...] = 1

        def rpn_layer(layer_name, block, idx, stride):
            self.solver.net.params[str(layer_name[0])][0].data[...] = np.load(weights_path + block[idx*stride]) #w

            self.solver.net.params[str(layer_name[1])][0].data[...] = np.load(weights_path + block[idx*stride+1]) #w
            self.solver.net.params[str(layer_name[1])][1].data[...] = np.load(weights_path + block[idx*stride+2]) #b

            self.solver.net.params[str(layer_name[2])][0].data[...] = np.load(weights_path + block[idx*stride+3]) #mean
            self.solver.net.params[str(layer_name[2])][1].data[...] = np.load(weights_path + block[idx*stride+4]) #var
            self.solver.net.params[str(layer_name[2])][2].data[...] = 1

        ##################################block1################################
        layer_name = 'rpn.block1'
        rpn_block = self.prepare_pretrained(weights_path, layer_name)
        # print(rpn_block)
        caffe_layre = ['ini_conv1_0', 'ini_conv1_sc_0', 'ini_conv1_bn_0']
        rpn_layer(caffe_layre, rpn_block, idx=0, stride=5)
        # caffe_layre = ['rpn_conv1_0', 'rpn_conv1_sc_0', 'rpn_conv1_bn_0']
        # rpn_layer(caffe_layre, rpn_block, idx=1, stride=5)
        # caffe_layre = ['rpn_conv1_1', 'rpn_conv1_sc_1', 'rpn_conv1_bn_1']
        # rpn_layer(caffe_layre, rpn_block, idx=2, stride=5)
        # caffe_layre = ['rpn_conv1_2', 'rpn_conv1_sc_2', 'rpn_conv1_bn_2']
        # rpn_layer(caffe_layre, rpn_block, idx=3, stride=5)

        for idx in range(3):
            caffe_layre = ['rpn_conv1_{}'.format(idx), 'rpn_conv1_sc_{}'.format(idx), 'rpn_conv1_bn_{}'.format(idx)]
            rpn_layer(caffe_layre, rpn_block, idx=idx+1, stride=5)

        layer_name = 'rpn.deconv1'
        rpn_deconv = self.prepare_pretrained(weights_path, layer_name)
        # print(rpn_deconv)
        caffe_layre = ['rpn_deconv1', 'rpn_deconv1_sc', 'rpn_deconv1_bn']
        rpn_layer(caffe_layre, rpn_deconv, idx=0, stride=5)



        ##################################block2################################
        layer_name = 'rpn.block2'
        rpn_block = self.prepare_pretrained(weights_path, layer_name)
        # print(rpn_block)
        caffe_layre = ['ini_conv2_0', 'ini_conv2_sc_0', 'ini_conv2_bn_0']
        rpn_layer(caffe_layre, rpn_block, idx=0, stride=5)
        # caffe_layre = ['rpn_conv2_0', 'rpn_conv2_sc_0', 'rpn_conv2_bn_0']
        # rpn_layer(caffe_layre, rpn_block, idx=1, stride=5)
        # caffe_layre = ['rpn_conv2_1', 'rpn_conv2_sc_1', 'rpn_conv2_bn_1']
        # rpn_layer(caffe_layre, rpn_block, idx=2, stride=5)
        # caffe_layre = ['rpn_conv2_2', 'rpn_conv2_sc_2', 'rpn_conv2_bn_2']
        # rpn_layer(caffe_layre, rpn_block, idx=3, stride=5)
        # caffe_layre = ['rpn_conv2_3', 'rpn_conv2_sc_3', 'rpn_conv2_bn_3']
        # rpn_layer(caffe_layre, rpn_block, idx=4, stride=5)
        # caffe_layre = ['rpn_conv2_4', 'rpn_conv2_sc_4', 'rpn_conv2_bn_4']
        # rpn_layer(caffe_layre, rpn_block, idx=5, stride=5)

        for idx in range(5):
            caffe_layre = ['rpn_conv2_{}'.format(idx), 'rpn_conv2_sc_{}'.format(idx), 'rpn_conv2_bn_{}'.format(idx)]
            rpn_layer(caffe_layre, rpn_block, idx=idx+1, stride=5)

        layer_name = 'rpn.deconv2'
        rpn_deconv = self.prepare_pretrained(weights_path, layer_name)
        # print(rpn_deconv)
        caffe_layre = ['rpn_deconv2', 'rpn_deconv2_sc', 'rpn_deconv2_bn']
        rpn_layer(caffe_layre, rpn_deconv, idx=0, stride=5)


        ##################################block3################################
        layer_name = 'rpn.block3'
        rpn_block = self.prepare_pretrained(weights_path, layer_name)
        # print(rpn_block)
        caffe_layre = ['ini_conv3_0', 'ini_conv3_sc_0', 'ini_conv3_bn_0']
        rpn_layer(caffe_layre, rpn_block, idx=0, stride=5)
        # caffe_layre = ['rpn_conv3_0', 'rpn_conv3_sc_0', 'rpn_conv3_bn_0']
        # rpn_layer(caffe_layre, rpn_block, idx=1, stride=5)
        # caffe_layre = ['rpn_conv3_1', 'rpn_conv3_sc_1', 'rpn_conv3_bn_1']
        # rpn_layer(caffe_layre, rpn_block, idx=2, stride=5)
        # caffe_layre = ['rpn_conv3_2', 'rpn_conv3_sc_2', 'rpn_conv3_bn_2']
        # rpn_layer(caffe_layre, rpn_block, idx=3, stride=5)
        # caffe_layre = ['rpn_conv3_3', 'rpn_conv3_sc_3', 'rpn_conv3_bn_3']
        # rpn_layer(caffe_layre, rpn_block, idx=4, stride=5)
        # caffe_layre = ['rpn_conv3_4', 'rpn_conv3_sc_4', 'rpn_conv3_bn_4']
        # rpn_layer(caffe_layre, rpn_block, idx=5, stride=5)

        for idx in range(5):
            caffe_layre = ['rpn_conv3_{}'.format(idx), 'rpn_conv3_sc_{}'.format(idx), 'rpn_conv3_bn_{}'.format(idx)]
            rpn_layer(caffe_layre, rpn_block, idx=idx+1, stride=5)

        layer_name = 'rpn.deconv3'
        rpn_deconv = self.prepare_pretrained(weights_path, layer_name)
        # print(rpn_deconv)
        caffe_layre = ['rpn_deconv3', 'rpn_deconv3_sc', 'rpn_deconv3_bn']
        rpn_layer(caffe_layre, rpn_deconv, idx=0, stride=5)


        ################################# Head ################################
        layer_name = 'rpn.conv'
        rpn_conv = self.prepare_pretrained(weights_path, layer_name)
        self.solver.net.params['cls_head'][0].data[...] = np.load(weights_path + rpn_conv[0]) #w cls_head
        self.solver.net.params['cls_head'][1].data[...] = np.load(weights_path + rpn_conv[1]) #b cls_head
        self.solver.net.params['reg_head'][0].data[...] = np.load(weights_path + rpn_conv[2]) #w box_head
        self.solver.net.params['reg_head'][1].data[...] = np.load(weights_path + rpn_conv[3]) #b box_head
