"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import tempfile
from caffe.proto import caffe_pb2
import caffe
import second.data.kitti_common as kitti
# from tqdm import tqdm
# from second.builder import target_assigner_builder, voxel_builder
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
        self.test_interval = 1856  #replace self.solver_param.test_interval #9280

        # self.logw = LogWriter("catdog_log", sync_cycle=100)
        # with self.logw.mode('train') as logger:
        #     self.sc_train_loss = logger.scalar("loss")
        #     self.sc_train_acc = logger.scalar("Accuracy")
        # with self.logw.mode('val') as logger:
        #     self.sc_val_acc = logger.scalar("Accuracy")
        #     self.sc_val_mAP = logger.scalar("mAP")

    def load_solver(self):
        return self.solver


    def train_model(self):
        #"""执行训练的整个流程，穿插了validation"""
        cur_iter = 0
        # test_batch_size, num_classes = self.solver.test_nets[0].blobs['reg_loss'].shape
        # print(self.solver.test_nets[0].blobs['reg_loss'])
        # num_test_images_tot = test_batch_size * self.solver_param.test_iter[0]
        while cur_iter < self.solver_param.max_iter:
            #self.solver.step(self.test_interval)
            print("perpare to start train")
            step=0
            for i in range(self.test_interval):

                # self.solver.net.forward()
                # w = self.solver.net.params["Mlp"][0].data[...]
                #mlp
                torch_mlp_weight_path = '0_voxel_feature_extractor.pfn_layers.0.Conv2d.weight.npy'
                torch_bn_mean_path = '14_voxel_feature_extractor.pfn_layers.0.norm.running_mean.npy'
                torch_bn_var_path = '15_voxel_feature_extractor.pfn_layers.0.norm.running_var.npy'
                torch_sc_weight_path = '1_voxel_feature_extractor.pfn_layers.0.norm.weight.npy'
                torch_sc_bias_path = '2_voxel_feature_extractor.pfn_layers.0.norm.bias.npy'

                #block1
                torch_block1_weight_path = '3_rpn.block1.0.weight.npy'
                torch_block1_bn_mean_path = '16_rpn.block1.1.running_mean.npy'
                torch_block1_bn_var_path = '17_rpn.block1.1.running_var.npy'
                torch_block1_alpha_path = '4_rpn.block1.1.weight.npy'
                torch_block1_belta_path = '5_rpn.block1.1.bias.npy'

                torch_block1_rpncov1_weight_path = '6_rpn.block1.3.weight.npy'
                torch_block1_rpncov1_bn_mean_path = '18_rpn.block1.4.running_mean.npy'
                torch_block1_rpncov1_bn_var_path = '19_rpn.block1.4.running_var.npy'
                torch_block1_rpncov1_alpha_path = '7_rpn.block1.4.weight.npy'
                torch_block1_rpncov1_belta_path = '8_rpn.block1.4.bias.npy'

                torch_block1_decov1_weight_path = '9_rpn.deconv1.0.weight.npy'
                torch_block1_decov1_bn_mean_path = '20_rpn.deconv1.1.running_mean.npy'
                torch_block1_decov1_bn_var_path = '21_rpn.deconv1.1.running_var.npy'
                torch_block1_decov1_alpha_path = '10_rpn.deconv1.1.weight.npy'
                torch_block1_decov1_belta_path = '11_rpn.deconv1.1.bias.npy'

                #result
                torch_cls_weight_path = '12_rpn.conv_cls.weight.npy'
                torch_cls_bias_path = '13_rpn.conv_cls.bias.npy'
                torch_box_weight_path = '14_rpn.conv_box.weight.npy'
                torch_box_bias_path = '15_rpn.conv_box.bias.npy'



                torch_mlp_w = np.load('./weights/'+torch_mlp_weight_path)
                torch_mlp_mean = np.load('./weights/'+torch_bn_mean_path)
                torch_mlp_var = np.load('./weights/'+torch_bn_var_path)
                torch_mlp_alpha = np.load('./weights/'+torch_sc_weight_path)
                torch_mlp_belta = np.load('./weights/'+torch_sc_bias_path)

                #block1
                torch_block1_w = np.load('./weights/'+torch_block1_weight_path)
                torch_block1_mean = np.load('./weights/'+torch_block1_bn_mean_path)
                torch_block1_var = np.load('./weights/'+torch_block1_bn_var_path)
                torch_block1_alpha = np.load('./weights/'+torch_block1_alpha_path)
                torch_block1_belta = np.load('./weights/'+torch_block1_belta_path)

                torch_block1_rpncov1_w = np.load('./weights/'+torch_block1_rpncov1_weight_path)
                torch_block1_rpncov1_mean = np.load('./weights/'+torch_block1_rpncov1_bn_mean_path)
                torch_block1_rpncov1_var = np.load('./weights/'+torch_block1_rpncov1_bn_var_path)
                torch_block1_rpncov1_alpha = np.load('./weights/'+torch_block1_rpncov1_alpha_path)
                torch_block1_rpncov1_belta = np.load('./weights/'+torch_block1_rpncov1_belta_path)

                torch_block1_decov1_w = np.load('./weights/'+torch_block1_decov1_weight_path)
                torch_block1_decov1_mean = np.load('./weights/'+torch_block1_decov1_bn_mean_path)
                torch_block1_decov1_var = np.load('./weights/'+torch_block1_decov1_bn_var_path)
                torch_block1_decov1_alpha = np.load('./weights/'+torch_block1_decov1_alpha_path)
                torch_block1_decov1_belta = np.load('./weights/'+torch_block1_decov1_belta_path)

                #result
                torch_clshead_w = np.load('./weights/'+torch_cls_weight_path)
                torch_clshead_b = np.load('./weights/'+torch_cls_bias_path)
                torch_boxhead_w = np.load('./weights/'+torch_box_weight_path)
                torch_boxhead_b = np.load('./weights/'+torch_box_bias_path)


                #mlp
                self.solver.net.params["Mlp"][0].data[...] = torch_mlp_w ## feed torch weights
                self.solver.net.params["bn1"][0].data[...] = torch_mlp_mean ## feed torch BN mean
                self.solver.net.params["bn1"][1].data[...] = torch_mlp_var ## feed torch BN var
                self.solver.net.params["bn1"][2].data[...] = 1 ## Force set caffe bn[2] ==1 , because caffe using sum weights !
                self.solver.net.params["sc1"][0].data[...] = torch_mlp_alpha ## feed torch SC alpha
                self.solver.net.params["sc1"][1].data[...] = torch_mlp_belta ## feed torch SC belta


                #block1
                self.solver.net.params["init_conv1"][0].data[...] = torch_block1_w ## feed torch weights
                self.solver.net.params["bn2"][0].data[...] = torch_block1_mean ## feed torch BN mean
                self.solver.net.params["bn2"][1].data[...] = torch_block1_var ## feed torch BN var
                self.solver.net.params["bn2"][2].data[...] = 1 ## Force set caffe bn[2] ==1 , because caffe using sum weights !
                self.solver.net.params["sc2"][0].data[...] = torch_block1_alpha ## feed torch SC alpha
                self.solver.net.params["sc2"][1].data[...] = torch_block1_belta ## feed torch SC belta

                self.solver.net.params["rpn_conv1_3"][0].data[...] = torch_block1_rpncov1_w ## feed torch weights
                self.solver.net.params["bn3"][0].data[...] = torch_block1_rpncov1_mean ## feed torch BN mean
                self.solver.net.params["bn3"][1].data[...] = torch_block1_rpncov1_var ## feed torch BN var
                self.solver.net.params["bn3"][2].data[...] = 1 ## Force set caffe bn[2] ==1 , because caffe using sum weights !
                self.solver.net.params["sc3"][0].data[...] = torch_block1_rpncov1_alpha ## feed torch SC alpha
                self.solver.net.params["sc3"][1].data[...] = torch_block1_rpncov1_belta ## feed torch SC belta

                self.solver.net.params["rpn_deconv1"][0].data[...] = torch_block1_decov1_w ## feed torch weights
                self.solver.net.params["bn4"][0].data[...] = torch_block1_decov1_mean ## feed torch BN mean
                self.solver.net.params["bn4"][1].data[...] = torch_block1_decov1_var ## feed torch BN var
                self.solver.net.params["bn4"][2].data[...] = 1 ## Force set caffe bn[2] ==1 , because caffe using sum weights !
                self.solver.net.params["sc4"][0].data[...] = torch_block1_decov1_alpha ## feed torch SC alpha
                self.solver.net.params["sc4"][1].data[...] = torch_block1_decov1_belta ## feed torch SC belta


                #result
                self.solver.net.params["cls_head"][0].data[...] = torch_clshead_w ## feed torch weights
                self.solver.net.params["cls_head"][1].data[...] = torch_clshead_b ## feed torch weights
                self.solver.net.params["reg_head"][0].data[...] = torch_boxhead_w ## feed torch weights
                self.solver.net.params["reg_head"][1].data[...] = torch_boxhead_b ## feed torch weights

                # print("1st", torch_mlp_w)

                # self.solver.step(1) #forward + backward + update weights
                self.solver.step(1)
                # print("second", self.solver.net.params["Mlp"][0].data[...])


                step+=1
                if step == 2:
                    exit()
                # reg_loss = self.solver.net.blobs['reg_loss'].data
                # cls_loss = self.solver.net.blobs['cls_loss'].data

                # step = self.solver.iter
                # self.sc_train_loss.add_record(step, loss)
                # self.sc_train_acc.add_record(step, acc) # for logger

            cur_iter += self.test_interval

    def eval_on_val(self):
        #"""在整个验证集上执行inference和evaluation"""
        self.solver.test_nets[0].share_with(self.solver.net)
        self.cur_epoch += 1
        for t in range(self.solver_param.test_iter[0]):

            self.solver.test_nets[0].forward()
            # output = self.solver.test_nets[0].blobs
            # box_preds = output['box_preds'].data
            # cls_preds = output['cls_preds'].data

            #self.predict(batch_box_preds = box_preds, batch_cls_preds = cls_preds)

    # def predict(self, example, batch_box_preds, batch_cls_preds):
    #     t = time.time()
    #     batch_size = 1 # TODO: pass
    #     batch_anchors = example["anchors"].view(batch_size, -1, 7)
    #     self._box_coder = target_assigner.box_coder
    #
    #     self._total_inference_count += batch_size
    #     batch_rect = example["rect"]
    #     batch_Trv2c = example["Trv2c"]
    #     batch_P2 = example["P2"]
    #     if "anchors_mask" not in example:
    #         batch_anchors_mask = [None] * batch_size
    #     else:
    #         batch_anchors_mask = example["anchors_mask"].view(batch_size, -1)
    #     batch_imgidx = example['image_idx']
    #
    #     self._total_forward_time += time.time() - t
    #     t = time.time()
    #
    #     batch_box_preds = batch_box_preds.reshape(batch_size, -1,
    #                                            self._box_coder.code_size)
    #     num_class_with_bg = self._num_class
    #     if not self._encode_background_as_zeros:
    #         num_class_with_bg = self._num_class + 1
    #
    #     batch_cls_preds = batch_cls_preds.reshape(batch_size, -1,
    #                                            num_class_with_bg)
    #     batch_box_preds = self._box_coder.decode_torch(batch_box_preds,
    #                                                    batch_anchors)
    #     if self._use_direction_classifier:
    #         batch_dir_preds = preds_dict["dir_cls_preds"]
    #         batch_dir_preds = batch_dir_preds.view(batch_size, -1, 2)
    #     else:
    #         batch_dir_preds = [None] * batch_size
    #
    #     predictions_dicts = []
    #     for box_preds, cls_preds, dir_preds, rect, Trv2c, P2, img_idx, a_mask in zip(
    #             batch_box_preds, batch_cls_preds, batch_dir_preds, batch_rect,
    #             batch_Trv2c, batch_P2, batch_imgidx, batch_anchors_mask
    #     ):
    #         if a_mask is not None:
    #             box_preds = box_preds[a_mask]
    #             cls_preds = cls_preds[a_mask]
    #         if self._use_direction_classifier:
    #             if a_mask is not None:
    #                 dir_preds = dir_preds[a_mask]
    #             # print(dir_preds.shape)
    #             dir_labels = torch.max(dir_preds, dim=-1)[1]
    #         if self._encode_background_as_zeros:
    #             # this don't support softmax
    #             assert self._use_sigmoid_score is True
    #             total_scores = torch.sigmoid(cls_preds)
    #         else:
    #             # encode background as first element in one-hot vector
    #             if self._use_sigmoid_score:
    #                 total_scores = torch.sigmoid(cls_preds)[..., 1:]
    #             else:
    #                 total_scores = F.softmax(cls_preds, dim=-1)[..., 1:]
    #         # Apply NMS in birdeye view
    #         if self._use_rotate_nms:
    #             nms_func = box_torch_ops.rotate_nms
    #         else:
    #             nms_func = box_torch_ops.nms
    #         selected_boxes = None
    #         selected_labels = None
    #         selected_scores = None
    #         selected_dir_labels = None
    #
    #         if self._multiclass_nms:
    #             # curently only support class-agnostic boxes.
    #             boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
    #             if not self._use_rotate_nms:
    #                 box_preds_corners = box_torch_ops.center_to_corner_box2d(
    #                     boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
    #                     boxes_for_nms[:, 4])
    #                 boxes_for_nms = box_torch_ops.corner_to_standup_nd(
    #                     box_preds_corners)
    #             boxes_for_mcnms = boxes_for_nms.unsqueeze(1)
    #             selected_per_class = box_torch_ops.multiclass_nms(
    #                 nms_func=nms_func,
    #                 boxes=boxes_for_mcnms,
    #                 scores=total_scores,
    #                 num_class=self._num_class,
    #                 pre_max_size=self._nms_pre_max_size,
    #                 post_max_size=self._nms_post_max_size,
    #                 iou_threshold=self._nms_iou_threshold,
    #                 score_thresh=self._nms_score_threshold,
    #             )
    #             selected_boxes, selected_labels, selected_scores = [], [], []
    #             selected_dir_labels = []
    #             for i, selected in enumerate(selected_per_class):
    #                 if selected is not None:
    #                     num_dets = selected.shape[0]
    #                     selected_boxes.append(box_preds[selected])
    #                     selected_labels.append(
    #                         torch.full([num_dets], i, dtype=torch.int64))
    #                     if self._use_direction_classifier:
    #                         selected_dir_labels.append(dir_labels[selected])
    #                     selected_scores.append(total_scores[selected, i])
    #             if len(selected_boxes) > 0:
    #                 selected_boxes = torch.cat(selected_boxes, dim=0)
    #                 selected_labels = torch.cat(selected_labels, dim=0)
    #                 selected_scores = torch.cat(selected_scores, dim=0)
    #                 if self._use_direction_classifier:
    #                     selected_dir_labels = torch.cat(
    #                         selected_dir_labels, dim=0)
    #             else:
    #                 selected_boxes = None
    #                 selected_labels = None
    #                 selected_scores = None
    #                 selected_dir_labels = None
    #         else:
    #             # get highest score per prediction, than apply nms
    #             # to remove overlapped box.
    #             if num_class_with_bg == 1:
    #                 top_scores = total_scores.squeeze(-1)
    #                 top_labels = torch.zeros(
    #                     total_scores.shape[0],
    #                     device=total_scores.device,
    #                     dtype=torch.long)
    #             else:
    #                 top_scores, top_labels = torch.max(total_scores, dim=-1)
    #
    #             if self._nms_score_threshold > 0.0:
    #                 thresh = torch.tensor(
    #                     [self._nms_score_threshold],
    #                     device=total_scores.device).type_as(total_scores)
    #                 top_scores_keep = (top_scores >= thresh)
    #                 top_scores = top_scores.masked_select(top_scores_keep)
    #             if top_scores.shape[0] != 0:
    #                 if self._nms_score_threshold > 0.0:
    #                     box_preds = box_preds[top_scores_keep]
    #                     if self._use_direction_classifier:
    #                         dir_labels = dir_labels[top_scores_keep]
    #                     top_labels = top_labels[top_scores_keep]
    #                 boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
    #                 if not self._use_rotate_nms:
    #                     box_preds_corners = box_torch_ops.center_to_corner_box2d(
    #                         boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
    #                         boxes_for_nms[:, 4])
    #                     boxes_for_nms = box_torch_ops.corner_to_standup_nd(
    #                         box_preds_corners)
    #                 # the nms in 3d detection just remove overlap boxes.
    #                 selected = nms_func(
    #                     boxes_for_nms,
    #                     top_scores,
    #                     pre_max_size=self._nms_pre_max_size,
    #                     post_max_size=self._nms_post_max_size,
    #                     iou_threshold=self._nms_iou_threshold,
    #                 )
    #             else:
    #                 selected = None
    #             if selected is not None:
    #                 selected_boxes = box_preds[selected]
    #                 if self._use_direction_classifier:
    #                     selected_dir_labels = dir_labels[selected]
    #                 selected_labels = top_labels[selected]
    #                 selected_scores = top_scores[selected]
    #         # finally generate predictions.
    #
    #         if selected_boxes is not None:
    #             box_preds = selected_boxes
    #             scores = selected_scores
    #             label_preds = selected_labels
    #             if self._use_direction_classifier:
    #                 dir_labels = selected_dir_labels
    #                 opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.byte()
    #                 box_preds[..., -1] += torch.where(
    #                     opp_labels,
    #                     torch.tensor(np.pi).type_as(box_preds),
    #                     torch.tensor(0.0).type_as(box_preds))
    #                 # box_preds[..., -1] += (
    #                 #     ~(dir_labels.byte())).type_as(box_preds) * np.pi
    #             final_box_preds = box_preds
    #             final_scores = scores
    #             final_labels = label_preds
    #             final_box_preds_camera = box_torch_ops.box_lidar_to_camera(
    #                 final_box_preds, rect, Trv2c)
    #             locs = final_box_preds_camera[:, :3]
    #             dims = final_box_preds_camera[:, 3:6]
    #             angles = final_box_preds_camera[:, 6]
    #             camera_box_origin = [0.5, 1.0, 0.5]
    #             box_corners = box_torch_ops.center_to_corner_box3d(
    #                 locs, dims, angles, camera_box_origin, axis=1)
    #             box_corners_in_image = box_torch_ops.project_to_image(
    #                 box_corners, P2)
    #             # box_corners_in_image: [N, 8, 2]
    #             minxy = torch.min(box_corners_in_image, dim=1)[0]
    #             maxxy = torch.max(box_corners_in_image, dim=1)[0]
    #             # minx = torch.min(box_corners_in_image[..., 0], dim=1)[0]
    #             # maxx = torch.max(box_corners_in_image[..., 0], dim=1)[0]
    #             # miny = torch.min(box_corners_in_image[..., 1], dim=1)[0]
    #             # maxy = torch.max(box_corners_in_image[..., 1], dim=1)[0]
    #             # box_2d_preds = torch.stack([minx, miny, maxx, maxy], dim=1)
    #             box_2d_preds = torch.cat([minxy, maxxy], dim=1)
    #             # predictions
    #             predictions_dict = {
    #                 "bbox": box_2d_preds,
    #                 "box3d_camera": final_box_preds_camera,
    #                 "box3d_lidar": final_box_preds,
    #                 "scores": final_scores,
    #                 "label_preds": label_preds,
    #                 "image_idx": img_idx,
    #             }
    #         else:
    #             predictions_dict = {
    #                 "bbox": None,
    #                 "box3d_camera": None,
    #                 "box3d_lidar": None,
    #                 "scores": None,
    #                 "label_preds": None,
    #                 "image_idx": img_idx,
    #             }
    #         predictions_dicts.append(predictions_dict)
    #     self._total_postprocess_time += time.time() - t
    #     return predictions_dicts

def predict_kitti_to_anno(
                          example,
                          class_names,
                          center_limit_range=None,
                          lidar_input=False,
                          global_set=None):

    class_names = list(["Car"]) # class_names = list(input_cfg.class_names)
    center_limit_range = [0, -39.68, -5, 69.12, 39.68, 5] # center_limit_range = model_cfg.post_center_limit_range
    lidar_input=False

    batch_image_shape = example['image_shape']
    # t = time.time()
    annos = []
    for i, preds_dict in enumerate(predictions_dicts):
        image_shape = batch_image_shape[i]
        img_idx = preds_dict["image_idx"]
        if preds_dict["bbox"] is not None:
            box_2d_preds = preds_dict["bbox"].detach().cpu().numpy()
            box_preds = preds_dict["box3d_camera"].detach().cpu().numpy()
            scores = preds_dict["scores"].detach().cpu().numpy()
            box_preds_lidar = preds_dict["box3d_lidar"].detach().cpu().numpy()
            # write pred to file
            label_preds = preds_dict["label_preds"].detach().cpu().numpy()
            # label_preds = np.zeros([box_2d_preds.shape[0]], dtype=np.int32)
            anno = kitti.get_start_result_anno()
            num_example = 0
            for box, box_lidar, bbox, score, label in zip(
                    box_preds, box_preds_lidar, box_2d_preds, scores,
                    label_preds):
                if not lidar_input:
                    if bbox[0] > image_shape[1] or bbox[1] > image_shape[0]:
                        continue
                    if bbox[2] < 0 or bbox[3] < 0:
                        continue
                # print(img_shape)
                if center_limit_range is not None:
                    limit_range = np.array(center_limit_range)
                    if (np.any(box_lidar[:3] < limit_range[:3])
                            or np.any(box_lidar[:3] > limit_range[3:])):
                        continue
                bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                bbox[:2] = np.maximum(bbox[:2], [0, 0])
                anno["name"].append(class_names[int(label)])
                anno["truncated"].append(0.0)
                anno["occluded"].append(0)
                anno["alpha"].append(-np.arctan2(-box_lidar[1], box_lidar[0]) +
                                     box[6])
                anno["bbox"].append(bbox)
                anno["dimensions"].append(box[3:6])
                anno["location"].append(box[:3])
                anno["rotation_y"].append(box[6])
                if global_set is not None:
                    for i in range(100000):
                        if score in global_set:
                            score -= 1 / 100000
                        else:
                            global_set.add(score)
                            break
                anno["score"].append(score)

                num_example += 1
            if num_example != 0:
                anno = {n: np.stack(v) for n, v in anno.items()}
                annos.append(anno)
            else:
                annos.append(kitti.empty_result_anno())
        else:
            annos.append(kitti.empty_result_anno())
        num_example = annos[-1]["name"].shape[0]
        annos[-1]["image_idx"] = np.array(
            [img_idx] * num_example, dtype=np.int64)
    return annos




        # ap, acc = perfeval.cls_eval(scores, gt_labels)
        # print('====================================================================\n')
        # print('\tDo validation after the {:d}-th training epoch\n'.format(self.cur_epoch))
        # print('>>>>', end='\t')  #设定标记，方便于解析日志获取出数据
        # for i in range(num_classes):
        #     print('AP[{:d}]={:.2f}'.format(i, ap[i]), end=', ')
        # mAP = np.average(ap)
        # print('mAP={:.2f}, Accuracy={:.2f}'.format(mAP, acc))
        # print('\n====================================================================\n')
        # step = self.solver.iter
        # self.sc_val_mAP.add_record(step, mAP)
        # self.sc_val_acc.add_record(step, acc)
