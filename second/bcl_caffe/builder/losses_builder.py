from core import losses
from second.protos import losses_pb2

def _build_loss(loss_config):

    classification_loss = _build_classification_loss(
        loss_config.classification_loss)
    localization_loss = _build_localization_loss(
        loss_config.localization_loss)
    classification_weight = loss_config.classification_weight
    localization_weight = loss_config.localization_weight
    hard_example_miner = None
    if loss_config.HasField('hard_example_miner'):
      raise ValueError('Pytorch don\'t support HardExampleMiner')

    return (classification_loss, localization_loss,
            classification_weight,
            localization_weight, hard_example_miner)

def _build_localization_loss(loss_config):
  """Builds a localization loss based on the loss config.

  Args:
    loss_config: A losses_pb2.LocalizationLoss object.

  Returns:
    Loss based on the config.

  Raises:
    ValueError: On invalid loss_config.
  """
  if not isinstance(loss_config, losses_pb2.LocalizationLoss):
    raise ValueError('loss_config not of type losses_pb2.LocalizationLoss.')

  loss_type = loss_config.WhichOneof('localization_loss')

  if loss_type == 'weighted_l2':
    config = loss_config.weighted_l2
    if len(config.code_weight) == 0:
      code_weight = None
    else:
      code_weight = config.code_weight
    return losses.WeightedL2LocalizationLoss(code_weight)

  if loss_type == 'weighted_smooth_l1':
    config = loss_config.weighted_smooth_l1
    if len(config.code_weight) == 0:
      code_weight = None
    else:
      code_weight = config.code_weight
    return losses.WeightedSmoothL1LocalizationLoss(config.sigma, code_weight)

  raise ValueError('Empty loss config.')

def _build_classification_loss(loss_config):
  """Builds a classification loss based on the loss config.

  Args:
    loss_config: A losses_pb2.ClassificationLoss object.

  Returns:
    Loss based on the config.

  Raises:
    ValueError: On invalid loss_config.
  """
  if not isinstance(loss_config, losses_pb2.ClassificationLoss):
    raise ValueError('loss_config not of type losses_pb2.ClassificationLoss.')

  loss_type = loss_config.WhichOneof('classification_loss')

  if loss_type == 'weighted_sigmoid':
    return losses.WeightedSigmoidClassificationLoss()

  if loss_type == 'weighted_sigmoid_focal':
    config = loss_config.weighted_sigmoid_focal
    # alpha = None
    # if config.HasField('alpha'):
    #   alpha = config.alpha
    if config.alpha > 0:
      alpha = config.alpha
    else:
      alpha = None
    return losses.SigmoidFocalClassificationLoss(
        gamma=config.gamma,
        alpha=alpha)
  if loss_type == 'weighted_softmax_focal':
    config = loss_config.weighted_softmax_focal
    # alpha = None
    # if config.HasField('alpha'):
    #   alpha = config.alpha
    if config.alpha > 0:
      alpha = config.alpha
    else:
      alpha = None
    return losses.SoftmaxFocalClassificationLoss(
        gamma=config.gamma,
        alpha=alpha)

  if loss_type == 'weighted_softmax':
    config = loss_config.weighted_softmax
    return losses.WeightedSoftmaxClassificationLoss(
        logit_scale=config.logit_scale)

  if loss_type == 'bootstrapped_sigmoid':
    config = loss_config.bootstrapped_sigmoid
    return losses.BootstrappedSigmoidClassificationLoss(
        alpha=config.alpha,
        bootstrap_type=('hard' if config.hard_bootstrap else 'soft'))

  raise ValueError('Empty loss config.')
