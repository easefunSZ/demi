'''''
--- Computes a Huber loss for neural net training and evaluation.
-- 
-- Computes the loss across buckets, but only on buckets that are
-- possible on a given board.
-- @classmod masked_huber_loss
'''''

require 'nn'
arguments = require 'Settings.arguments'

class MaskedHuberLoss(object):

  ##--- Constructor
  def __init__(self):
    self.criterion = nn.SmoothL1Criterion()

  '''
  --- Moves the torch criterion (used for loss and gradient computation)
  -- to the GPU.
  -- @return the MaskedHuberLoss object that `cuda()` is called on
  '''
  def cuda(self):
    self.criterion = self.criterion.cuda()
    return self

  '''
  --- Computes the loss over a batch of neural net outputs and targets.
  --
  -- @param outputs an NxM tensor containing N vectors of values over buckets,
  -- output by the neural net
  -- @param targets an NxM tensor containing N vectors of actual values over
  -- buckets, produced by @{data_generation_call}
  -- @param mask an NxM tensor containing N mask vectors generated with
  -- @{bucket_conversion.get_possible_bucket_mask}
  -- @return the sum of Huber loss applied elementwise on `outputs` and `targets`,
  -- masked so that only valid buckets are included
  '''
  def forward(self,outputs, targets, mask):

    batch_size = outputs.size(1)
    feature_size = outputs.size(2)

    #--1.0 zero out the outputs/target so that the error does not depend on these
    outputs.cmul(mask)
    targets.cmul(mask)

    loss = self.criterion.forward(outputs, targets)

    #--2.0 if the batch size has changed, create new storage for the sum, otherwise reuse
    if not self.mask_sum or (self.mask_sum.size(1) != batch_size):
      self.mask_placeholder = arguments.Tensor(mask.size()).fill(0)
      self.mask_sum = arguments.Tensor(batch_size).fill(0)
      self.mask_multiplier = self.mask_sum.clone().fill(0).view(-1, 1)

    #--3.0 compute mask sum for each batch
    self.mask_placeholder.copy(mask)
    torch.sum(self.mask_sum, self.mask_placeholder, 2)

    #--3.1 mask multiplier - note that mask is 1 for impossible features
    self.mask_multiplier.fill(feature_size)
    self.mask_multiplier.csub(self.mask_sum)
    self.mask_multiplier.div(feature_size)

    # --4.0 multiply to get a new losss
    # --loss is not really computed batch-wise correctly,
    # --but that does not really matter now since gradients are correct
    loss_multiplier = (batch_size * feature_size) / (batch_size * feature_size - self.mask_sum.sum() )
    new_loss = loss_multiplier * loss
    return new_loss

  '''
  
  --- Computes the gradient of the loss function @{forward} with
  -- arguments `outputs`, `targets`, and `mask`.
  --
  -- Must be called after a @{forward} call with the same arguments.
  --
  -- @param outputs an NxM tensor containing N vectors of values over buckets,
  -- output by the neural net
  -- @param targets an NxM tensor containing N vectors of actual values over
  -- buckets, produced by @{data_generation_call}
  -- @param mask an NxM tensor containing N mask vectors generated with
  -- @{bucket_conversion.get_possible_bucket_mask}
  -- @return the gradient of @{forward} applied to the arguments
  '''
  def backward(self,outputs, targets, mask):
    dloss_doutput = self.criterion.backward(outputs, targets)
    #--we use the multiplier computed with the mask during forward call
    dloss_doutput.cdiv(self.mask_multiplier.expandAs(dloss_doutput))
    return dloss_doutput