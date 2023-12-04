from __future__ import absolute_import

import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from . import create
from .attention_recognition_head import AttentionRecognitionHead
from .embedding_head import Embedding, Embedding_self_att
from ..loss.sequenceCrossEntropyLoss import SequenceCrossEntropyLoss
from ..loss.embeddingRegressionLoss import EmbeddingRegressionLoss
from .tps_spatial_transformer import TPSSpatialTransformer
from .stn_head import STNHead

# from config import get_args
# global_args = get_args(sys.argv[1:])


class ModelBuilder(nn.Module):
  """
  This is the integrated model.
  """
  def __init__(self, arch, rec_num_classes, sDim, attDim, max_len_labels, eos, time_step=25, STN_ON=False):
    super(ModelBuilder, self).__init__()

    self.arch = arch
    self.rec_num_classes = rec_num_classes
    self.sDim = sDim
    self.attDim = attDim
    self.max_len_labels = max_len_labels
    self.eos = eos
    self.STN_ON = STN_ON
    self.time_step = time_step
    self.tps_inputsize = (32, 64)# global_args.tps_inputsize

    self.encoder = create(self.arch,
                      with_lstm=True, #global_args.with_lstm
                      n_group=1) #global_args.n_group
    encoder_out_planes = self.encoder.out_planes

    self.decoder = AttentionRecognitionHead(
                      num_classes=rec_num_classes,
                      in_planes=encoder_out_planes,
                      sDim=sDim,
                      attDim=attDim,
                      max_len_labels=max_len_labels)
    self.embeder = Embedding(self.time_step, encoder_out_planes)
    # self.embeder = Embedding_self_att(self.time_step, encoder_out_planes, n_head=4, n_layers=4)
    self.rec_crit = SequenceCrossEntropyLoss()
    self.embed_crit = EmbeddingRegressionLoss(loss_func='cosin')

    if self.STN_ON:
      self.tps = TPSSpatialTransformer(
        output_image_size=tuple((32, 100)), #global_args.tps_outputsize
        num_control_points=20, #global_args.num_control_points
        margins=tuple((0.05, 0.05))) #global_args.tps_margins
      self.stn_head = STNHead(
        in_planes=3,
        num_ctrlpoints=20, #global_args.num_control_points
        activation=None) #global_args.stn_activation

  def forward(self, input_dict):
    return_dict = {}
    return_dict['losses'] = {}
    return_dict['output'] = {}

    if type(input_dict) == torch.Tensor:
      x = input_dict
      # print("x:", x.shape)
      rec_targets = None
      rec_lengths = [25]
      rec_embeds = None
    else:
      x, rec_targets, rec_lengths, rec_embeds = input_dict['images'], \
                                              input_dict['rec_targets'], \
                                              input_dict['rec_lengths'], \
                                              input_dict['rec_embeds']

      # rec_lengths = torch.tensor(rec_lengths).to(x.device)

    # rectification
    if self.STN_ON:
      # input images are downsampled before being fed into stn_head.
      stn_input = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
      stn_img_feat, ctrl_points = self.stn_head(stn_input)
      x, _ = self.tps(x, ctrl_points)
      if not self.training:
        # save for visualization
        return_dict['output']['ctrl_points'] = ctrl_points
        return_dict['output']['rectified_images'] = x

    # print("x:", x.device)

    encoder_feats = self.encoder(x)
    encoder_feats = encoder_feats.contiguous()
    embedding_vectors = self.embeder(encoder_feats)
    if self.training:
      rec_pred = self.decoder([encoder_feats, rec_targets, rec_lengths], embedding_vectors)
      loss_rec = self.rec_crit(rec_pred, rec_targets, rec_lengths)
      loss_embed = self.embed_crit(embedding_vectors, rec_embeds)
      return_dict['losses']['loss_rec'] = loss_rec
      return_dict['losses']['loss_embed'] = loss_embed
    else:
      rec_pred, rec_pred_scores = self.decoder.beam_search(encoder_feats, 5, self.eos, embedding_vectors) #global_args.beam_width
      # rec_pred_ = self.decoder.sample([encoder_feats, rec_targets, rec_lengths]) #, embedding_vectors
      loss_rec = torch.zeros(x.shape[0]) # self.rec_crit(rec_pred_, rec_targets, rec_lengths)
      loss_embed = torch.zeros(x.shape[0]) ##self.embed_crit(embedding_vectors, rec_embeds)
      return_dict['losses']['loss_rec'] = loss_rec
      return_dict['losses']['loss_embed'] = loss_embed
      return_dict['output']['pred_rec'] = rec_pred
      return_dict['output']['pred_embed'] = embedding_vectors
      return_dict['output']['pred_rec_score'] = rec_pred_scores

    # pytorch0.4 bug on gathering scalar(0-dim) tensors
    for k, v in return_dict['losses'].items():
      return_dict['losses'][k] = v.unsqueeze(0)

    return return_dict