import sys
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.autograd import Variable

import eval_metric
sys.path.insert(0, './resources')
import constant
import logging

sigmoid_fn = nn.Sigmoid()


def get_eval_string(true_prediction, val_type_name):
  """
  Given a list of (gold, prediction)s, generate output string.
  """
  count, pred_count, avg_pred_count, p, r, f1 = eval_metric.micro(true_prediction)
  _, _, _, ma_p, ma_r, ma_f1 = eval_metric.macro(true_prediction)
  output_str = "Eval: {0} {1} {2:.3f} P:{3:.3f} R:{4:.3f} F1:{5:.3f} Ma_P:{6:.3f} Ma_R:{7:.3f} Ma_F1:{8:.3f}".format(
    count, pred_count, avg_pred_count, p, r, f1, ma_p, ma_r, ma_f1)
  accuracy = sum([set(y) == set(yp) for y, yp in true_prediction]) * 1.0 / len(true_prediction)
  output_str += '\t {} accuracy: {:.1f}%'.format(val_type_name, accuracy * 100)
  return output_str


def get_figet_evaluation_str(true_prediction):
  _, _, strict_f1 = eval_metric.strict(true_prediction)
  _, _, _, ma_p, ma_r, ma_f1 = eval_metric.macro(true_prediction)
  _, _, _, mi_p, mi_r, mi_f1 = eval_metric.micro(true_prediction)
  res = "%.2f\t%.2f\t%.2f\t" % (strict_f1 * 100, strict_f1 * 100, strict_f1 * 100)
  res += "%.2f\t%.2f\t%.2f\t" % (ma_p * 100, ma_r * 100, ma_f1 * 100)
  res += "%.2f\t%.2f\t%.2f\t" % (mi_p * 100, mi_r * 100, mi_f1 * 100)
  return res


def eval_stratified(true_prediction):
  log = get_logging()
  coarse_true_and_predictions = []
  fine_true_and_predictions = []
  ultrafine_true_and_predictions = []

  for true_labels, predicted_labels in true_prediction:
    coarse_gold, fine_gold, ultrafine_gold = stratify(true_labels)
    coarse_pred, fine_pred, ultrafine_pred = stratify(predicted_labels)
    coarse_true_and_predictions.append((coarse_gold, coarse_pred))
    fine_true_and_predictions.append((fine_gold, fine_pred))
    ultrafine_true_and_predictions.append((ultrafine_gold, ultrafine_pred))

  titles = ["Coarse", "Fine", "Ultrafine"]
  i = 0
  for true_and_predictions in [coarse_true_and_predictions, fine_true_and_predictions, ultrafine_true_and_predictions]:
    result = get_figet_evaluation_str(true_and_predictions)
    log.info(titles[i])
    log.info(result)
    i += 1


def stratify(labels):
  """
  Divide label into three categories.
  """
  id2label = constant.ID2ANS_DICT["open"]
  coarse_ids = range(0, constant.ANSWER_NUM_DICT["gen"])
  fine_ids = range(constant.ANSWER_NUM_DICT["gen"], constant.ANSWER_NUM_DICT["kb"])

  coarse = set([id2label[idx] for idx in coarse_ids])
  fine = set([id2label[idx] for idx in fine_ids])

  return ([l for l in labels if l in coarse],
          [l for l in labels if ((l in fine) and (l not in coarse))],
          [l for l in labels if (l not in coarse) and (l not in fine)])


def get_output_index(outputs):
  """
  Given outputs from the decoder, generate prediction index.
  :param outputs: batch x type_amount
  :return: pred_idx <list>: of len 'batch' with the ids of the predicted types
  """
  pred_idx = []
  outputs = sigmoid_fn(outputs).data.cpu().clone()
  for single_dist in outputs:
    single_dist = single_dist.numpy()
    arg_max_ind = np.argmax(single_dist)
    pred_id = [arg_max_ind]
    pred_id.extend(
      [i for i in range(len(single_dist)) if single_dist[i] > 0.5 and i != arg_max_ind])
    pred_idx.append(pred_id)
  return pred_idx


def get_gold_pred_str(pred_idx, gold, goal):
  """
  Given predicted ids and gold ids, generate a list of (gold, pred) pairs of length batch_size.
  """
  id2word_dict = constant.ID2ANS_DICT[goal]
  gold_strs = []
  for gold_i in gold:
    gold_strs.append([id2word_dict[i] for i in range(len(gold_i)) if gold_i[i] == 1])
  pred_strs = []
  for pred_idx1 in pred_idx:
    pred_strs.append([(id2word_dict[ind]) for ind in pred_idx1])
  return list(zip(gold_strs, pred_strs))


def sort_batch_by_length(tensor: torch.autograd.Variable, sequence_lengths: torch.autograd.Variable):
  """
  @ from allennlp
  Sort a batch first tensor by some specified lengths.

  Parameters
  ----------
  tensor : Variable(torch.FloatTensor), required.
      A batch first Pytorch tensor.
  sequence_lengths : Variable(torch.LongTensor), required.
      A tensor representing the lengths of some dimension of the tensor which
      we want to sort by.

  Returns
  -------
  sorted_tensor : Variable(torch.FloatTensor)
      The original tensor sorted along the batch dimension with respect to sequence_lengths.
  sorted_sequence_lengths : Variable(torch.LongTensor)
      The original sequence_lengths sorted by decreasing size.
  restoration_indices : Variable(torch.LongTensor)
      Indices into the sorted_tensor such that
      ``sorted_tensor.index_select(0, restoration_indices) == original_tensor``
  """

  if not isinstance(tensor, Variable) or not isinstance(sequence_lengths, Variable):
    raise ValueError("Both the tensor and sequence lengths must be torch.autograd.Variables.")

  sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
  sorted_tensor = tensor.index_select(0, permutation_index)
  # This is ugly, but required - we are creating a new variable at runtime, so we
  # must ensure it has the correct CUDA vs non-CUDA type. We do this by cloning and
  # refilling one of the inputs to the function.
  index_range = sequence_lengths.data.clone().copy_(torch.arange(0, len(sequence_lengths)))
  # This is the equivalent of zipping with index, sorting by the original
  # sequence lengths and returning the now sorted indices.
  index_range = Variable(index_range.long())
  _, reverse_mapping = permutation_index.sort(0, descending=False)
  restoration_indices = index_range.index_select(0, reverse_mapping)
  return sorted_tensor, sorted_sequence_lengths, restoration_indices


class MultiSimpleDecoder(nn.Module):
  """
    Simple decoder in multi-task setting.
  """

  def __init__(self, output_dim):
    super(MultiSimpleDecoder, self).__init__()
    self.linear = nn.Linear(output_dim, constant.ANSWER_NUM_DICT['open'],
                            bias=False).cuda()  # (out_features x in_features)

  def forward(self, inputs, output_type):
    if output_type == "open":
      return self.linear(inputs)
    elif output_type == 'wiki':
      return F.linear(inputs, self.linear.weight[:constant.ANSWER_NUM_DICT['wiki'], :], self.linear.bias)
    elif output_type == 'kb':
      return F.linear(inputs, self.linear.weight[:constant.ANSWER_NUM_DICT['kb'], :], self.linear.bias)
    else:
      raise ValueError('Decoder error: output type not one of the valid')


class SimpleDecoder(nn.Module):
  def __init__(self, output_dim, answer_num):
    super(SimpleDecoder, self).__init__()
    self.answer_num = answer_num
    self.linear = nn.Linear(output_dim, answer_num, bias=False)

  def forward(self, inputs, output_type):
    output_embed = self.linear(inputs)
    return output_embed


class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1d = nn.Conv1d(100, 50, 5)  # input, output, filter_number
    self.char_W = nn.Embedding(115, 100)

  def forward(self, span_chars):
    char_embed = self.char_W(span_chars).transpose(1, 2)  # [batch_size, char_embedding, max_char_seq]
    conv_output = [self.conv1d(char_embed)]  # list of [batch_size, filter_dim, max_char_seq, filter_number]
    conv_output = [F.relu(c) for c in conv_output]  # batch_size, filter_dim, max_char_seq, filter_num
    cnn_rep = [F.max_pool1d(i, i.size(2)) for i in conv_output]  # batch_size, filter_dim, 1, filter_num
    cnn_output = torch.squeeze(torch.cat(cnn_rep, 1), 2)  # batch_size, filter_num * filter_dim, 1
    return cnn_output


class SelfAttentiveSum(nn.Module):
  """
  Attention mechanism to get a weighted sum of RNN output sequence to a single RNN output dimension.
  """

  def __init__(self, output_dim, hidden_dim):
    super(SelfAttentiveSum, self).__init__()
    self.key_maker = nn.Linear(output_dim, hidden_dim, bias=False)
    self.key_rel = nn.ReLU()
    self.hidden_dim = hidden_dim
    self.key_output = nn.Linear(hidden_dim, 1, bias=False)
    self.key_softmax = nn.Softmax()

  def forward(self, input_embed):
    input_embed_squeezed = input_embed.view(-1, input_embed.size()[2])
    k_d = self.key_maker(input_embed_squeezed)
    k_d = self.key_rel(k_d)
    if self.hidden_dim == 1:
      k = k_d.view(input_embed.size()[0], -1)
    else:
      k = self.key_output(k_d).view(input_embed.size()[0], -1)  # (batch_size, seq_length)
    weighted_keys = self.key_softmax(k).view(input_embed.size()[0], -1, 1)
    weighted_values = torch.sum(weighted_keys * input_embed, 1)  # batch_size, seq_length, embed_dim
    return weighted_values, weighted_keys


def get_logging(level=logging.DEBUG):
    log = logging.getLogger(__name__)
    if log.handlers:
        return log
    log.setLevel(level)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    ch.setFormatter(formatter)
    log.addHandler(ch)
    return log
