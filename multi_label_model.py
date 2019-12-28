import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import BertPreTrainedModel, BertModel
from torch.nn import CrossEntropyLoss
from torch.nn import BCEWithLogitsLoss

__call__ = ['CrossEntropy', 'BCEWithLogLoss']


class CrossEntropy(object):
    '''
    CrossEntropyLoss就是把Softmax–Log–NLLLoss合并成一步
    log-softmax 就是log和softmax合并在一起执行
    NLLLoss 该函数的全程是negative log likelihood loss，函数表达式为
    f(x,class) = -x[class]
    例如假设 x = [1,2,3] class = 2, 那 f(x,class) = -x[2] = -3
    所以为什么pytorch版计算CrossEntropyLoss时 模型计算的logits的shape是m*n ， label 的shape是[m,]
    '''
    def __init__(self):
        self.loss_f = CrossEntropyLoss()

    def __call__(self, output, target):
        loss = self.loss_f(input=output, target=target)
        return loss


class BCEWithLogLoss(object):
    '''
    BCEWithLogitsLoss就是把Sigmoid-BCELoss合成一步
    BCELoss 计算方式是 -(target[0]*math.log(lossinput[0])+(1-target[0])*math.log(1-lossinput[0]))
    '''
    def __init__(self):
        self.weight = torch.Tensor([[1], [1], [1], [1], [1], [1], [1], [1
                                                                        ], [1],
                                    [1], [1], [1], [1], [1], [1], [1], [1],
                                    [1], [1], [1], [1], [1], [1], [1], [1
                                                                        ], [1],
                                    [1], [1], [1], [1], [1], [5], [5], [5]])
        self.loss_fn = BCEWithLogitsLoss(weight=self.weight)

    def __call__(self, output, target):
        output = output.float()
        target = target.float()
        loss = self.loss_fn(input=output, target=target)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.Tensor(torch.ones(class_num, 1))
        else:
            self.alpha = torch.Tensor(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, output, target):
        output = output.float()
        target = target.float()
        output = torch.sigmoid(output)
        y_t = output * target + (1 - output) * (1 - target)
        ce = -torch.log(y_t)
        weight = torch.pow((1 - y_t), self.gamma)

        if output.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        batch_loss = torch.matmul((weight * ce), self.alpha)
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class BertForMultiLable(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForMultiLable, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()
        self.class_num = config.num_labels

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                head_mask=None,
                label_ids=None):
        outputs = self.bert(input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            head_mask=head_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        outputs = (logits, ) + outputs[2:]
        if label_ids is not None:
            # criterion = BCEWithLogLoss()
            criterion = FocalLoss(class_num=self.class_num,
                                  alpha=[[1], [1], [1], [1], [1], [1], [1],
                                         [1], [1], [1], [1], [1], [1], [1],
                                         [1], [1], [1], [1], [1], [1], [1],
                                         [1], [1], [1], [1], [1], [1], [1],
                                         [5], [5], [5], [5], [5], [5]])
            loss = criterion(output=logits, target=label_ids)
            outputs = (loss, ) + outputs
            return outputs
        else:
            return outputs

    def unfreeze(self, start_layer, end_layer):
        def children(m):
            return m if isinstance(m, (list, tuple)) else list(m.children())

        def set_trainable_attr(m, b):
            m.trainable = b
            for p in m.parameters():
                p.requires_grad = b

        def apply_leaf(m, f):
            c = children(m)
            if isinstance(m, nn.Module):
                f(m)
            if len(c) > 0:
                for l in c:
                    apply_leaf(l, f)

        def set_trainable(l, b):
            apply_leaf(l, lambda m: set_trainable_attr(m, b))

        # You can unfreeze the last layer of bert by calling set_trainable(model.bert.encoder.layer[23], True)
        set_trainable(self.bert, False)
        for i in range(start_layer, end_layer + 1):
            set_trainable(self.bert.encoder.layer[i], True)
