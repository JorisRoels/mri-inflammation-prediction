import pytorch_lightning as pl
import os
import csv

import torch.nn.functional as F

from neuralnets.networks.blocks import *
from neuralnets.util.augmentation import *
from neuralnets.util.io import mkdir
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score

from util.constants import *
from util.losses import SPARCCSimilarityLoss
from util.tools import scores, save, mae


class SPARCC_Prediction_Module(nn.Module):
    """
    Transforms a set of I and II feature vectors into a SPARCC score
    """

    def __init__(self, f_dim=512, f_hidden=128, n_classes=1):
        super().__init__()

        p = 0.50
        self.n_classes = n_classes

        self.i_module = nn.Sequential(
            nn.Linear(N_SLICES * N_SIDES * N_QUARTILES * f_dim, f_hidden),
            nn.Dropout(p=p),
            nn.ReLU()
        )
        self.ii_module = nn.Sequential(
            nn.Linear(N_SLICES * N_SIDES * f_dim, f_hidden),
            nn.Dropout(p=p),
            nn.ReLU()
        )
        if n_classes == 1:
            self.merge_module = nn.Sequential(
                nn.Linear(2 * f_hidden, n_classes),
                nn.Sigmoid()
            )
        else:
            self.merge_module = nn.Sequential(
                # nn.Dropout(p=p),
                nn.Linear(2 * f_hidden, n_classes)
            )

    def forward(self, f_i, f_ii):
        """
        Perform forward propagation of the module

        :param f_i: the inflammation feature vector that originates from the I feature extractor
                    should have shape [B, N_SLICES, N_SIDES, N_QUARTILES, F_DIM]
        :param f_ii: the intense inflammation feature vector that originates from the II feature extractor
                     should have shape [B, N_SLICES, N_SIDES, F_DIM]
        :return: output of the module y, i.e. the sparcc score [B]
        """

        b = f_i.size(0)

        # reshape to appropriate siz
        f_i = f_i.view(b, -1)
        f_ii = f_ii.view(b, -1)

        # propagate first module to reduce dimensionality
        f_i = self.i_module(f_i)
        f_ii = self.ii_module(f_ii)

        # concatenate processed features and compute sparcc score
        y = self.merge_module(torch.cat((f_i, f_ii), dim=1))
        if self.n_classes == 1:
            y = y.view(f_i.size(0))

        return y


class SPARCC_Linear_Module(nn.Module):
    """
    Transforms a set of I and II feature vectors into a SPARCC score
    """

    def __init__(self, f_dim=512):
        super().__init__()

        self.i_module = nn.Linear(N_SLICES * N_SIDES * f_dim, f_dim)
        self.ii_module = nn.Linear(N_SLICES * N_SIDES * f_dim, f_dim)
        self.merge_module = nn.Linear(2 * f_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, f_im, f_ii):
        """
        Perform forward propagation of the module

        :param f_im: the merged inflammation feature vector that originates from the merging module
                     should have shape [B, N_SLICES, N_SIDES, F_DIM]
        :param f_ii: the intense inflammation feature vector that originates from the II feature extractor
                     should have shape [B, N_SLICES, N_SIDES, F_DIM]
        :return: output of the module y, i.e. the sparcc score [B]
        """

        # reshape to appropriate size
        f_im = torch.cat([f_im[:, :, i, :] for i in range(N_SIDES)], dim=-1)
        f_im = torch.cat([f_im[:, i, :] for i in range(N_SLICES)], dim=-1)
        f_ii = torch.cat([f_ii[:, :, i, :] for i in range(N_SIDES)], dim=-1)
        f_ii = torch.cat([f_ii[:, i, :] for i in range(N_SLICES)], dim=-1)
        # f_im = f_im.view(f_im.size(0), -1)
        # f_ii = f_ii.view(f_ii.size(0), -1)

        # propagate first module to reduce dimensionality
        f_i = self.i_module(f_im)
        f_ii = self.ii_module(f_ii)

        # concatenate processed features and compute sparcc score
        y = self.merge_module(torch.cat((f_i, f_ii), dim=1))
        y = self.sigmoid(y).view(f_i.size(0))

        return y


class SPARCC_CNN_Module(nn.Module):
    """
    Main module for inflammation and SPARCC prediction
    """

    def __init__(self, backbone='AlexNet', pretrained=False, lambda_s=1, use_t1_input=True, use_t2_input=True):
        super().__init__()

        self.pretrained = pretrained
        self.backbone = backbone
        if backbone in BACKBONES:
            self.model = BACKBONES[backbone]
        else:
            self.model = BACKBONES['AlexNet']
        self.lambda_s = lambda_s
        self.use_t1_input = use_t1_input
        self.use_t2_input = use_t2_input
        in_channels = 2 if use_t1_input and use_t2_input else 1

        # construct main modules
        self.feature_extractor_i = self._construct_feature_extractor(in_channels=in_channels)
        self.feature_extractor_ii = self._construct_feature_extractor(in_channels=in_channels*N_QUARTILES)
        self.classifier_i = self._construct_classifier()
        self.classifier_ii = self._construct_classifier(double_inputs=True)
        self.merge_q_module = self._construct_merge_q_module()
        self.sparcc_module = self._construct_sparcc_module()

    def _modify_fe_channels(self, model, in_channels):

        if self.backbone in ['AlexNet', 'VGG11', 'VGG16']:
            model[0][0] = nn.Conv2d(in_channels, model[0][0].out_channels, kernel_size=model[0][0].kernel_size,
                                    stride=model[0][0].stride, padding=model[0][0].padding, bias=False)
        elif self.backbone in ['ResNet18', 'ResNet101', 'ResNeXt101']:
            model[0] = nn.Conv2d(in_channels, model[0].out_channels, kernel_size=model[0].kernel_size,
                                 stride=model[0].stride, padding=model[0].padding, bias=False)
        elif self.backbone in ['DenseNet121', 'DenseNet201']:
            model[0][0] = nn.Conv2d(in_channels, model[0][0].out_channels, kernel_size=model[0][0].kernel_size,
                                    stride=model[0][0].stride, padding=model[0][0].padding, bias=False)
        return model

    def _modify_c_channels(self, model, out_features=2, double_inputs=False):

        f = 2 if double_inputs else 1

        if self.backbone in ['AlexNet']:
            model[1] = nn.Linear(f * model[1].in_features, model[1].out_features)
            model[-1] = nn.Linear(model[-1].in_features, out_features)
        elif self.backbone in ['VGG11', 'VGG16']:
            model[0] = nn.Linear(f * model[0].in_features, model[0].out_features)
            model[-1] = nn.Linear(model[-1].in_features, out_features)
        elif self.backbone in ['ResNet18', 'ResNet101', 'ResNeXt101', 'DenseNet121', 'DenseNet201']:
            model = nn.Linear(f * model.in_features, out_features)

        return model

    def _construct_feature_extractor(self, in_channels=1):

        model = self.model(pretrained=self.pretrained)

        if self.backbone in ['AlexNet', 'VGG11', 'VGG16']:
            f = nn.Sequential(model.features, model.avgpool)
        elif self.backbone in ['ResNet18', 'ResNet101', 'ResNeXt101']:
            f = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2,
                              model.layer3, model.layer4, model.avgpool)
        elif self.backbone in ['DenseNet121', 'DenseNet201']:
            f = nn.Sequential(model.features, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)))
        else:
            f = None

        f = self._modify_fe_channels(f, in_channels)

        return f

    def _construct_classifier(self, double_inputs=False):

        model = self.model(pretrained=self.pretrained)

        if self.backbone in ['AlexNet', 'VGG11', 'VGG16']:
            c = model.classifier
        elif self.backbone in ['ResNet18', 'ResNet101', 'ResNeXt101']:
            c = model.fc
        elif self.backbone in ['DenseNet121', 'DenseNet201']:
            c = model.classifier
        else:
            c = None

        c = self._modify_c_channels(c, out_features=2, double_inputs=double_inputs)

        return c

    def _construct_merge_q_module(self):

        if self.backbone in ['AlexNet']:
            f_dim = self.classifier_i[1].in_features
        elif self.backbone in ['VGG11', 'VGG16']:
            f_dim = self.classifier_i[0].in_features
        elif self.backbone in ['ResNet18', 'ResNet101', 'ResNeXt101', 'DenseNet121', 'DenseNet201']:
            f_dim = self.classifier_i.in_features
        else:
            f_dim = -1

        mq = nn.Sequential(nn.Linear(N_QUARTILES * f_dim, f_dim), nn.ReLU())

        return mq

    def _construct_sparcc_module(self):

        if self.backbone in ['AlexNet']:
            f_dim = self.classifier_i[1].in_features
        elif self.backbone in ['VGG11', 'VGG16']:
            f_dim = self.classifier_i[0].in_features
        elif self.backbone in ['ResNet18', 'ResNet101', 'ResNeXt101', 'DenseNet121', 'DenseNet201']:
            f_dim = self.classifier_i.in_features
        else:
            f_dim = -1

        sm = SPARCC_Linear_Module(f_dim)

        return sm

    def forward(self, x, mode=JOINT):
        """
        Perform forward propagation of the module

        :param x: the input, format depends on the training mode:
                    - INFLAMMATION_MODULE: [B, CHANNELS, QUARTILE_SIZE, QUARTILE_SIZE]
                    - DEEP_INFLAMMATION_MODULE: [B, CHANNELS, N_QUARTILES, QUARTILE_SIZE, QUARTILE_SIZE]
                    - JOINT or SPARCC_MODULE: [B, CHANNELS, N_SLICES, N_SIDES, N_QUARTILES, QUARTILE_SIZE, QUARTILE_SIZE]
        :param mode: training mode:
                    - INFLAMMATION_MODULE: trains the inflammation classfier alone
                    - DEEP_INFLAMMATION_MODULE: trains both the inflammation and intense inflammation classifier
                    - JOINT: trains all classifiers and additionally returns sparcc predictions
        :return: output of the module (y_i, y_ii, y_s):
                    - y_i: inflammation prediction [B, N_CLASSES, N_SLICES, N_SIDES, N_QUARTILES]
                    - y_ii: intense inflammation prediction [B, N_CLASSES, N_SLICES, N_SIDES]
                            (None, if training mode is INFLAMMATION_MODULE)
                    - y_s: sparcc score prediction [B] (None, if training mode is not JOINT)
        """

        # get shape values
        b = x.size(0)
        channels = x.size(1)
        q = x.size(-1)

        # predict inflammation
        x_sq = x.view(-1, channels, q, q)
        f_i = self.feature_extractor_i(x_sq)
        f_i = torch.flatten(f_i, 1)
        y_i = self.classifier_i(f_i)

        # predict intense inflammation
        y_ii = None
        if mode == DEEP_INFLAMMATION_MODULE or mode == SPARCC_MODULE or mode == JOINT:
            # feature extractor
            x_s = x.view(-1, channels * N_QUARTILES, q, q)
            f_ii = self.feature_extractor_ii(x_s)
            f_ii = torch.flatten(f_ii, 1)

            # merging module (ii)
            f_iv = f_i.view(-1, N_QUARTILES, f_i.size(1))
            f_im = torch.cat([f_iv[:, i, :] for i in range(N_QUARTILES)], dim=-1)
            f_im = self.merge_q_module(f_im)

            # apply classifier
            y_ii = self.classifier_ii(torch.cat((f_ii, f_im), dim=1))

        # reshape to proper format
        if mode == INFLAMMATION_MODULE:
            y_i = y_i.view(b, y_i.size(1))
        elif mode == DEEP_INFLAMMATION_MODULE:
            y_i = y_i.view(b, y_i.size(1), N_QUARTILES)
        else:
            y_i = y_i.view(b, y_i.size(1), N_SLICES, N_SIDES, N_QUARTILES)
        if mode == SPARCC_MODULE or mode == JOINT:
            y_ii = y_ii.view(b, y_ii.size(1), N_SLICES, N_SIDES)

        # compute predicted sparcc score
        y_s = None
        if mode == SPARCC_MODULE or mode == JOINT:
            y_i_ = torch.softmax(y_i, dim=1)[:, 1, ...].view(b, -1)
            y_ii_ = torch.softmax(y_ii, dim=1)[:, 1, ...].view(b, -1)
            y_s = (torch.sum(y_i_, dim=1) + torch.sum(y_ii_, dim=1)) / 72
            # # reshape feature vectors
            # f_im = f_im.view(-1, N_SLICES, N_SIDES, f_im.size(-1))
            # f_ii = f_ii.view(-1, N_SLICES, N_SIDES, f_ii.size(-1))
            # # compute sparcc score
            # y_s = self.sparcc_module(f_im, f_ii)

        # output
        return y_i, y_ii, y_s


class SPARCC_MLP(pl.LightningModule):

    def __init__(self, f_dim=512, f_hidden=128, lr=1e-3, w_sparcc=None, categories=None):
        super().__init__()

        # define model
        if categories is not None:
            self.n_classes = len(categories) + 1
        else:
            self.n_classes = 1

        self.model = SPARCC_Prediction_Module(f_dim=f_dim, f_hidden=f_hidden, n_classes=self.n_classes)

        # define loss function
        # self.loss_sim = SPARCCSimilarityLoss(w_sparcc=w_sparcc)
        # self.loss_sim = nn.MSELoss()
        self.loss_sim = nn.CrossEntropyLoss(weight=torch.Tensor(w_sparcc).to('cuda:0'))
        self.lr = lr

    def forward(self, fi, f_ii):

        return self.model(fi, f_ii)

    def base_step(self, batch, batch_idx, phase='train'):

        # transfer to suitable device and get labels
        f_i, f_ii, y_s = batch
        f_i = f_i.float()
        f_ii = f_ii.float()
        if self.n_classes == 1:
            y_s = y_s.float()
        else:
            y_s = y_s.long()

        # forward prop
        y_s_pred = self(f_i, f_ii)

        # compute loss and log output
        loss = self.loss_sim(y_s_pred, y_s)
        y_s = y_s.cpu().numpy()
        if self.n_classes == 1:
            y_s_pred = y_s_pred.detach().cpu().numpy()
        else:
            y_s_pred = torch.argmax(F.softmax(y_s_pred, dim=1), dim=1).detach().cpu().numpy()
        self.y_true[phase].append(y_s)
        self.y_pred[phase].append(y_s_pred)
        self.running_loss[phase] += loss.detach().cpu().numpy()
        self.running_count[phase] += 1

        return loss

    def training_step(self, batch, batch_idx):
        return self.base_step(batch, batch_idx, phase='train')

    def validation_step(self, batch, batch_idx):
        return self.base_step(batch, batch_idx, phase='val')

    def test_step(self, batch, batch_idx):
        return self.base_step(batch, batch_idx, phase='test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        optimizer_dict = {"optimizer": optimizer}
        # if self.scheduler_name == 'reduce_lr_on_plateau':
        #     scheduler = ReduceLROnPlateau(optimizer, 'max', patience=self.step_size, factor=self.gamma)
        #     optimizer_dict.update({"lr_scheduler": scheduler, "monitor": 'val/mIoU'})
        # elif self.scheduler_name == 'step_lr':
        #     scheduler = StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        #     optimizer_dict.update({"lr_scheduler": scheduler})
        return optimizer_dict

    def on_epoch_start(self):
        self.y_true, self.y_pred, self.running_loss, self.running_count, = {}, {}, {}, {}
        for phase in ['train', 'val', 'test']:
            self.y_true[phase], self.y_pred[phase] = [], []
            self.running_loss[phase], self.running_count[phase] = 0, 0

    def on_epoch_end(self):
        for phase in ['train', 'val', 'test']:
            if len(self.y_true[phase]) > 0:
                    self._log_prediction_metrics(np.concatenate(self.y_true[phase]), np.concatenate(self.y_pred[phase]),
                                                 prefix=phase + '/')
                    self._write_sparcc_results(np.concatenate(self.y_true[phase]), np.concatenate(self.y_pred[phase]),
                                               prefix=phase + '/')
            if self.running_count[phase] > 0:
                self.log(phase + '/loss', self.running_loss[phase] / self.running_count[phase])

    def _write_sparcc_results(self, y_true, y_pred, prefix=''):

        # convert to points
        y_true = y_true * 72
        y_pred = y_pred * 72
        err = np.abs(y_true - y_pred)

        # write CSV file
        log_dir = self.logger.log_dir
        mkdir(os.path.join(log_dir, os.path.dirname(prefix)))
        filename = os.path.join(log_dir, prefix + str(self.global_step) + '.csv')
        with open(filename, mode='w') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['SPARCC scores', ''])
            writer.writerow(['Target', 'Predicted', 'Error'])
            for i in range(len(y_true)):
                writer.writerow([int(y_true[i]), int(y_pred[i]), float(err[i])])

    def _log_prediction_metrics(self, y_true, y_pred, prefix='', suffix=''):

        # flatten everything
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()

        # compute scores
        if self.n_classes == 1:
            m = mae(y_true, y_pred)

            # log scores
            self.log(prefix + 'mae' + suffix, m)

        else:
            acc = accuracy_score(y_true, y_pred)
            ba = balanced_accuracy_score(y_true, y_pred)

            # log scores
            self.log(prefix + 'acc' + suffix, acc)
            self.log(prefix + 'ba' + suffix, ba)


class SPARCC_CNN(pl.LightningModule):

    def __init__(self, backbone='AlexNet', pretrained=False, lambda_s=1, lr=1e-3, w_sparcc=None, use_t1_input=True,
                 use_t2_input=True):
        super().__init__()

        # define model
        self.model = SPARCC_CNN_Module(backbone=backbone, pretrained=pretrained, lambda_s=lambda_s,
                                       use_t1_input=use_t1_input, use_t2_input=use_t2_input)

        # define loss function
        self.loss_ce_i = nn.CrossEntropyLoss(weight=torch.Tensor(INFLAMMATION_WEIGHTS))
        self.loss_ce_ii = nn.CrossEntropyLoss(weight=torch.Tensor(INTENSE_INFLAMMATION_WEIGHTS))
        self.loss_sim = SPARCCSimilarityLoss(w_sparcc=w_sparcc)
        # self.loss_sim = nn.L1Loss()
        self.lr = lr
        self.use_t1_input = use_t1_input
        self.use_t2_input = use_t2_input

        # set training mode
        self.set_training_mode(JOINT)

    def forward(self, x, mode=JOINT):

        return self.model(x, mode=mode)

    def _select_relevant_inputs(self, x):
        if self.use_t1_input and self.use_t2_input:
            return x
        elif self.use_t2_input:
            return x[:, 1:2]
        else:
            return x[:, 0:1]

    def base_step_i(self, batch, batch_idx, phase='train'):

        # transfer to suitable device and get labels
        x, y_i = batch
        x = x.float()
        x = self._select_relevant_inputs(x)
        y_i = y_i.long()

        # forward prop
        y_i_pred, _, _ = self(x, mode=self.training_mode)

        # compute loss and log output
        loss_total = self.loss_ce_i(y_i_pred, y_i)
        y_i_pred = torch.softmax(y_i_pred, dim=1).detach().cpu().numpy()
        y_i = y_i.cpu().numpy()
        self.running_loss[phase]['i'] += loss_total
        self.running_count[phase]['i'] += 1
        self.y_true[phase]['i'].append(y_i)
        self.y_pred[phase]['i'].append(y_i_pred)
        self.running_loss[phase]['total'] += loss_total
        self.running_count[phase]['total'] += 1

        return loss_total

    def base_step_ii(self, batch, batch_idx, phase='train'):

        # transfer to suitable device and get labels
        x, _, y_ii = batch
        x = x.float()
        x = self._select_relevant_inputs(x)
        y_ii = y_ii.long()

        # forward prop
        _, y_ii_pred, _ = self(x, mode=self.training_mode)

        # compute loss and log output
        loss_total = self.loss_ce_ii(y_ii_pred, y_ii)
        y_ii_pred = torch.softmax(y_ii_pred, dim=1).detach().cpu().numpy()
        y_ii = y_ii.cpu().numpy()
        self.running_loss[phase]['ii'] += loss_total
        self.running_count[phase]['ii'] += 1
        self.y_true[phase]['ii'].append(y_ii)
        self.y_pred[phase]['ii'].append(y_ii_pred)
        self.running_loss[phase]['total'] += loss_total
        self.running_count[phase]['total'] += 1

        return loss_total

    def base_step_s(self, batch, batch_idx, phase='train'):

        # transfer to suitable device and get labels
        x, _, _, _, y_s = batch
        x = x.float()
        x = self._select_relevant_inputs(x)
        y_s = y_s.float()

        # forward prop
        _, _, y_s_pred = self(x, mode=self.training_mode)
        # print(y_s_pred)
        # print(y_s)

        # compute loss and log output
        loss_total = self.loss_sim(y_s_pred, y_s)
        y_s_pred = y_s_pred.detach().cpu().numpy()
        y_s = y_s.cpu().numpy()
        self.running_loss[phase]['sim'] += loss_total
        self.running_count[phase]['sim'] += 1
        self.y_true[phase]['sim'].append(y_s)
        self.y_pred[phase]['sim'].append(np.reshape(y_s_pred, y_s.shape))
        self.running_loss[phase]['total'] += loss_total
        self.running_count[phase]['total'] += 1

        return loss_total

    def base_step_joint(self, batch, batch_idx, phase='train'):

        # transfer to suitable device and get labels
        x, y_i, y_ii, y_di, y_s = batch
        x = x.float()
        y_i = y_i.long()
        y_ii = y_ii.long()
        y_s = y_s.float()

        # forward prop
        y_i_pred, y_ii_pred, y_s_pred = self(x, mode=self.training_mode)
        # print(y_s_pred)
        # print(y_s)

        # inflammation loss
        loss_i_ce = N_SIDES * N_QUARTILES * self.loss_ce_i(y_i_pred, y_i)
        loss_total = loss_i_ce
        y_i_pred = torch.softmax(y_i_pred, dim=1).detach().cpu().numpy()
        y_i = y_i.cpu().numpy()
        self.running_loss[phase]['i'] += loss_i_ce
        self.running_count[phase]['i'] += 1
        self.y_true[phase]['i'].append(y_i)
        self.y_pred[phase]['i'].append(y_i_pred)

        # intense inflammation loss
        loss_ii_ce = N_SIDES * self.loss_ce_ii(y_ii_pred, y_ii)
        loss_total = loss_total + loss_ii_ce
        y_ii_pred = torch.softmax(y_ii_pred, dim=1).detach().cpu().numpy()
        y_ii = y_ii.cpu().numpy()
        self.running_loss[phase]['ii'] += loss_ii_ce
        self.running_count[phase]['ii'] += 1
        self.y_true[phase]['ii'].append(y_ii)
        self.y_pred[phase]['ii'].append(y_ii_pred)

        # sparcc similarity loss
        loss_sim = self.loss_sim(y_s_pred, y_s)
        loss_total = loss_total + self.model.lambda_s * loss_sim
        y_s_pred = y_s_pred.detach().cpu().numpy()
        y_s = y_s.cpu().numpy()
        self.running_loss[phase]['sim'] += loss_sim
        self.running_count[phase]['sim'] += 1
        self.y_true[phase]['sim'].append(y_s)
        self.y_pred[phase]['sim'].append(np.reshape(y_s_pred, y_s.shape))
        self.running_loss[phase]['total'] += loss_total
        self.running_count[phase]['total'] += 1

        return loss_total

    def base_step(self, batch, batch_idx, phase='train'):
        if self.training_mode == INFLAMMATION_MODULE:
            return self.base_step_i(batch, batch_idx, phase=phase)
        elif self.training_mode == DEEP_INFLAMMATION_MODULE:
            return self.base_step_ii(batch, batch_idx, phase=phase)
        elif self.training_mode == SPARCC_MODULE:
            return self.base_step_s(batch, batch_idx, phase=phase)
        elif self.training_mode == JOINT:
            return self.base_step_joint(batch, batch_idx, phase=phase)

    def training_step(self, batch, batch_idx):
        return self.base_step(batch, batch_idx, phase='train')

    def validation_step(self, batch, batch_idx):
        return self.base_step(batch, batch_idx, phase='val')

    def test_step(self, batch, batch_idx):
        return self.base_step(batch, batch_idx, phase='test')

    def configure_optimizers(self):
        parameters = {INFLAMMATION_MODULE: list(self.model.feature_extractor_i.parameters()) +
                                           list(self.model.classifier_i.parameters()),
                      DEEP_INFLAMMATION_MODULE: list(self.model.feature_extractor_ii.parameters()) +
                                                list(self.model.classifier_ii.parameters()) +
                                                list(self.model.merge_q_module.parameters()),
                      SPARCC_MODULE: self.model.sparcc_module.parameters(),
                      JOINT: self.model.parameters()}
        lr = {INFLAMMATION_MODULE: self.lr, DEEP_INFLAMMATION_MODULE: self.lr, SPARCC_MODULE: self.lr,
              JOINT: self.lr}
        optimizer = torch.optim.Adam(parameters[self.training_mode], lr=lr[self.training_mode])
        optimizer_dict = {"optimizer": optimizer}
        # if self.scheduler_name == 'reduce_lr_on_plateau':
        #     scheduler = ReduceLROnPlateau(optimizer, 'max', patience=self.step_size, factor=self.gamma)
        #     optimizer_dict.update({"lr_scheduler": scheduler, "monitor": 'val/mIoU'})
        # elif self.scheduler_name == 'step_lr':
        #     scheduler = StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        #     optimizer_dict.update({"lr_scheduler": scheduler})
        return optimizer_dict

    def on_epoch_start(self):
        self.y_true, self.y_pred, self.running_loss, self.running_count, = {}, {}, {}, {}
        for phase in ['train', 'val', 'test']:
            self.y_true[phase], self.y_pred[phase], self.running_loss[phase], self.running_count[phase] = {}, {}, {}, {}
            for case in ['i', 'di', 'ii', 'sim', 'total']:
                self.y_true[phase][case], self.y_pred[phase][case] = [], []
                self.running_loss[phase][case], self.running_count[phase][case] = 0, 0

    def on_epoch_end(self):
        for phase in ['train', 'val', 'test']:
            for case in ['i', 'di', 'ii', 'sim', 'total']:
                if len(self.y_true[phase][case]) > 0:
                    if case == 'sim':
                        self._log_regression_metrics(np.concatenate(self.y_true[phase][case]),
                                                     np.concatenate(self.y_pred[phase][case]),
                                                     prefix=phase + '/' + case + '-')
                        self._write_sparcc_results(np.concatenate(self.y_true[phase][case]),
                                                   np.concatenate(self.y_pred[phase][case]), prefix=phase + '/')
                    else:
                        self._log_accuracy_metrics(np.concatenate(self.y_true[phase][case]),
                                                   np.concatenate(self.y_pred[phase][case]),
                                                   prefix=phase + '/' + case + '-')
                if self.running_count[phase][case] > 0:
                    self.log(phase + '/loss-' + case, self.running_loss[phase][case] / self.running_count[phase][case])

    def set_training_mode(self, mode):
        if mode in [INFLAMMATION_MODULE, DEEP_INFLAMMATION_MODULE, SPARCC_MODULE, JOINT]:
            self.training_mode = mode

    def _write_sparcc_results(self, y_true, y_pred, prefix=''):

        # convert to points
        y_true = y_true * 72
        y_pred = y_pred * 72
        err = np.abs(y_true - y_pred)

        # write CSV file
        log_dir = self.logger.log_dir
        mkdir(os.path.join(log_dir, os.path.dirname(prefix)))
        filename = os.path.join(log_dir, prefix + str(self.global_step) + '.csv')
        with open(filename, mode='w') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['SPARCC scores', ''])
            writer.writerow(['Target', 'Predicted', 'Error'])
            for i in range(len(y_true)):
                writer.writerow([int(y_true[i]), int(y_pred[i]), float(err[i])])

    def _log_accuracy_metrics(self, y_true, y_pred, prefix='', suffix=''):

        # second channel corresponds to classes, we are only interested in class 1
        y_pred = y_pred[:, 1, ...]

        # flatten everything
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()

        # compute scores
        acs, bas, rs, ps, fprs, fs, scores_opt = scores(y_true, y_pred)
        a, ba, r, p, fpr, f = scores_opt
        if np.sum(y_true) > 0:
            auc = roc_auc_score(y_true, y_pred)
        else:
            auc = 0.5

        # log scores
        self.log(prefix + 'accuracy' + suffix, a)
        self.log(prefix + 'balanced-accuracy' + suffix, ba)
        self.log(prefix + 'recall' + suffix, r)
        self.log(prefix + 'precision' + suffix, p)
        self.log(prefix + 'fpr' + suffix, fpr)
        self.log(prefix + 'f1-score' + suffix, f)
        self.log(prefix + 'roc-auc' + suffix, auc)

        # save results
        log_dir = self.logger.log_dir
        results = {'accuracy': acs, 'balanced-accuracy': bas, 'recall': rs, 'precision': ps, 'fpr': fprs,
                   'f1-score': fs, 'roc-auc': auc, 'scores-opt': {'accuracy': a, 'balanced-accuracy': ba, 'recall': r,
                                                                  'precision': p, 'fpr': fpr, 'f1-score': f}}
        mkdir(os.path.join(log_dir, os.path.dirname(prefix)))
        save(results, os.path.join(log_dir, prefix + str(self.global_step) + '.pickle'))

    def _log_regression_metrics(self, y_true, y_pred, prefix='', suffix=''):

        # flatten everything
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()

        # compute scores
        m = mae(y_true, y_pred)

        # log scores
        self.log(prefix + 'mae' + suffix, m)

        # # save results
        # log_dir = self.logger.log_dir
        # results = {'mae': m}
        # mkdir(os.path.join(log_dir, os.path.dirname(prefix)))
        # save(results, os.path.join(log_dir, prefix + str(self.global_step) + '.pickle'))


class Inflammation_Module_Base(nn.Module):
    """
    Base module for inflammation prediction
    """

    def __init__(self, backbone='AlexNet', pretrained=False, use_t1_input=True, use_t2_input=True):
        super().__init__()

        self.pretrained = pretrained
        self.backbone = backbone
        if backbone in BACKBONES:
            self.model = BACKBONES[backbone]
        else:
            self.model = BACKBONES['AlexNet']
        self.use_t1_input = use_t1_input
        self.use_t2_input = use_t2_input
        self.in_channels = 2 if use_t1_input and use_t2_input else 1

    def _modify_fe_channels(self, model, in_channels):

        if self.backbone in ['AlexNet', 'VGG11', 'VGG16']:
            model[0][0] = nn.Conv2d(in_channels, model[0][0].out_channels, kernel_size=model[0][0].kernel_size,
                                    stride=model[0][0].stride, padding=model[0][0].padding, bias=False)
        elif self.backbone in ['ResNet18', 'ResNet101', 'ResNeXt101']:
            model[0] = nn.Conv2d(in_channels, model[0].out_channels, kernel_size=model[0].kernel_size,
                                 stride=model[0].stride, padding=model[0].padding, bias=False)
        elif self.backbone in ['DenseNet121', 'DenseNet201']:
            model[0][0] = nn.Conv2d(in_channels, model[0][0].out_channels, kernel_size=model[0][0].kernel_size,
                                    stride=model[0][0].stride, padding=model[0][0].padding, bias=False)
        return model

    def _modify_c_channels(self, model, out_features=2, double_inputs=False):

        f = 2 if double_inputs else 1

        if self.backbone in ['AlexNet']:
            model[1] = nn.Linear(f * model[1].in_features, model[1].out_features)
            model[-1] = nn.Linear(model[-1].in_features, out_features)
        elif self.backbone in ['VGG11', 'VGG16']:
            model[0] = nn.Linear(f * model[0].in_features, model[0].out_features)
            model[-1] = nn.Linear(model[-1].in_features, out_features)
        elif self.backbone in ['ResNet18', 'ResNet101', 'ResNeXt101', 'DenseNet121', 'DenseNet201']:
            model = nn.Linear(f * model.in_features, out_features)

        return model

    def _construct_feature_extractor(self, in_channels=1):

        model = self.model(pretrained=self.pretrained)

        if self.backbone in ['AlexNet', 'VGG11', 'VGG16']:
            f = nn.Sequential(model.features, model.avgpool)
        elif self.backbone in ['ResNet18', 'ResNet101', 'ResNeXt101']:
            f = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2,
                              model.layer3, model.layer4, model.avgpool)
        elif self.backbone in ['DenseNet121', 'DenseNet201']:
            f = nn.Sequential(model.features, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)))
        else:
            f = None

        f = self._modify_fe_channels(f, in_channels)

        return f

    def _construct_classifier(self, double_inputs=False):

        model = self.model(pretrained=self.pretrained)

        if self.backbone in ['AlexNet', 'VGG11', 'VGG16']:
            c = model.classifier
        elif self.backbone in ['ResNet18', 'ResNet101', 'ResNeXt101']:
            c = model.fc
        elif self.backbone in ['DenseNet121', 'DenseNet201']:
            c = model.classifier
        else:
            c = None

        c = self._modify_c_channels(c, out_features=2, double_inputs=double_inputs)

        return c


class Inflammation_Base(pl.LightningModule):

    def __init__(self, lr=1e-3, use_t1_input=True, use_t2_input=True, weights=None):
        super().__init__()
        self.weights = None if weights is None else torch.Tensor(weights)
        self.loss_ce = nn.CrossEntropyLoss(weight=self.weights)
        self.lr = lr
        self.use_t1_input = use_t1_input
        self.use_t2_input = use_t2_input

    def forward(self, x):
        return self.model(x)

    def _select_relevant_inputs(self, x):
        if self.use_t1_input and self.use_t2_input:
            return x
        elif self.use_t2_input:
            return x[:, 1:2]
        else:
            return x[:, 0:1]

    def base_step(self, batch, batch_idx, phase='train'):
        pass

    def training_step(self, batch, batch_idx):
        return self.base_step(batch, batch_idx, phase='train')

    def validation_step(self, batch, batch_idx):
        return self.base_step(batch, batch_idx, phase='val')

    def test_step(self, batch, batch_idx):
        return self.base_step(batch, batch_idx, phase='test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        optimizer_dict = {"optimizer": optimizer}
        # if self.scheduler_name == 'reduce_lr_on_plateau':
        #     scheduler = ReduceLROnPlateau(optimizer, 'max', patience=self.step_size, factor=self.gamma)
        #     optimizer_dict.update({"lr_scheduler": scheduler, "monitor": 'val/mIoU'})
        # elif self.scheduler_name == 'step_lr':
        #     scheduler = StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        #     optimizer_dict.update({"lr_scheduler": scheduler})
        return optimizer_dict

    def on_epoch_start(self):
        self.y_true, self.y_pred, self.running_loss, self.running_count, = {}, {}, {}, {}
        for phase in ['train', 'val', 'test']:
            self.y_true[phase], self.y_pred[phase] = [], []
            self.running_loss[phase], self.running_count[phase] = 0, 0

    def on_epoch_end(self):
        for phase in ['train', 'val', 'test']:
            if len(self.y_true[phase]) > 0:
                self._log_accuracy_metrics(np.concatenate(self.y_true[phase]), np.concatenate(self.y_pred[phase]),
                                           prefix=phase + '/')
            if self.running_count[phase] > 0:
                self.log(phase + '/loss', self.running_loss[phase] / self.running_count[phase])

    def _log_accuracy_metrics(self, y_true, y_pred, prefix='', suffix=''):

        # second channel corresponds to classes, we are only interested in class 1
        y_pred = y_pred[:, 1, ...]

        # flatten everything
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()

        # compute scores
        acs, bas, rs, ps, fprs, fs, scores_opt = scores(y_true, y_pred)
        a, ba, r, p, fpr, f = scores_opt
        if np.sum(y_true) > 0:
            auc = roc_auc_score(y_true, y_pred)
        else:
            auc = 0.5

        # log scores
        self.log(prefix + 'accuracy' + suffix, a)
        self.log(prefix + 'balanced-accuracy' + suffix, ba)
        self.log(prefix + 'recall' + suffix, r)
        self.log(prefix + 'precision' + suffix, p)
        self.log(prefix + 'fpr' + suffix, fpr)
        self.log(prefix + 'f1-score' + suffix, f)
        self.log(prefix + 'roc-auc' + suffix, auc)

        # save results
        log_dir = self.logger.log_dir
        results = {'accuracy': acs, 'balanced-accuracy': bas, 'recall': rs, 'precision': ps, 'fpr': fprs,
                   'f1-score': fs, 'roc-auc': auc, 'scores-opt': {'accuracy': a, 'balanced-accuracy': ba, 'recall': r,
                                                                  'precision': p, 'fpr': fpr, 'f1-score': f}}
        mkdir(os.path.join(log_dir, os.path.dirname(prefix)))
        save(results, os.path.join(log_dir, prefix + str(self.global_step) + '.pickle'))


class Inflammation_CNN_Module(Inflammation_Module_Base):
    """
    Main module for inflammation prediction
    """

    def __init__(self, backbone='AlexNet', pretrained=False, use_t1_input=True, use_t2_input=True):
        super().__init__(backbone=backbone, pretrained=pretrained, use_t1_input=use_t1_input, use_t2_input=use_t2_input)
        # construct main modules
        self.feature_extractor = self._construct_feature_extractor(in_channels=self.in_channels)
        self.classifier = self._construct_classifier()

    def forward(self, x):
        """
        Perform forward propagation of the module

        :param x: the input, format should be [B, CHANNELS, QUARTILE_SIZE, QUARTILE_SIZE]
        :return: inflammation prediction y_i with shape [B, N_CLASSES]
        """

        # predict inflammation
        f_i = self.feature_extractor(x)
        f_i = torch.flatten(f_i, 1)
        y_i = self.classifier(f_i)

        # output
        return y_i


class Inflammation_CNN(Inflammation_Base):

    def __init__(self, backbone='AlexNet', pretrained=False, lr=1e-3, use_t1_input=True, use_t2_input=True, weights=None):
        super().__init__(lr=lr, use_t1_input=use_t1_input, use_t2_input=use_t2_input, weights=weights)
        # define model
        self.model = Inflammation_CNN_Module(backbone=backbone, pretrained=pretrained, use_t1_input=use_t1_input,
                                             use_t2_input=use_t2_input)

    def base_step(self, batch, batch_idx, phase='train'):

        # transfer to suitable device and get labels
        x, y_i = batch
        x = x.float()
        x = self._select_relevant_inputs(x)
        y_i = y_i.long()

        # forward prop
        y_i_pred = self(x)

        # compute loss and log output
        loss_total = self.loss_ce(y_i_pred, y_i)
        y_i_pred = torch.softmax(y_i_pred, dim=1).detach().cpu().numpy()
        y_i = y_i.cpu().numpy()
        self.running_loss[phase] += loss_total
        self.running_count[phase] += 1
        self.y_true[phase].append(y_i)
        self.y_pred[phase].append(y_i_pred)

        return loss_total


class DeepInflammation_CNN_Module(Inflammation_Module_Base):
    """
    Main module for deep inflammation prediction
    """

    def __init__(self, backbone='AlexNet', pretrained=False, use_t1_input=True, use_t2_input=True,
                 inflammation_model=None):
        super().__init__(backbone=backbone, pretrained=pretrained, use_t1_input=use_t1_input, use_t2_input=use_t2_input)
        self.inflammation_model = inflammation_model
        # construct main modules
        self.feature_extractor = self._construct_feature_extractor(in_channels=self.in_channels*N_QUARTILES)
        self.classifier = self._construct_classifier(double_inputs=self.inflammation_model is not None)
        if self.inflammation_model is not None:
            self.merge_q_module = self._construct_merge_q_module()

    def _construct_merge_q_module(self):

        if self.inflammation_model.backbone in ['AlexNet']:
            f_dim = self.inflammation_model.classifier[1].in_features
        elif self.inflammation_model.backbone in ['VGG11', 'VGG16']:
            f_dim = self.inflammation_model.classifier[0].in_features
        elif self.inflammation_model.backbone in ['ResNet18', 'ResNet101', 'ResNeXt101', 'DenseNet121', 'DenseNet201']:
            f_dim = self.inflammation_model.classifier.in_features
        else:
            f_dim = -1

        mq = nn.Sequential(nn.Linear(N_QUARTILES * f_dim, f_dim), nn.ReLU())

        return mq

    def forward(self, x):
        """
        Perform forward propagation of the module

        :param x: the input, should be of shape [B, CHANNELS, N_QUARTILES, QUARTILE_SIZE, QUARTILE_SIZE]
        :return: intense inflammation prediction y_ii of shape [B, N_CLASSES]
        """

        # get shape values
        channels = x.size(1)
        q = x.size(-1)

        # predict intense inflammation
        # feature extractor
        x_s = x.view(-1, channels * N_QUARTILES, q, q)
        f_ii = self.feature_extractor(x_s)
        f_ii = torch.flatten(f_ii, 1)

        if self.inflammation_model is not None:

            # predict inflammation
            x_sq = x.view(-1, channels, q, q)
            f_i = self.inflammation_model.feature_extractor(x_sq)
            f_i = torch.flatten(f_i, 1)

            # merging module
            f_iv = f_i.view(-1, N_QUARTILES, f_i.size(1))
            f_im = torch.cat([f_iv[:, i, :] for i in range(N_QUARTILES)], dim=-1)
            f_im = self.merge_q_module(f_im)
            # concatenate with deep inflammation features
            f = torch.cat((f_ii, f_im), dim=1)
        else:
            f = f_ii

        # predict deep inflammation
        y_ii = self.classifier(f)

        # output
        return y_ii


class DeepInflammation_CNN(Inflammation_Base):

    def __init__(self, backbone='AlexNet', pretrained=False, lr=1e-3, use_t1_input=True, use_t2_input=True,
                 inflammation_model=None, weights=None):
        super().__init__(lr=lr, use_t1_input=use_t1_input, use_t2_input=use_t2_input, weights=weights)
        # define model
        self.model = DeepInflammation_CNN_Module(backbone=backbone, pretrained=pretrained, use_t1_input=use_t1_input,
                                                 use_t2_input=use_t2_input, inflammation_model=inflammation_model)

    def base_step(self, batch, batch_idx, phase='train'):

        # transfer to suitable device and get labels
        x, _, y_ii = batch
        x = x.float()
        x = self._select_relevant_inputs(x)
        y_ii = y_ii.long()

        # forward prop
        y_ii_pred = self(x)

        # compute loss and log output
        loss_total = self.loss_ce(y_ii_pred, y_ii)
        y_ii_pred = torch.softmax(y_ii_pred, dim=1).detach().cpu().numpy()
        y_ii = y_ii.cpu().numpy()
        self.running_loss[phase] += loss_total
        self.running_count[phase] += 1
        self.y_true[phase].append(y_ii)
        self.y_pred[phase].append(y_ii_pred)

        return loss_total
