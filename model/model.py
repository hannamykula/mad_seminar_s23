import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from typing import List
from torch import Tensor
import yaml
from method.fpi import patch_ex_batch

class AutoencoderModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.encoder = WideResNetEncoder(16, 1, 2)
        self.decoder = WideResNetDecoder(16, 1, 2)
        self.loss_fn = nn.BCELoss()
        self.config = config

    def forward(self, x: Tensor):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def detect_anomaly(self, x: Tensor):
        rec = self(x)
        anomaly_map = torch.abs(x - rec)
        anomaly_score = torch.sum(anomaly_map, dim=(1, 2, 3))
        return {
            'reconstruction': rec,
            'anomaly_map': anomaly_map,
            'anomaly_score': anomaly_score
        }

    def training_step(self, batch: Tensor, batch_idx):
        batch_size = batch.size()[0]
        if batch_size > 1:
            split = batch_size//2
            batch1, batch2 = torch.split(batch, split_size_or_sections=split)
            patch1, patch2, label = patch_ex_batch(batch1, batch2)
    
            y1 = self(patch1)
            loss = self.loss_fn(y1, label)
            self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)

            y2 = self(patch2)
            loss = self.loss_fn(y2, label)
            self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)

            return loss

    def validation_step(self, batch: Tensor, batch_idx):
        batch_size = batch.size()[0]
        if batch_size > 1:
            split = batch_size//2
            batch1, batch2 = torch.split(batch, split_size_or_sections=split)
            patch1, patch2, label = patch_ex_batch(batch1, batch2)
    
            y1 = self(patch1)
            loss = self.loss_fn(y1, label)
            self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)

            y2 = self(patch2)
            loss = self.loss_fn(y2, label)
            self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)

            return loss

    def configure_optimizers(self):
        print(self.config)
        return optim.Adam(self.parameters(), lr=self.config['lr'])

class WideResNetBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride: List[int], dropRate=0.0, direction='down'):
        super(WideResNetBlock, self).__init__()
        self.direction = direction
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride[0],
                                padding=1, bias=False)
        if self.direction == 'up':
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride[1],
                                padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        if not self.equalInOut:
            self.convShortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride[0],
                               padding=0, bias=False)
            if self.direction == 'up':
                self.upsampleShortcut = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.convShortcut = None
    
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.conv1(out if self.equalInOut else x)
        if self.direction == 'up':
            out = self.upsample(out)

        out = self.relu2(self.bn2(out))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)

        if not self.equalInOut:
            z = self.convShortcut(x)
        else:
            z = x
        if self.direction == 'up':
            z = self.upsampleShortcut(z)

        return torch.add(z, out)

class GroupBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, direction='down'):
        super(GroupBlock, self).__init__()
        self.direction = direction
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or [1, 1], dropRate, self.direction))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)


class WideResNetEncoder(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNetEncoder, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = WideResNetBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(1, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.block1 = GroupBlock(n, nChannels[0], nChannels[1], block, [1, 1], dropRate, direction='down')
        self.block2 = GroupBlock(n, nChannels[1], nChannels[2], block, [2, 1], dropRate, direction='down')
        self.block3 = GroupBlock(n, nChannels[2], nChannels[3], block, [2, 1], dropRate, direction='down')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        return out

class WideResNetDecoder(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNetDecoder, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        # n = (depth - 4) / 6
        n = 1
        block = WideResNetBlock
        self.conv1 = nn.Conv2d(nChannels[3], nChannels[2], kernel_size=3, stride=1,
                        padding=1, bias=False)
        self.conv2 = nn.Conv2d(nChannels[2], nChannels[1], kernel_size=3, stride=1,
                        padding=1, bias=False)
        self.block1 = GroupBlock(n, nChannels[1], nChannels[0], block, [1, 1], dropRate, direction='up')
        self.block2 = GroupBlock(n, nChannels[0], num_classes, block, [1, 1], dropRate, direction='up')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.block1(out)
        out = self.block2(out)
        out = torch.nn.Sigmoid()(out)
        return out