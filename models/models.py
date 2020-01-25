import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

device = torch.device('cuda')
import random
# discriminator network
def dicriminator():
    discriminator = nn.Sequential(
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    ).to(device)
    return discriminator


# CNN Model
class cnn_fe(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(cnn_fe, self).__init__()
        self.out_dim = out_dim
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(self.input_dim, 8, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1))
        self.fc1 = nn.Linear(8, self.out_dim)  # changed from

    def forward(self, input):
        conv_out = self.encoder(input)
        conv_out = F.dropout(conv_out, p=0.5)  # we didn't need it when source domain is zero condition
        feat = self.fc1(conv_out.view(conv_out.shape[0], -1))
        return feat


class cnn_pred(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(cnn_pred, self).__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)

    def forward(self, feat):
        out = self.fc1(feat)
        out = F.dropout(F.relu(out), p=self.dropout, training=self.training)
        out = self.fc2(out)
        out = F.dropout(F.relu(out), p=self.dropout, training=self.training)
        out = self.fc3(out)
        return out


class CNN(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, device):
        super().__init__()
        self.encoder = cnn_fe(in_dim, out_dim)
        self.dropout = dropout
        self.predictor = cnn_pred(self.encoder.out_dim, self.dropout)
        self.device = device

    def forward(self, src):
        features = self.encoder(src)
        predictions = self.predictor(features)
        return predictions, features


# LSTM Model
class lstm(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, drop, bid, device):
        super().__init__()
        self.encoder = lstm_fe(input_dim, hid_dim, n_layers, drop, bid)
        self.dropout = drop
        self.predictor = lstm_regressor((hid_dim + hid_dim * bid), self.dropout)
        self.device = device

    def param_init_net(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def forward(self, src):
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        features = self.encoder(src)
        predictions = self.predictor(features)
        return predictions, features


class lstm_fe(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, dropout, bidirectional):
        super(lstm_fe, self).__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.bid = bidirectional
        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # input shape [batch_size, seq_length, input_dim]
        # outputs = [batch size, src sent len,  hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        outputs, (hidden, cell) = self.rnn(src)
        outputs = F.dropout(torch.relu(outputs), p=0.5, training=self.training)
        features = outputs[:, -1:].squeeze()

        # outputs are always from the top hidden layer
        return features  # , hidden, cell


class lstm_regressor(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(lstm_regressor, self).__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)

    def forward(self, feat):
        out = self.fc1(feat)
        out = F.dropout(F.relu(out), p=self.dropout, training=self.training)
        out = self.fc2(out)
        out = F.dropout(F.relu(out), p=self.dropout, training=self.training)
        out = self.fc3(out)
        return out


# 1D Variant of ResNet taking in 200 dimensional fixed time series inputs
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        # print('out', out.size(), 'res', residual.size(), self.downsample)
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=1, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, padding=1, stride=stride, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, in_dim, num_classes, arch):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(in_dim, 16, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1])  # , stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2])  # , stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3])  # , stride=2)
        self.avgpool = nn.AvgPool1d(7, stride=1)
        self.fc = nn.Linear(256, num_classes)  # 512 * block.expansion
        self.arch = arch

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # print(x.size())
        predictions = self.fc(x)

        return predictions, x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], arch='resnet18', **kwargs)

    return model


def resnet34(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], arch='resnet34', **kwargs)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], arch='resnet50', **kwargs)

    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], arch='resnet101', **kwargs)

    return model

def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], arch='resnet152', **kwargs)

    return model


# TCN
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
class tcn_regressor(nn.Module):
    def __init__(self, hidden_size, dropout, output_size):
        super(tcn_regressor, self).__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, feat):
        out = self.fc1(feat)
        out = F.dropout(F.relu(out), p=self.dropout, training=self.training)
        out = self.fc2(out)
        out = F.dropout(F.relu(out), p=self.dropout, training=self.training)
        out = self.fc3(out)
        return out


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.encoder = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.predictor = tcn_regressor(num_channels[-1], dropout, output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.encoder(inputs)  # input should have dimension (N, C, L)
        o = self.predictor(y1[:, :, -1])
        # return F.log_softmax(o, dim=1)
        return o, y1

    def initlize_(self, config):
        model = TCN(config['input_channels'], config['n_classes'], config['num_channels'], config['kernel_size'],
                    config['drop']).to(device)
        return model


# VRNN
class VRNN_model(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers, drop, device):
        super().__init__()
        self.encoder = VRNN(x_dim, h_dim, z_dim, n_layers, bias=False)
        self.dropout = drop
        self.predictor = vrnn_regressor((h_dim), self.dropout)
        self.device = device

    # def param_init_net(m):
    #     for name, param in m.named_parameters():
    #         nn.init.uniform_(param.data, -0.08, 0.08)

    def forward(self, src):
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        kld_loss, features = self.encoder(src)
        predictions = self.predictor(features)
        return predictions, features


class VRNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers, bias=False):
        super(VRNN, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        # feature-extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU())

        # encoder
        self.enc = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        # prior
        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        # decoder
        self.dec = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.dec_std = nn.Sequential(
            nn.Linear(h_dim, x_dim),
            nn.Softplus())
        # self.dec_mean = nn.Linear(h_dim, x_dim)
        self.dec_mean = nn.Sequential(
            nn.Linear(h_dim, x_dim),
            nn.Sigmoid())

        # recurrence
        self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers, bias, batch_first=True)

    def forward(self, x):

        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        kld_loss = 0
        nll_loss = 0

        h = Variable(torch.zeros(self.n_layers, x.size(0), self.h_dim)).cuda()
        for t in range(x.size(1)):
            phi_x_t = self.phi_x(x[:, t])
            # encoder
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            # so if we want to use only for supervised learning no need for decoding
            # #decoder
            # dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            # dec_mean_t = self.dec_mean(dec_t)
            # dec_std_t = self.dec_std(dec_t)

            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(1), h)

            # computing losses
            kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
            # nll_loss += self._nll_gauss(dec_mean_t, dec_std_t, x[t])
            # nll_loss += self._nll_bernoulli(dec_mean_t, x[t])

            all_enc_std.append(enc_std_t)
            all_enc_mean.append(enc_mean_t)
            # all_dec_mean.append(dec_mean_t)
            # all_dec_std.append(dec_std_t)

        return kld_loss, phi_z_t
        #    nll_loss, \
        # (all_enc_mean, all_enc_std), \
        # (all_dec_mean, all_dec_std)

    def sample(self, seq_len):

        sample = torch.zeros(seq_len, self.x_dim)

        h = Variable(torch.zeros(self.n_layers, 1, self.h_dim))
        for t in range(seq_len):
            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            # dec_std_t = self.dec_std(dec_t)

            phi_x_t = self.phi_x(dec_mean_t)

            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            sample[t] = dec_mean_t.data

        return sample

    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

    def _init_weights(self, stdv):
        pass

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""

        kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1) +
                       (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
                       std_2.pow(2) - 1)
        return 0.5 * torch.sum(kld_element)

    def _nll_bernoulli(self, theta, x):
        return - torch.sum(x * torch.log(theta) + (1 - x) * torch.log(1 - theta))

    def _nll_gauss(self, mean, std, x):
        pass


class vrnn_regressor(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(vrnn_regressor, self).__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)

    def forward(self, feat):
        out = self.fc1(feat)
        out = F.dropout(F.relu(out), p=self.dropout, training=self.training)
        out = self.fc2(out)
        out = F.dropout(F.relu(out), p=self.dropout, training=self.training)
        out = self.fc3(out)
        return out


# LSTM_autoencoder
class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, dropout, bidirectional):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.bid = bidirectional
        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, dropout=dropout, batch_first=True,
                           bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # input shape [batch_size, seq_length, input_dim]
        # outputs = [batch size, src sent len,  hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        outputs, (hidden, cell) = self.rnn(src)
        outputs = F.dropout(torch.relu(outputs), p=0.5, training=self.training)
        features = outputs[:, -1:].squeeze()

        # outputs are always from the top hidden layer
        return features, hidden, cell
class Decoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers=1, dropout=0.5, bidirectional=False):
        super().__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = input_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.bid = bidirectional
        self.lstm = nn.LSTM(input_dim, hid_dim, n_layers, dropout=0.5, batch_first=True)
        self.out = nn.Linear(hid_dim + hid_dim * self.bid, self.output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, cell):
        # input_shape = [batch size, 1 , input_dim]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]
        x = x.view(x.size(0), 1, self.input_dim)
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        # output = [batch size, seq len, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # the input to the predictor will be

        prediction = self.out(output[:, 0, :])

        # prediction = [batch size, output dim]

        return prediction, hidden, cell

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
class AttnDecoderLSTM(nn.Module):
    # input shape {batch_size, seq_length, input_dim} # batch_first true
    # hidden shape{n_layers, batch_size, hidden_dim}
    # output_shape{batch_size, seq_length, hidden}
    def __init__(self, input_dim, output_dim, hid_dim, n_layers, seq_len, dropout, bidirectional):
        super().__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.bid = bidirectional
        self.seq_len = seq_len

        self.attn = nn.Linear(self.hid_dim + self.hid_dim * bidirectional + self.output_dim, self.seq_len)
        self.attn_combine = nn.Linear(self.hid_dim + self.hid_dim * bidirectional + self.input_dim, self.output_dim)
        self.lstm = nn.LSTM(output_dim, hid_dim, n_layers, dropout=dropout, batch_first=True,
                            bidirectional=bidirectional)
        self.out = nn.Linear(self.hid_dim + self.hid_dim * bidirectional, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, h, c, encoder_outputs):
        batch_size = input.shape[0]
        # h[-1] indicate the hidden state of the last layer as input
        attn_weights = F.softmax(self.attn(torch.cat((input.view(batch_size, -1), h[-1].view(batch_size, -1)), 1)),
                                 dim=1)
        # bmm is batch matrix multiplication
        # If input is a (b \times n \times m)(b×n×m) tensor, mat2 is a (b \times m \times p)(b×m×p) tensor,
        # out will be a (b \times n \times p)(b×n×p) tensor.

        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        output = torch.cat((input.squeeze(), attn_applied.squeeze()), 1)
        output = self.attn_combine(output).unsqueeze(1)
        #         output = F.relu(output)
        #         set_trace()
        output, (h, c) = self.lstm(output.view(output.size(0), 1, self.input_dim), (h, c))
        output = self.out(output[:, 0, :])
        return output, h, c, attn_weights

class seq_seq_reg(nn.Module):

    def __init__(self, hidden_size, dropout):
        super(seq_seq_reg, self).__init__()
        self.dropout = dropout
        # self.fc1 = nn.Linear(hidden_size * 2 + hidden_size * 2 * , hidden_size) # bidierctional
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)

    def forward(self, feat):
        out = self.fc1(feat)
        out = F.dropout(F.relu(out), p=self.dropout, training=self.training)
        out = self.fc2(out)
        out = F.dropout(F.relu(out), p=self.dropout, training=self.training)
        out = self.fc3(out)
        return out

class seq2seq(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, drop, bid,device):
        super().__init__()

        self.encoder = Encoder(input_dim, hid_dim, n_layers, drop, bid)
        self.decoder = Decoder(input_dim, hid_dim)
        self.predictor = seq_seq_reg(hid_dim,drop)
        self.device = device
    def param_init_net(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def forward(self, src, teacher_forcing_ratio=0.5):
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = src.shape[0]
        max_len = src.shape[1]

        # tensor to store decoder outputs
        dec_outputs = torch.zeros(src.size()).to(self.device)
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        encoder_outputs, encoder_hidden, encoder_cell = self.encoder(src)
        decoder_hidden, decoder_cell = encoder_hidden[-1].unsqueeze(0), encoder_cell[-1].unsqueeze(0)
        # first input to the decoder is the <sos> tokens
        decoder_input = torch.zeros(batch_size, 1, src.size(2), device=device)
        for t in range(1, max_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            #             set_trace()
            # normal deocder
            #             output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell)
            # attention_deocder
            decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input,
                                                                        decoder_hidden, decoder_cell)
            # place predictions in a tensor holding predictions for each token
            dec_outputs[:, t, :] = decoder_output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = decoder_output
            # if teacher forcing, use actual next token as next input
            # if not, use predicted input
            decoder_input = src[:, t, :] if teacher_force else top1
        #         set_trace()
        features = encoder_outputs
        predictions = self.predictor(features)
        return predictions, features, dec_outputs
class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, input_dims, hidden_dims, output_dims):
        super(Discriminator, self).__init__()
        #         self.restored = False
        self.layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims),
            nn.LogSoftmax()
        )

    def forward(self, input):
        out = self.layer(input)
        return out
