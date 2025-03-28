import os, sys
import torch
import torchvision
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
import math
import scene_experimental.diff_operators as diff_operators


class NetworksA(nn.Module):
    def __init__(self, 
                 in_features=3, 
                 out_features=4,
                 hidden_features=256, 
                 num_hidden_layers=3, 
                 type='relu',
                 mode='mlp',
                 fft_mode="simple", 
                 **kwargs):
        super().__init__()
        self.mode = mode
        self.type = type
        self.in_features = in_features
        self.out_features = out_features

        if self.mode == "fft":
            self.positional_encoding = PosEncodingFFT(
                                        in_features=self.in_features, 
                                        mode=fft_mode)
            self.in_features = self.positional_encoding.out_dim

        self.body = FCBlock(in_features=self.in_features, 
                           out_features=hidden_features,
                           num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features, 
                           outermost_linear=False, 
                           nonlinearity=type)
        self.linear_mean = nn.Linear(hidden_features, self.out_features)
        self.linear_std = nn.Linear(hidden_features, self.out_features)

        self.body2 = FCBlock(in_features=2, 
                            out_features=hidden_features,
                            num_hidden_layers=2,
                            hidden_features=hidden_features,
                            outermost_linear=False,
                            nonlinearity=type)
        self.linear_mean2 = nn.Linear(hidden_features, 3)
        self.linear_std2 = nn.Linear(hidden_features, 3)

        # self.xyz_act = nn.ReLU()
        self.sigma_act = nn.ReLU()
        self.rgb_act = nn.Sigmoid()
        self.xyz_lowerbound = None
        self.xyz_upperbound = None
    

    def find_boundary(self, points, keep_aspect_ratio=True, extend_factor=0.1):
        if keep_aspect_ratio:
            self.xyz_lowerbound = points.min()
            self.xyz_upperbound = points.max()
        else:
            self.xyz_lowerbound = torch.max(points, dim=0, keepdim=True).values
            self.xyz_upperbound = torch.min(points, dim=0, keepdim=True).values
        self.boundary = self.xyz_upperbound - self.xyz_lowerbound
        self.xyz_lowerbound -= extend_factor * self.boundary
        self.xyz_upperbound += extend_factor * self.boundary


    def forward(self, net_in, attributes=None, noise_method=None, params=None, raw=True):
        if params is None:
            params = OrderedDict(self.named_parameters())
        
        if raw:
            # NOTE: net_in is organized as [xyz, opacity, color]
            # xyz, opacity, color = torch.split(net_in, [3, 1, 3], dim=-1)
            # xyz_normalzied = (xyz - self.xyz_lowerbound) / (self.xyz_upperbound - self.xyz_lowerbound)
            # coords = torch.cat([xyz_normalzied, opacity, color], dim=-1) * 2 - 1
            coords = (net_in - self.xyz_lowerbound) / (self.xyz_upperbound - self.xyz_lowerbound) * 2 - 1
        else:
            coords = net_in

        coords = torch.clamp(coords, -1, 1)

        if self.mode == 'fft':
            coords_01 = (coords + 1) / 2
            coords_enc = self.positional_encoding(coords_01)
        else:
            coords_enc = coords

        pred = self.body(coords_enc)
        pred_mean = self.linear_mean(pred)
        # pred_std = self.linear_std(pred)

        # NOTE: get mean pred
        sigma_mean, rgb_mean = pred_mean[..., :1], pred_mean[..., 1:]
        rgb_mean_pred = self.rgb_act(rgb_mean)
        sigma_mean_pred = self.sigma_act(sigma_mean)
        sigma_mean_pred = 1. - torch.exp(-sigma_mean_pred)

        # NOTE: get std pred
        # sigma_std, rgb_std = pred_std[..., :1], pred_std[..., 1:]
        # rgb_std_pred = self.rgb_act(rgb_std)
        # sigma_std_pred = self.sigma_act(sigma_std)
        # sigma_std_pred = 1. - torch.exp(-sigma_std_pred)

        # NOTE: compute pred
        # TODO: think about the residual method
        # TODO: if take residual method, all mean pred should be -1 to 1
        # sigma_mean_pred = 2 * sigma_mean_pred - 1
        # rgb_mean_pred = 2 * rgb_mean_pred - 1

        # sigma_pred = sigma_mean_pred + sigma_std_pred * torch.randn_like(sigma_mean_pred)
        # rgb_pred = rgb_mean_pred + rgb_std_pred * torch.randn_like(rgb_mean_pred)
        # sigma_pred = torch.clamp(sigma_pred, 0, 1)
        # rgb_pred = torch.clamp(rgb_pred, 0, 1)
        sigma_pred = sigma_mean_pred
        rgb_pred = rgb_mean_pred

        # NOTE: convert to output format
        # sigma_pred = torch.clamp(sigma_pred, -1, 1)
        # rgb_pred = torch.clamp(rgb_pred, -1, 1)

        # NOTE: generate noise with another network
        if attributes is not None:
            net_in2 = torch.cat([sigma_pred, attributes], dim=-1)
            pred2 = self.body2(net_in2)
            pred2_std = self.linear_std2(pred2)

            if noise_method == 'opacity-detach':
                xyz_noise = torch.exp(pred2_std) * torch.randn_like(pred2_std) * attributes.detach().clone()
            elif noise_method == 'opacity':
                xyz_noise = torch.exp(pred2_std) * torch.randn_like(pred2_std) * attributes
            elif noise_method == 'sigma':
                xyz_noise = torch.exp(pred2_std) * torch.randn_like(pred2_std) * sigma_pred
            elif noise_method == 'sigma-detach':
                xyz_noise = torch.exp(pred2_std) * torch.randn_like(pred2_std) * sigma_pred.detach().clone()
            elif noise_method == '1-sigma-detach':
                # xyz_noise = torch.exp(pred2_std) * torch.randn_like(pred2_std) * (1 - sigma_pred.detach().clone())
                xyz_noise = torch.exp(pred2_std) * torch.randn_like(pred2_std) * 1 / (1 + torch.exp(100 * sigma_pred.detach().clone()))
            elif noise_method == '1-opacity-detach':
                # xyz_noise = torch.exp(pred2_std) * torch.randn_like(pred2_std) * (1 - attributes.detach().clone())
                xyz_noise = torch.exp(pred2_std) * torch.randn_like(pred2_std) * 1 / (1 + torch.exp(100 * attributes.detach().clone()))
            elif noise_method == 'opacity-sigma':
                xyz_noise = torch.exp(pred2_std) * torch.randn_like(pred2_std) * torch.abs(attributes - sigma_pred)
            elif noise_method == 'opacity-sigma-detach':
                xyz_noise = torch.exp(pred2_std) * torch.randn_like(pred2_std) * torch.abs(attributes.detach().clone() - sigma_pred.detach().clone())
            elif noise_method == "1-opacity-sigma-detach":
                xyz_noise = torch.exp(pred2_std) * torch.randn_like(pred2_std) * 1 / (1 + torch.exp(100 * torch.abs(attributes.detach().clone() - sigma_pred.detach().clone())))
            else:
                xyz_noise = torch.exp(pred2_std) * torch.randn_like(pred2_std) * torch.abs(attributes[..., :1] - sigma_pred)
        else:
            xyz_noise = None


        return {
            'net_in': coords, 
            'rgb': rgb_pred,
            'sigma': sigma_pred,
            'xyz_noise': xyz_noise
            }

    
    def forward_with_activations(self, model_input):
        '''Returns not only model output, but also intermediate activations.'''
        coords = model_input['coords'].clone().detach().requires_grad_(True)
        activations = self.net.forward_with_activations(coords)
        return {'pred': activations.popitem(), 'activations': activations}



class PosEncodingFFT(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, in_features, sidelength=None, fn_samples=None, use_nyquist=True, mode="simple"):
        super().__init__()
        self.in_features = in_features
        self.mode = mode

        if mode == "simple":
            self.out_dim = in_features * 2
        elif mode == "multiple":
            self.num_frequencies = 3
            self.out_dim = self.num_frequencies * in_features * 2
        elif mode == "gaussian":
            self.out_dim = 64
            torch.manual_seed(8)
            self.B = torch.randn(self.out_dim // 2, in_features).cuda()
            self.B_scale = 1
        else:
            raise NotImplementedError

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward_multiple(self, coords):
        all_freqs = []
        for i in range(self.num_frequencies):
            x_proj = (2. ** i) * torch.pi * coords
            x_freq = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], -1)
            all_freqs.append(x_freq)
        return torch.cat(all_freqs, -1)
    
    def forward_simple(self, coords):
        x_proj = (2. * torch.pi * coords).float()
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], -1)
    
    def forward_gaussians(self, coords):
        x_proj = (2. * torch.pi * coords).float()
        x_proj = x_proj @ (self.B.T * self.B_scale)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], -1)
    
    def forward(self, coords):
        coords = (coords + 1) / 2  # [-1, 1] -> [0, 1]
        if self.mode == "simple":
            coords_ = self.forward_simple(coords)
        elif self.mode == "multiple":
            coords_ = self.forward_multiple(coords)
        elif self.mode == "gaussian":
            coords_ = self.forward_gaussians(coords)
        return coords_



class FCBlock(nn.Module):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=True, nonlinearity='relu', weight_init=None):
        super().__init__()

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init),
                         'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
                         'sigmoid':(nn.Sigmoid(), init_weights_xavier, None),
                         'tanh':(nn.Tanh(), init_weights_xavier, None),
                         'selu':(nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus':(nn.Softplus(), init_weights_normal, None),
                         'elu':(nn.ELU(inplace=True), init_weights_elu, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.append(nn.Sequential(
            BatchLinear(in_features, hidden_features), nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(
                BatchLinear(hidden_features, hidden_features), nl
            ))

        if outermost_linear:
            self.net.append(nn.Sequential(BatchLinear(hidden_features, out_features)))
        else:
            self.net.append(nn.Sequential(
                BatchLinear(hidden_features, out_features), nl
            ))

        self.net = nn.Sequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

    def forward(self, coords, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        # output = self.net(coords, params=get_subdict(params, 'net'))
        output = self.net(coords)
        return output

    def forward_with_activations(self, coords, params=None, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.'''
        if params is None:
            params = OrderedDict(self.named_parameters())

        activations = OrderedDict()

        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            # subdict = get_subdict(params, 'net.%d' % i)
            for j, sublayer in enumerate(layer):
                if isinstance(sublayer, BatchLinear):
                    # x = sublayer(x, params=get_subdict(subdict, '%d' % j))
                    x = sublayer(x)
                else:
                    x = sublayer(x)

                if retain_grad:
                    x.retain_grad()
                activations['_'.join((str(sublayer.__class__), "%d" % i))] = x
        return activations



class BatchLinear(nn.Linear, nn.Module):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output



class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)



# Initialization methods
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    # grab from upstream pytorch branch and paste here for now
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def init_weights_trunc_normal(m):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            fan_in = m.weight.size(1)
            fan_out = m.weight.size(0)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            mean = 0.
            # initialize with the same behavior as tf.truncated_normal
            # "The generated values follow a normal distribution with specified mean and
            # standard deviation, except that values whose magnitude is more than 2
            # standard deviations from the mean are dropped and re-picked."
            _no_grad_trunc_normal_(m.weight, mean, std, -2 * std, 2 * std)


def init_weights_normal(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_selu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)
