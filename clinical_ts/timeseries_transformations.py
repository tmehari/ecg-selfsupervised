import torch
import torchvision.transforms
import random
import math
import numpy as np
from scipy.interpolate import interp1d
from .timeseries_utils import RandomCrop

###########################################################
# UTILITIES
###########################################################


def interpolate(data, marker):
    timesteps, channels = data.shape
    data = data.flatten(order="F")
    data[data == marker] = np.interp(np.where(data == marker)[0], np.where(
        data != marker)[0], data[data != marker])
    data = data.reshape(timesteps, channels, order="F")
    return data

def Tinterpolate(data, marker):
    timesteps, channels = data.shape
    data = data.transpose(0, 1).flatten()
    ndata = data.numpy()
    interpolation = torch.from_numpy(np.interp(np.where(ndata == marker)[0], np.where(ndata != marker)[0], ndata[ndata != marker]))
    data[data == marker] = interpolation.type(data.type())
    data = data.reshape(channels, timesteps).T
    return data

def squeeze(arr, center, radius, step):
    squeezed = arr[center-step*radius:center+step*radius+1:step, :].copy()
    arr[center-step*radius:center+step*radius+1, :] = np.inf
    arr[center-radius:center+radius+1, :] = squeezed
    return arr

def Tsqueeze(arr, center, radius, step):
    squeezed = arr[center-step*radius:center+step*radius+1:step, :].clone()
    arr[center-step*radius:center+step*radius+1, :]=float("inf")
    arr[center-radius:center+radius+1, :] = squeezed
    return arr

def refill(arr, center, radius, step):
    left_fill_values = arr[center-radius*step -
                           radius:center-radius*step, :].copy()
    right_fill_values = arr[center+radius*step +
                            1:center+radius*step+radius+1, :].copy()
    arr[center-radius*step-radius:center-radius*step, :] = arr[center +
                                                               radius*step+1:center+radius*step+radius+1, :] = np.inf
    arr[center-radius*step-radius:center-radius:step, :] = left_fill_values
    arr[center+radius+step:center+radius*step +
        radius+step:step, :] = right_fill_values
    return arr

def Trefill(arr, center, radius, step):
    left_fill_values = arr[center-radius*step-radius:center-radius*step, :].clone()
    right_fill_values = arr[center+radius*step+1:center+radius*step+radius+1, :].clone()
    arr[center-radius*step-radius:center-radius*step, :] = arr[center+radius*step+1:center+radius*step+radius+1, :] = float("inf")
    arr[center-radius*step-radius:center-radius:step, :] = left_fill_values
    arr[center+radius+step:center+radius*step+radius+step:step, :] = right_fill_values
    return arr


###########################################################
# Pretraining Transformations
###########################################################


class Transformation:
    def __init__(self, *args, **kwargs):
        self.params = kwargs

    def get_params(self):
        return self.params


class GaussianNoise(Transformation):
    """Add gaussian noise to sample.
    """

    def __init__(self, scale=0.1):
        super(GaussianNoise, self).__init__(scale=scale)
        self.scale = scale

    def __call__(self, sample):
        if self.scale == 0:
            return sample
        else:
            data, label = sample
            # np.random.normal(scale=self.scale,size=data.shape).astype(np.float32)
            data = data + np.reshape(np.array([random.gauss(0, self.scale)
                                               for _ in range(np.prod(data.shape))]), data.shape)
            return data, label

    def __str__(self):
        return "GaussianNoise"

class TGaussianNoise(Transformation):
    """Add gaussian noise to sample.
    """

    def __init__(self, scale=0.01):
        super(TGaussianNoise, self).__init__(scale=scale)
        self.scale = scale

    def __call__(self, sample):
        if self.scale ==0:
            return sample
        else:
            data, label = sample
            data = data + self.scale * torch.randn(data.shape)
            return data, label
        
    def __str__(self):
        return "GaussianNoise"

class RandomResizedCrop(Transformation):
    """ Extract crop at random position and resize it to full size
    """

    def __init__(self, crop_ratio_range=[0.5, 1.0], output_size=250):
        super(RandomResizedCrop, self).__init__(
            crop_ratio_range=crop_ratio_range, output_size=output_size)
        self.crop_ratio_range = crop_ratio_range
        self.output_size = output_size

    def __call__(self, sample):
        data, label = sample
        timesteps, channels = data.shape
        output = np.full((self.output_size, channels), np.inf)
        output_timesteps, channels = output.shape
        crop_ratio = random.uniform(*self.crop_ratio_range)
        data, label = RandomCrop(
            int(crop_ratio*timesteps))(sample)  # apply random crop
        cropped_timesteps = data.shape[0]
        if output_timesteps >= cropped_timesteps:
            indices = np.sort(np.random.choice(
                np.arange(output_timesteps-2)+1, size=cropped_timesteps-2, replace=False))
            indices = np.concatenate(
                [np.array([0]), indices, np.array([output_timesteps-1])])
            # fill output array randomly (but in right order) with values from random crop
            output[indices, :] = data

            # use interpolation to resize random crop
            output = interpolate(output, np.inf)
        else:
            indices = np.sort(np.random.choice(
                np.arange(cropped_timesteps), size=output_timesteps, replace=False))
            output = data[indices]
        return output, label

    def __str__(self):
        return "RandomResizedCrop"

class TRandomResizedCrop(Transformation):
    """ Extract crop at random position and resize it to full size
    """
    
    def __init__(self, crop_ratio_range=[0.5, 1.0], output_size=250):
        super(TRandomResizedCrop, self).__init__(
            crop_ratio_range=crop_ratio_range, output_size=output_size)
        self.crop_ratio_range = crop_ratio_range
       
        
    def __call__(self, sample):
        output = torch.full(sample[0].shape, float("inf")).type(sample[0].type())
        timesteps, channels = output.shape
        crop_ratio = random.uniform(*self.crop_ratio_range)
        data, label = TRandomCrop(int(crop_ratio*timesteps))(sample)  # apply random crop
        cropped_timesteps = data.shape[0]
        indices = torch.sort((torch.randperm(timesteps-2)+1)[:cropped_timesteps-2])[0]
        indices = torch.cat([torch.tensor([0]), indices, torch.tensor([timesteps-1])])
        output[indices, :] = data  # fill output array randomly (but in right order) with values from random crop
        
        # use interpolation to resize random crop
        output = Tinterpolate(output, float("inf"))
        return output, label 
    
    def __str__(self):
        return "RandomResizedCrop"

class TRandomCrop(object):
    """Crop randomly the image in a sample.
    """

    def __init__(self, output_size,annotation=False):
        self.output_size = output_size
        self.annotation = annotation

    def __call__(self, sample):
        data, label = sample

        timesteps, _ = data.shape
        assert(timesteps>=self.output_size)
        if(timesteps==self.output_size):
            start=0
        else:
            start = random.randint(0, timesteps - self.output_size-1) #np.random.randint(0, timesteps - self.output_size)

        data = data[start: start + self.output_size, :]
        
        return data, label
    
    def __str__(self):
        return "RandomCrop"

class OldDynamicTimeWarp(Transformation):
    """Stretch and squeeze signal randomly along time axis"""

    def __init__(self):
        pass

    def __call__(self, sample):
        data, label = sample
        data = data.copy()
        timesteps, channels = data.shape
        warp_indices = np.sort(np.random.choice(timesteps, size=timesteps))
        data = data[warp_indices, :]
        return data, label

    def __str__(self):
        return "OldDynamicTimeWarp"

class DynamicTimeWarp(Transformation):
    """Stretch and squeeze signal randomly along time axis"""

    def __init__(self, warps=3, radius=10, step=2):
        super(DynamicTimeWarp, self).__init__(
            warps=warps, radius=radius, step=step)
        self.warps = warps
        self.radius = radius
        self.step = step
        self.min_center = self.radius*(self.step+1)

    def __call__(self, sample):
        data, label = sample
        data = data.copy()
        timesteps, channels = data.shape
        for _ in range(self.warps):
            center = np.random.randint(
                self.min_center, timesteps-self.min_center-self.step)
            data = squeeze(data, center, self.radius, self.step)
            data = refill(data, center, self.radius, self.step)
            data = interpolate(data, np.inf)
        return data, label

    def __str__(self):
        return "DynamicTimeWarp"

class TDynamicTimeWarp(Transformation):
    """Stretch and squeeze signal randomly along time axis"""
    
    def __init__(self, warps=3, radius=10, step=2):
        super(TDynamicTimeWarp, self).__init__(
            warps=warps, radius=radius, step=step)
        self.warps=warps
        self.radius = radius
        self.step = step
        self.min_center = self.radius*(self.step+1)
    
    
    def __call__(self, sample):
        data, label = sample 
        timesteps, channels = data.shape
        for _ in range(self.warps):
            center = random.randint(self.min_center, timesteps-self.min_center-self.step-1)
            data = Tsqueeze(data, center, self.radius, self.step)
            data = Trefill(data, center, self.radius, self.step)
            data = Tinterpolate(data, float("inf"))
        return data, label
    
    def __str__(self):
        return "DynamicTimeWarp"

class TimeWarp(Transformation):
    """apply random monotoneous transformation (random walk) to the time axis"""

    def __init__(self, epsilon=10, interpolation_kind="linear", annotation=False):
        super(TimeWarp, self).__init__(epsilon=epsilon,
                                       interpolation_kind=interpolation_kind, annotation=annotation)
        self.scale = 1.
        self.loc = 0.
        self.epsilon = epsilon
        self.annotation = annotation
        self.interpolation_kind = interpolation_kind

    def __call__(self, sample):
        data, label = sample
        data = data.copy()
        timesteps, channels = data.shape

        pmf = np.random.normal(loc=self.loc, scale=self.scale, size=timesteps)
        pmf = np.cumsum(pmf)  # random walk
        pmf = pmf - np.min(pmf)+self.epsilon  # make it positive

        cdf = np.cumsum(pmf)  # by definition monotonically increasing
        tnew = (cdf-cdf[0])/(cdf[-1]-cdf[0]) * \
            (len(cdf)-1)  # correct normalization
        told = np.arange(timesteps)

        for c in range(channels):
            f = interp1d(tnew, data[:, c], kind=self.interpolation_kind)
            data[:, c] = f(told)
        if(self.annotation):
            for c in range(label.shape[0]):
                f = interp1d(tnew, label[:, c], kind=self.interpolation_kind)
                label[:, c] = f(told)

        return data, label

    def __str__(self):
        return "TimeWarp"

class ChannelResize(Transformation):
    """Scale amplitude of sample (per channel) by random factor in given magnitude range"""

    def __init__(self, magnitude_range=(0.5, 2)):
        super(ChannelResize, self).__init__(magnitude_range=magnitude_range)
        self.log_magnitude_range = np.log(magnitude_range)

    def __call__(self, sample):
        data, label = sample
        timesteps, channels = data.shape
        resize_factors = np.exp(np.random.uniform(
            *self.log_magnitude_range, size=channels))
        resize_factors_same_shape = np.tile(
            resize_factors, timesteps).reshape(data.shape)
        data = np.multiply(resize_factors_same_shape, data)
        return data, label

    def __str__(self):
        return "ChannelResize"

class TChannelResize(Transformation):
    """Scale amplitude of sample (per channel) by random factor in given magnitude range"""
    
    def __init__(self, magnitude_range=(0.33, 3)):
        super(TChannelResize, self).__init__(magnitude_range=magnitude_range)
        self.log_magnitude_range = torch.log(torch.tensor(magnitude_range))
        
        
    def __call__(self, sample):
        data, label = sample
        timesteps, channels = data.shape
        resize_factors = torch.exp(torch.empty(channels).uniform_(*self.log_magnitude_range))
        resize_factors_same_shape = resize_factors.repeat(timesteps).reshape(data.shape)
        data = resize_factors_same_shape * data
        return data, label
    
    def __str__(self):
        return "ChannelResize"

class Negation(Transformation):
    """Flip signal horizontally"""

    def __init__(self):
        super(Negation, self).__init__()
        pass

    def __call__(self, sample):
        data, label = sample
        return -1*data, label

    def __str__(self):
        return "Negation"

class TNegation(Transformation):
    """Flip signal horizontally"""
    
    def __init__(self):
        super(TNegation, self).__init__()
    
    
    def __call__(self, sample):
        data, label = sample 
        return -1*data, label
    
    def __str__(self):
        return "Negation"

class DownSample(Transformation):
    """Downsample signal"""

    def __init__(self, downsample_ratio=0.2):
        super(DownSample, self).__init__(downsample_ratio=downsample_ratio)
        self.downsample_ratio = 0.5

    def __call__(self, sample):
        data, label = sample
        data = data.copy()
        timesteps, channels = data.shape
        inpt_indices = np.random.choice(np.arange(
            timesteps-2)+1, size=int(self.downsample_ratio*timesteps), replace=False)
        data[inpt_indices, :] = np.inf
        data = interpolate(data, np.inf)
        return data, label

    def __str__(self):
        return "DownSample"

class TDownSample(Transformation):
    """Downsample signal"""
    
    def __init__(self, downsample_ratio=0.8):
        super(TDownSample, self).__init__(downsample_ratio=downsample_ratio)
        self.downsample_ratio = downsample_ratio
    
    
    def __call__(self, sample):
        data, label = sample 
        timesteps, channels = data.shape
        inpt_indices = (torch.randperm(timesteps-2)+1)[:int(1-self.downsample_ratio*timesteps)]
        output = data.clone()
        output[inpt_indices, :] = float("inf")
        output = Tinterpolate(output, float("inf"))
        return output, label 
    
    def __str__(self):
        return "DownSample"

class TimeOut(Transformation):
    """ replace random crop by zeros
    """

    def __init__(self, crop_ratio_range=[0.0, 0.5]):
        super(TimeOut, self).__init__(crop_ratio_range=crop_ratio_range)
        self.crop_ratio_range = crop_ratio_range

    def __call__(self, sample):
        data, label = sample
        data = data.copy()
        timesteps, channels = data.shape
        crop_ratio = random.uniform(*self.crop_ratio_range)
        crop_timesteps = int(crop_ratio*timesteps)
        start_idx = random.randint(0, timesteps - crop_timesteps-1)
        data[start_idx:start_idx+crop_timesteps, :] = 0
        return data, label

class TTimeOut(Transformation):
    """ replace random crop by zeros
    """

    def __init__(self, crop_ratio_range=[0.0, 0.5]):
        super(TTimeOut, self).__init__(crop_ratio_range=crop_ratio_range)
        self.crop_ratio_range = crop_ratio_range

    def __call__(self, sample):
        data, label = sample
        data = data.clone()
        timesteps, channels = data.shape
        crop_ratio = random.uniform(*self.crop_ratio_range)
        crop_timesteps = int(crop_ratio*timesteps)
        start_idx = random.randint(0, timesteps - crop_timesteps-1)
        data[start_idx:start_idx+crop_timesteps, :] = 0
        return data, label

    def __str__(self):
        return "TimeOut"

class TGaussianBlur1d(Transformation):
    def __init__(self):
        super(TGaussianBlur1d, self).__init__()
        self.conv = torch.nn.modules.conv.Conv1d(1,1,5,1,2, bias=False)
        self.conv.weight.data = torch.nn.Parameter(torch.tensor([[[0.1, 0.2, 0.4, 0.2, 0.1]]]))
        self.conv.weight.requires_grad = False
        
    def __call__(self, sample):
        data, label = sample
        transposed = data.T
        transposed = torch.unsqueeze(transposed, 1)
        blurred = self.conv(transposed)
        return blurred.reshape(data.T.shape).T, label
        
    def __str__(self):
        return "GaussianBlur"

class ToTensor(Transformation):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, transpose_data=True, transpose_label=False):
        super(ToTensor, self).__init__(
            transpose_data=transpose_data, transpose_label=transpose_label)
        # swap channel and time axis for direct application of pytorch's convs
        self.transpose_data = transpose_data
        self.transpose_label = transpose_label

    def __call__(self, sample):

        def _to_tensor(data, transpose=False):
            if(isinstance(data, np.ndarray)):
                if(transpose):  # seq,[x,y,]ch
                    return torch.from_numpy(np.moveaxis(data, -1, 0))
                else:
                    return torch.from_numpy(data)
            else:  # default_collate will take care of it
                return data

        data, label = sample

        if not isinstance(data, tuple):
            data = _to_tensor(data, self.transpose_data)
        else:
            data = tuple(_to_tensor(x, self.transpose_data) for x in data)

        if not isinstance(label, tuple):
            label = _to_tensor(label, self.transpose_label)
        else:
            label = tuple(_to_tensor(x, self.transpose_label) for x in label)

        return data, label  # returning as a tuple (potentially of lists)

    def __str__(self):
        return "ToTensor"

class TNormalize(Transformation):
    """Normalize using given stats.
    """
    def __init__(self, stats_mean=None, stats_std=None, input=True, channels=[]):
        super(TNormalize, self).__init__(
            stats_mean=stats_mean, stats_std=stats_std, input=input, channels=channels)
        self.stats_mean = torch.tensor([-0.00184586, -0.00130277,  0.00017031, -0.00091313, -0.00148835,  -0.00174687, -0.00077071, -0.00207407,  0.00054329,  0.00155546,  -0.00114379, -0.00035649])
        self.stats_std = torch.tensor([0.16401004, 0.1647168 , 0.23374124, 0.33767231, 0.33362807,  0.30583013, 0.2731171 , 0.27554379, 0.17128962, 0.14030828,   0.14606956, 0.14656108])
        self.stats_mean = self.stats_mean if stats_mean is None else stats_mean
        self.stats_std = self.stats_std if stats_std is None else stats_std
        self.input = input
        if(len(channels)>0):
            for i in range(len(stats_mean)):
                if(not(i in channels)):
                    self.stats_mean[:,i]=0
                    self.stats_std[:,i]=1

    def __call__(self, sample):
        datax, labelx = sample
        data = datax if self.input else labelx
        #assuming channel last
        if(self.stats_mean is not None):
            data = data - self.stats_mean
        if(self.stats_std is not None):
            data = data/self.stats_std

        if(self.input):
            return (data, labelx)
        else:
            return (datax, data)


class Transpose(Transformation):

    def __init__(self):
        super(Transpose, self).__init__()

    def __call__(self, sample):
        data, label = sample 
        data = data.T
        return data, label
    
    def __str__(self):
        return "Transpose"
###########################################################
# ECG Noise Transformations
###########################################################

def signal_power(s):
    return np.mean(s*s)


def snr(s1, s2):
    return 10*np.log10(signal_power(s1)/signal_power(s2))


def baseline_wonder(ss_length=250, fs=100, C=1, K=50, df=0.01):
    """
        Args:
            ss_length: sample size length in steps, default 250
            st_length: sample time legnth in secondes, default 10
            C:         scaling factor of baseline wonder, default 1
            K:         number of sinusoidal functions, default 50
            df:        f_s/ss_length with f_s beeing the sampling frequency, default 0.01
    """
    t = np.tile(np.arange(0, ss_length/fs, 1./fs), K).reshape(K, ss_length)
    k = np.tile(np.arange(K), ss_length).reshape(K, ss_length, order="F")
    phase_k = np.random.uniform(0, 2*np.pi, size=K)
    phase_k = np.tile(phase_k, ss_length).reshape(K, ss_length, order="F")
    a_k = np.tile(np.random.uniform(0, 1, size=K),
                  ss_length).reshape(K, ss_length, order="F")
    # a_k /= a_k[:, 0].sum() # normalize a_k's for convex combination?
    pre_cos = 2*np.pi * k * df * t + phase_k
    cos = np.cos(pre_cos)
    weighted_cos = a_k * cos
    res = weighted_cos.sum(axis=0)
    return C*res


def noise_baseline_wander(fs=100, N=1000, C=1.0, fc=0.5, fdelta=0.01, channels=1, independent_channels=False):
    '''baseline wander as in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5361052/
    fs: sampling frequency (Hz)
    N: lenght of the signal (timesteps)
    C: relative scaling factor (default scale : 1)
    fc: cutoff frequency for the baseline wander (Hz)
    fdelta: lowest resolvable frequency (defaults to fs/N if None is passed)
    channels: number of output channels
    independent_channels: different channels with genuinely different outputs (but all components in phase) instead of just a global channel-wise rescaling
    '''
    if(fdelta is None):  # 0.1
        fdelta = fs/N

    t = np.arange(0, N/fs, 1./fs)
    K = int(np.round(fc/fdelta))

    signal = np.zeros((N, channels))
    for k in range(1, K+1):
        phik = random.uniform(0, 2*math.pi)
        ak = random.uniform(0, 1)
        for c in range(channels):
            if(independent_channels and c > 0):  # different amplitude but same phase
                ak = random.uniform(0, 1)*(2*random.randint(0, 1)-1)
            signal[:, c] += C*ak*np.cos(2*math.pi*k*fdelta*t+phik)

    if(not(independent_channels) and channels > 1):  # just rescale channels by global factor
        channel_gains = np.array(
            [(2*random.randint(0, 1)-1)*random.gauss(1, 1) for _ in range(channels)])
        signal = signal*channel_gains[None]
    return signal

def Tnoise_baseline_wander(fs=100, N=1000, C=1.0, fc=0.5, fdelta=0.01,channels=1,independent_channels=False):
    '''baseline wander as in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5361052/
    fs: sampling frequency (Hz)
    N: lenght of the signal (timesteps)
    C: relative scaling factor (default scale : 1)
    fc: cutoff frequency for the baseline wander (Hz)
    fdelta: lowest resolvable frequency (defaults to fs/N if None is passed)
    channels: number of output channels
    independent_channels: different channels with genuinely different outputs (but all components in phase) instead of just a global channel-wise rescaling
    '''
    if(fdelta is None):# 0.1
        fdelta = fs/N

    K = int((fc/fdelta)+0.5)
    t = torch.arange(0, N/fs, 1./fs).repeat(K).reshape(K, N)
    k = torch.arange(K).repeat(N).reshape(N, K).T
    phase_k = torch.empty(K).uniform_(0, 2*math.pi).repeat(N).reshape(N, K).T
    a_k = torch.empty(K).uniform_(0, 1).repeat(N).reshape(N, K).T
    pre_cos = 2*math.pi * k * fdelta * t + phase_k
    cos = torch.cos(pre_cos)
    weighted_cos = a_k * cos
    res = weighted_cos.sum(dim=0)
    return C*res
            
#     if(not(independent_channels) and channels>1):#just rescale channels by global factor
#         channel_gains = np.array([(2*random.randint(0,1)-1)*random.gauss(1,1) for _ in range(channels)])
#         signal = signal*channel_gains[None]
#     return signal

def noise_electromyographic(N=1000, C=1, channels=1):
    '''electromyographic (hf) noise inspired by https://ieeexplore.ieee.org/document/43620
    N: lenght of the signal (timesteps)
    C: relative scaling factor (default scale: 1)
    channels: number of output channels
    '''
    # C *=0.3 #adjust default scale

    signal = []
    for c in range(channels):
        signal.append(np.array([random.gauss(0.0, C) for i in range(N)]))

    return np.stack(signal, axis=1)

def Tnoise_electromyographic(N=1000,C=1, channels=1):
    '''electromyographic (hf) noise inspired by https://ieeexplore.ieee.org/document/43620
    N: lenght of the signal (timesteps)
    C: relative scaling factor (default scale: 1)
    channels: number of output channels
    '''
    #C *=0.3 #adjust default scale

    signal = torch.empty((N, channels)).normal_(0.0, C)
    
    return signal

def noise_powerline(fs=100, N=1000, C=1, fn=50., K=3, channels=1):
    '''powerline noise inspired by https://ieeexplore.ieee.org/document/43620
    fs: sampling frequency (Hz)
    N: lenght of the signal (timesteps)
    C: relative scaling factor (default scale: 1)
    fn: base frequency of powerline noise (Hz)
    K: number of higher harmonics to be considered
    channels: number of output channels (just rescaled by a global channel-dependent factor)
    '''
    # C *= 0.333 #adjust default scale
    t = np.arange(0, N/fs, 1./fs)

    signal = np.zeros(N)
    phi1 = random.uniform(0, 2*math.pi)
    for k in range(1, K+1):
        ak = random.uniform(0, 1)
        signal += C*ak*np.cos(2*math.pi*k*fn*t+phi1)
    signal = C*signal[:, None]
    if(channels > 1):
        channel_gains = np.array([random.uniform(-1, 1)
                                  for _ in range(channels)])
        signal = signal*channel_gains[None]
    return signal

def Tnoise_powerline(fs=100, N=1000,C=1,fn=50.,K=3, channels=1):
    '''powerline noise inspired by https://ieeexplore.ieee.org/document/43620
    fs: sampling frequency (Hz)
    N: lenght of the signal (timesteps)
    C: relative scaling factor (default scale: 1)
    fn: base frequency of powerline noise (Hz)
    K: number of higher harmonics to be considered
    channels: number of output channels (just rescaled by a global channel-dependent factor)
    '''
    #C *= 0.333 #adjust default scale
    t = torch.arange(0,N/fs,1./fs)
    
    signal = torch.zeros(N)
    phi1 = random.uniform(0,2*math.pi)
    for k in range(1,K+1):
        ak = random.uniform(0,1)
        signal += C*ak*torch.cos(2*math.pi*k*fn*t+phi1)
    signal = C*signal[:,None]
    if(channels>1):
        channel_gains = torch.empty(channels).uniform_(-1,1)
        signal = signal*channel_gains[None]
    return signal

def noise_baseline_shift(fs=100, N=1000, C=1.0, mean_segment_length=3, max_segments_per_second=0.3, channels=1):
    '''baseline shifts inspired by https://ieeexplore.ieee.org/document/43620
    fs: sampling frequency (Hz)
    N: lenght of the signal (timesteps)
    C: relative scaling factor (default scale: 1)
    mean_segment_length: mean length of a shifted baseline segment (seconds)
    max_segments_per_second: maximum number of baseline shifts per second (to be multiplied with the length of the signal in seconds)
    '''
    # C *=0.5 #adjust default scale
    signal = np.zeros(N)

    maxsegs = int(np.ceil(max_segments_per_second*N/fs))

    for i in range(random.randint(0, maxsegs)):
        mid = random.randint(0, N-1)
        seglen = random.gauss(mean_segment_length, 0.2*mean_segment_length)
        left = max(0, int(mid-0.5*fs*seglen))
        right = min(N-1, int(mid+0.5*fs*seglen))
        ak = random.uniform(-1, 1)
        signal[left:right+1] = ak
    signal = C*signal[:, None]

    if(channels > 1):
        channel_gains = np.array(
            [(2*random.randint(0, 1)-1)*random.gauss(1, 1) for _ in range(channels)])
        signal = signal*channel_gains[None]
    return signal

def Tnoise_baseline_shift(fs=100, N=1000,C=1.0,mean_segment_length=3,max_segments_per_second=0.3,channels=1):
    '''baseline shifts inspired by https://ieeexplore.ieee.org/document/43620
    fs: sampling frequency (Hz)
    N: lenght of the signal (timesteps)
    C: relative scaling factor (default scale: 1)
    mean_segment_length: mean length of a shifted baseline segment (seconds)
    max_segments_per_second: maximum number of baseline shifts per second (to be multiplied with the length of the signal in seconds)
    '''
    #C *=0.5 #adjust default scale
    signal = torch.zeros(N)
    
    maxsegs = int((max_segments_per_second*N/fs)+0.5)
    
    for i in range(random.randint(0,maxsegs)):
        mid = random.randint(0,N-1)
        seglen = random.gauss(mean_segment_length,0.2*mean_segment_length)
        left = max(0,int(mid-0.5*fs*seglen))
        right = min(N-1,int(mid+0.5*fs*seglen))
        ak = random.uniform(-1,1)
        signal[left:right+1]=ak
    signal = C*signal[:,None]
    
    if(channels>1):
        channel_gains = 2*torch.randint(2, (channels,))-1 * torch.empty(channels).normal_(1, 1)
        signal = signal*channel_gains[None]
    return signal

def baseline_wonder(N=250, fs=100, C=1, fc=0.5, df=0.01):
    """
        Args:
            ss_length: sample size length in steps, default 250
            st_length: sample time legnth in secondes, default 10
            C:         scaling factor of baseline wonder, default 1
            K:         number of sinusoidal functions, default 50
            df:        f_s/ss_length with f_s beeing the sampling frequency, default 0.01
    """
    K = int(np.round(fc/df))
    t = np.tile(np.arange(0,N/fs,1./fs), K).reshape(K, N)
    k = np.tile(np.arange(K), N).reshape(K, N, order="F")
    phase_k = np.random.uniform(0, 2*np.pi, size=K)
    phase_k = np.tile(phase_k, N).reshape(K, N, order="F")
    a_k = np.tile(np.random.uniform(0, 1, size=K), N).reshape(K, N, order="F")
   
    pre_cos = 2*np.pi * k * df * t + phase_k
    cos = np.cos(pre_cos)
    weighted_cos = a_k * cos
    res = weighted_cos.sum(axis=0)
    return C*res


class BaselineWander(Transformation):
    """Adds baseline wander to the sample.
    """

    def __init__(self, fs=100, Cmax=0.3, fc=0.5, fdelta=0.01,independent_channels=False):
        super(BaselineWander, self).__init__(fs=fs, Cmax=Cmax, fc=fc, fdelta=fdelta,independent_channels=independent_channels)

    def __call__(self, sample):
        data, label = sample
        timesteps, channels = data.shape
        C= random.uniform(0,self.params["Cmax"])
        data = data + noise_baseline_wander(fs=self.params["fs"], N=len(data), C=0.05, fc=self.params["fc"], fdelta=self.params["fdelta"],channels=channels,independent_channels=self.params["independent_channels"])
        return data, label

    def __str__(self):
        return "BaselineWander"

class TBaselineWander(Transformation):
    """Adds baseline wander to the sample.
    """

    def __init__(self, fs=100, Cmax=0.1, fc=0.5, fdelta=0.01,independent_channels=False):
        super(TBaselineWander, self).__init__(fs=fs, Cmax=Cmax, fc=fc, fdelta=fdelta,independent_channels=independent_channels)

    def __call__(self, sample):
        data, label = sample
        timesteps, channels = data.shape
        C= random.uniform(0,self.params["Cmax"])
        noise = Tnoise_baseline_wander(fs=self.params["fs"], N=len(data), C=C, fc=self.params["fc"], fdelta=self.params["fdelta"],channels=channels,independent_channels=self.params["independent_channels"])
        data += noise.repeat(channels).reshape(channels, timesteps).T
        return data, label

    def __str__(self):
        return "BaselineWander"

class PowerlineNoise(Transformation):
    """Adds powerline noise to the sample.
    """

    def __init__(self, fs=100, Cmax=2, K=3):
        super(PowerlineNoise, self).__init__(fs=fs, Cmax=Cmax, K=K)

    def __call__(self, sample):
        data, label = sample
        C = random.uniform(0, self.params["Cmax"])
        data = data + noise_powerline(fs=self.params["fs"], N=len(
            data), C=C, K=self.params["K"], channels=len(data[0]))
        return data, label

    def __str__(self):
        return "PowerlineNoise"

class TPowerlineNoise(Transformation):
    """Adds powerline noise to the sample.
    """

    def __init__(self, fs=100, Cmax=1.0, K=3):
        super(TPowerlineNoise, self).__init__(fs=fs, Cmax=Cmax, K=K)

    def __call__(self, sample):
        data, label = sample
        C= random.uniform(0,self.params["Cmax"])
        data = data + noise_powerline(fs=self.params["fs"], N=len(data), C=C, K=self.params["K"],channels=len(data[0]))
        return data, label

    def __str__(self):
        return "PowerlineNoise"

class EMNoise(Transformation):
    """Adds electromyographic hf noise to the sample.
    """

    def __init__(self, Cmax=0.5, K=3):
        super(EMNoise, self).__init__(Cmax=Cmax, K=K)

    def __call__(self, sample):
        data, label = sample
        C = random.uniform(0, self.params["Cmax"])
        data = data + \
            noise_electromyographic(N=len(data), C=C, channels=len(data[0]))
        return data, label

    def __str__(self):
        return "EMNoise"

class TEMNoise(Transformation):
    """Adds electromyographic hf noise to the sample.
    """

    def __init__(self, Cmax=0.1, K=3):
        super(TEMNoise, self).__init__(Cmax=Cmax, K=K)

    def __call__(self, sample):
        data, label = sample  
        C= random.uniform(0,self.params["Cmax"])
        data = data + Tnoise_electromyographic(N=len(data), C=C, channels=len(data[0]))
        return data, label

    def __str__(self):
        return "EMNoise"

class BaselineShift(Transformation):
    """Adds abrupt baseline shifts to the sample.
    """

    def __init__(self, fs=100, Cmax=3, mean_segment_length=3, max_segments_per_second=0.3):
        super(BaselineShift, self).__init__(fs=fs, Cmax=Cmax,
                                            mean_segment_length=mean_segment_length, max_segments_per_second=max_segments_per_second)

    def __call__(self, sample):
        data, label = sample
        C = random.uniform(0, self.params["Cmax"])
        data = data + noise_baseline_shift(fs=self.params["fs"], N=len(data), C=C, mean_segment_length=self.params["mean_segment_length"],
                                           max_segments_per_second=self.params["max_segments_per_second"], channels=len(data[0]))
        return data, label

    def __str__(self):
        return "BaselineShift"

class TBaselineShift(Transformation):
    """Adds abrupt baseline shifts to the sample.
    """
    def __init__(self, fs=100, Cmax=1.0, mean_segment_length=3,max_segments_per_second=0.3):
        super(TBaselineShift, self).__init__(fs=fs, Cmax=Cmax, mean_segment_length=mean_segment_length, max_segments_per_second=max_segments_per_second)

    def __call__(self, sample):
        data, label = sample     
        C= random.uniform(0,self.params["Cmax"])
        data = data + Tnoise_baseline_shift(fs=self.params["fs"], N=len(data),C=C,mean_segment_length=self.params["mean_segment_length"],max_segments_per_second=self.params["max_segments_per_second"],channels=len(data[0]))
        return data, label

    def __str__(self):
        return "BaselineShift"