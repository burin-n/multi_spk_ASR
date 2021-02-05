import torch
import torchaudio
torchaudio.set_audio_backend('sox_io')

import random
from vocabs import Chars

class Perturbation(object):
    def __init__(self, speed=[0.9,1.1], vol=[0.9, 1.1], sample_rate = None, rng = None):
        self._min_sp_rate, self._max_sp_rate = speed
        self._min_vol_gain, self._max_vol_gain = vol
        self._sr = sample_rate
        self._rng = random.Random() if rng is None else rng

    def __call__(self, sample):
        sample_rate = self._sr if self._sr != None else sample['sample_rate']
        
        sp = self._rng.uniform(self._min_sp_rate, self._max_sp_rate)
        vol = self._rng.uniform(self._min_vol_gain, self._max_vol_gain)

        self.effects = [
            ['gain', '-n'],  # normalises to 0dB
            ['speed', f'{sp:.5f}'],  
            ['vol', f'{vol:.5f}'],  
            ['rate', f'{sample_rate}'],  # preserve sampling rate after applying speed perturbation
        ]
        sample['data'], sf = torchaudio.sox_effects.apply_effects_tensor(sample['data'], 
                                sample_rate, self.effects, channels_first=True)
        return sample


class WavFeaturizer(object):
    def __init__(self, n_mels=80, win_length = 25, sample_rate = 16000):
        """
            win_length (ms) 
        """
        win_length = int(win_length / 1000 * sample_rate)
        self.featurizer = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, 
            n_fft = 1024, 
            win_length=  400, 
            hop_length = None, 
            f_min = 0.0, 
            f_max = None, 
            pad = 0, 
            n_mels = n_mels, 
            window_fn= torch.hann_window, 
            power = 2.0, 
            normalized = False
        )

    def __call__(self, sample):
        log_mel_specgram = torch.log(self.featurizer(sample['data']))
        sample['data'] = log_mel_specgram[0] # default featurizer returns data in (nbatch, ...)
        return sample



class TextFeaturizer(object):
    def __init__(self, n_mels=80, win_length = 25, sample_rate = 16000):
        self.vocab = Chars()

    def __call__(self, sample):
        labels = []
        for txt in sample['texts']:
            labels.append([self.vocab.bos] + self.vocab.encode(txt) + [self.vocab.eos])
        sample['labels'] = labels
        return sample