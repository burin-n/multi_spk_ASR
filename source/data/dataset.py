import os,sys,inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0, os.pa) 
import json

import torch
from torch.utils.data import Dataset

import torchaudio
torchaudio.set_audio_backend('sox_io')

from features import Perturbation, WavFeaturizer, TextFeaturizer


class LibriSpeechMixDataset(Dataset):
    """LibriSpeechMix dataset."""

    def __init__(self, jsonl_path, root_dir, transform=[Perturbation(), WavFeaturizer(), TextFeaturizer()]):
        """
        Args:
            jsonl_path (string): Path to the jsonl_path file.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(jsonl_path) as f:
            self.jsonl = [json.loads(line) for line in f.readlines()[:10]]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.jsonl)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        metadata = self.jsonl[idx]
        wav_path = os.path.join(self.root_dir,
                                metadata["mixed_wav"])
        waveform, sample_rate = torchaudio.load(wav_path)
        metadata['audio'] = waveform
        metadata['sample_rate'] = sample_rate
        
        if self.transform:
            if( type(self.transform) == list):
                for trans in self.transform:
                    metadata = trans(metadata)
            else:
                metadata = self.transform(metadata)
        metadata['audio_length'] = metadata['audio'].shape[-1]
        metadata['labels_length'] = [len(lab) for lab in metadata['labels']]
        return metadata



if __name__ == '__main__':
    import time
    jsonl_path = "../dataset/list/train-2mix.jsonl"
    root_dir = "../dataset/train/" 

    path = os.path.join(root_dir, "train-2mix/train-2mix_00000.wav" )

    train_dataset = LibriSpeechMixDataset(jsonl_path, root_dir, transform=[
        Perturbation(), WavFeaturizer(), TextFeaturizer()])
    import time
    start = time.time()
    EP = 1
    N = 1

    start = time.time()
    for j in range(EP):
        for i in range(N):
            fetch = train_dataset[i]
    print('{:.5f} sec/sample'.format((time.time() - start)/EP/N))
    print(fetch)
    print(torch.cat([fetch['audio'], fetch['audio']]).shape)