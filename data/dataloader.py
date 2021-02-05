import torch
from torch.utils.data import DataLoader
from dataset import LibriSpeechMixDataset
from vocabs import Chars


class LibriMixDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        self.pad_id = kwargs.pop('pad_id', None)
        if(self.pad_id == None):
            raise ValueError("pad_id = None")
        kwargs["collate_fn"] = lambda x: self._speech_collate_fn(x, self.pad_id)
        super(LibriMixDataLoader, self).__init__(*args, **kwargs)
    

    def _speech_collate_fn(self, batch, pad_id):
        """collate batch of audio sig, audio len, tokens, tokens len
        Args:
            batch (Optional[FloatTensor], Optional[LongTensor], LongTensor,
                    LongTensor):  A tuple of tuples of signal, signal lengths,
                    encoded tokens, and encoded tokens length.  This collate func
                    assumes the signals are 1d torch tensors (i.e. mono audio).
        """
        max_audio_len = 0
        max_labs_len = 0
        for metadata in batch:
            max_audio_len = max(metadata['audio_length'], max_audio_len)
            max_labs_len = max(max(metadata['labels_length']), max_labs_len)
            
        audio_batch, labs_batch = [], []
        audio_len_batch, labs_len_batch = [], []
        ids = []

        for metadata in batch:
            audio = metadata['audio']
            audio_len = metadata['audio_length']
            labels = metadata['labels']
            labels_len = metadata['labels_length']
            ids.append(metadata['id'])

            if audio_len < max_audio_len:
                pad = (0, max_audio_len - audio_len)
                audio = torch.nn.functional.pad(audio, pad)
            audio_batch.append(audio)
            audio_len_batch.append(audio_len)
            
            lab_padded = []
            for i, lab_i in enumerate(labels):
                lab_i = torch.IntTensor(lab_i)
                lab_i_len = labels_len[i]
                # if lab_len < max_labs_len:
                pad = (0, max_labs_len - lab_i_len)
                lab_i = torch.nn.functional.pad(lab_i, pad, value=pad_id)
                lab_padded.append(lab_i)

            labs_batch.append(torch.stack(lab_padded))
            labs_len_batch.append(torch.IntTensor(labels_len))
        audio_batch = torch.stack(audio_batch)
        audio_len_batch = torch.IntTensor(audio_len_batch)
        labs_batch = torch.stack(labs_batch)
        labs_len_batch = torch.stack(labs_len_batch)

        return audio_batch, labs_batch, audio_len_batch, labs_len_batch, ids


if __name__ == '__main__':
    jsonl_path = "../dataset/list/train-2mix.jsonl"
    root_dir = "../dataset/train/" 
    
    dataset = LibriSpeechMixDataset(jsonl_path, root_dir)
    vocabs = Chars()
    dataloader = LibriMixDataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, pad_id=vocabs.pad)
    _, a = next(enumerate(dataloader))
    print(a)
    print(a[0].shape, a[1].shape, a[2].shape, a[3].shape)