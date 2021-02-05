import torch
from torch.utils.data import DataLoader
from dataset import LibriSpeechMixDataset
from vocabs import Chars

def _speech_collate_fn(batch, pad_id):
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
        max_audio_len = max(metadata['data_length'], max_audio_len)
        max_labs_len = max(max(metadata['labels_length']), max_labs_len)
        
    data_batch, labs_batch = [], []
    data_len_batch, labs_len_batch = [], []
    for metadata in batch:
        data = metadata['data']
        data_len = metadata['data_length']
        labels = metadata['labels']
        labels_len = metadata['labels_length']

        if data_len < max_audio_len:
            pad = (0, max_audio_len - data_len)
            data = torch.nn.functional.pad(data, pad)
        data_batch.append(data)
        data_len_batch.append(data_len)
        
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
    data_batch = torch.stack(data_batch)
    data_len_batch = torch.IntTensor(data_len_batch)
    labs_batch = torch.stack(labs_batch)
    labs_len_batch = torch.stack(labs_len_batch)

    return data_batch, labs_batch, data_len_batch, labs_len_batch 

if __name__ == '__main__':
    jsonl_path = "../dataset/list/train-2mix.jsonl"
    root_dir = "../dataset/train/" 
    
    dataset = LibriSpeechMixDataset(jsonl_path, root_dir)
    vocabs = Chars()
    dataloader = DataLoader(dataset, batch_size=2,
                        shuffle=True, num_workers=0, 
                        collate_fn=lambda x : _speech_collate_fn(x, vocabs.pad))
    i, a = next(enumerate(dataloader))
    print(a)
    print(a[0].shape, a[1].shape, a[2].shape, a[3].shape)