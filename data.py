from pandas.core.arrays.sparse import dtype
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
import json
from typing import Any, List, Dict, Callable, Tuple, Union
import torch
from torchvision import transforms
import os
from tqdm import tqdm


def character_tokenize(seq: str) -> List[str]:
    return list(seq)


class IsItEnglishTrainDataset(Dataset):

    def __init__(self, 
                 ds: List,
                 transform: Union[None, Callable] = None) -> None:
        self.ds = ds
        self.transform = transform
    
    def __getitem__(self, i: int) -> Dict[str, Any]:
        row =  {
            'original': self.ds[i][0],
            'corrupted': self.ds[i][1]
        }
        if self.transform:
            row = self.transform(row)
        return row

    def __len__(self) -> int:
        return len(self.ds)
    
    @classmethod
    def from_json(cls, 
                  json_file,
                  transform: Union[None, Callable] = None):
        '''
        json_file: files where the dataset is stored
        kwargs: arguments taken by the constructor
        '''
        with open(json_file, 'r') as f:
            ds = json.load(f)
        return cls(
            ds,
            transform
        )

class Vocab:
    
    def __init__(self, 
                 dict: Dict[str, Any],
                 unk_tok: str = '[UNK]',
                 sep_tok: str = '[SEP]',
                 pad_tok: str = '[PAD]') -> None:
        self.dict = dict
        self.unk_tok = unk_tok
        self.sep_tok = sep_tok
        self.pad_tok = pad_tok

        self._add_unk_tok()
        self._add_sep_tok()
        self._add_alpha_numeric_toks()
        self._add_pad_tok()

        self.itos_lookup = self._construct_itos_lookup()
    
    def _construct_itos_lookup(self) -> List[str]:
        itos_lookup = [None] * len(self.dict)
        for key, value in self.dict.items():
            itos_lookup[value] = key
        return itos_lookup
    
    def _add_alpha_numeric_toks(self):
        alpha_numeric_characters = ([chr(n) for n in range(65, 90)] +      # Capital alphabet
                                    [chr(n) for n in range(97, 122)] +     # lower-case alphabet
                                    [chr(n) for n in range(48, 57)])       # numeric characters
        for c in alpha_numeric_characters:
            if c not in self.dict:
                i = max(len(self.dict)) + 1
                self.dict[c] = i     
    
    def _add_unk_tok(self):
        if self.unk_tok not in self.dict:
            i = max(self.dict.values()) + 1
            self.dict[self.unk_tok] = i
    
    def _add_sep_tok(self):
        if self.sep_tok not in self.dict:
            i = max(self.dict.values()) + 1
            self.dict[self.sep_tok] = i
    
    def _add_pad_tok(self):
        if self.pad_tok not in self.dict:
            i = max(self.dict.values()) + 1
            self.dict[self.pad_tok] = i
    
    def __getitem__(self, key: str) -> int:
        '''
        Take a token in the vocabulary and return its id
        if the token is not in the vocabulary, return the unknown token
        '''
        return self.dict[key]
    
    def itos(self, i: int) -> str:
        return self.itos_lookup[i]
    
    def __len__(self) -> int:
        return len(self.dict)
    
    @staticmethod
    def _construct_from_dataset(ds: List[Tuple[str, str]]):
        counter = 0
        d = dict()
        for i, inst in tqdm(enumerate(ds), total=len(ds)):
            for sent in inst:
                for c in sent:
                    if c not in d:
                        d[c] = counter
                        counter += 1
        return d
    
    @classmethod
    def construct_from_json_dataset(cls, 
                                    ds_file: Union[str, os.PathLike],
                                    unk_tok: str = '[UNK]',
                                    sep_tok: str = '[SEP]',
                                    pad_tok: str = '[PAD]'):
        with open(ds_file, 'r') as f:
            ds = json.load(f)
        vocab_dict = Vocab._construct_from_dataset(ds)
        return cls(
            dict = vocab_dict,
            unk_tok = unk_tok,
            sep_tok = sep_tok,
            pad_tok = pad_tok
        )


class Collate(object):
    
    def __init__(self, vocab: Vocab) -> None:
        self.vocab = vocab

    def __call__(self, insts: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = dict()
        batch['original'] = [inst['original'] for inst in insts]
        batch['corrupted'] = [inst['corrupted'] for inst in insts]
        list_of_input_ids = [inst['input_ids'].transpose(0, 1) for inst in insts] #fed with 2 * L tensors, transpose to L * 2
        batch['input_ids'] = torch.nn.utils.rnn.pad_sequence(list_of_input_ids,
                                                             batch_first=True,
                                                             padding_value=self.vocab[self.vocab.pad_tok]).transpose(1, 2)
        batch['gold'] = torch.Tensor([inst['gold'] for inst in insts]).long()
        return batch
        

class Tokenize(object):

    '''
    Convert the original and corrupted sentences into tokens
    '''

    def __init__(self, tokenize_fn: Callable) -> None:
        '''
        tokenize_fn: a function that converts a string sequence into tokens
        '''
        self.tokenize_fn = tokenize_fn

    def __call__(self, row: Dict[str, Any]) -> Any:
        row['original_tokens'] = self.tokenize_fn(row['original'])
        row['corrupted_tokens'] = self.tokenize_fn(row['corrupted'])
        return row

class TokensToIDs(object):

    def __init__(self,
                 vocab: Any) -> None:
        self.vocab = vocab
    
    def __call__(self, row: Dict[str, Any]) -> Any:
        row['original_ids'] = [self.vocab[tok] for tok in row['original_tokens']]
        row['corrupted_ids'] = [self.vocab[tok] for tok in row['corrupted_tokens']]
        # pad the original sequence to the same length as the corrupted sequence
        pad_target_len = max(len(row['original_ids']), len(row['corrupted_ids']))
        if len(row['original_ids']) < pad_target_len:
            row['original_ids'] = row['original_ids'] + [self.vocab[self.vocab.pad_tok]] * (pad_target_len - len(row['original_ids']))
        else:
            row['corrupted_ids'] = row['corrupted_ids'] + [self.vocab[self.vocab.pad_tok]] * (pad_target_len - len(row['corrupted_ids']))
        return row

class ToTensor(object):

    '''
    Groups the original IDs and corrupted IDs together to form a PyTorch tensor iin a random order.
    Give the instance label based on the order
    '''

    def __call__(self, row: Dict[str, Any]) -> Any:
        coin = random.choice([0, 1])
        if coin == 0:
            row['input_ids'] = torch.Tensor([
                row['original_ids'],
                row['corrupted_ids']
            ]).long()
            row['gold'] = 0
        else:
            row['input_ids'] = torch.Tensor([
                row['corrupted_ids'],
                row['original_ids']
            ]).long()
            row['gold'] = 1
        return row

class RidAttributes(object):

    def __init__(self, attributes: List[str]) -> None:
        '''
        attributes: the attributes to get rid of
        '''
        self.to_rid = attributes

    def __call__(self, row: Dict[str, Any]):
        for a in self.to_rid:
            row.pop(a)
        return row
        