from data import IsItEnglishTrainDataset, Vocab, Tokenize, TokensToIDs, ToTensor, character_tokenize, RidAttributes
from data import Collate
from torchvision.transforms import Compose
from train import BinaryClassification
from model import CNNForCorruptionClassification
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import torch



TRAIN_PATH = 'data/train.json'
VALID_PATH = 'data/valid.json'

def train(bsz: int,
          lr: float
          ) -> None:
    vocab = Vocab.construct_from_json_dataset(TRAIN_PATH)
    transform = Compose([
        Tokenize(tokenize_fn = character_tokenize),
        TokensToIDs(vocab = vocab),
        ToTensor(),
        RidAttributes(['original_tokens', 'corrupted_tokens', 'original_ids', 'corrupted_ids'])
    ])
    train_ds = IsItEnglishTrainDataset.from_json(TRAIN_PATH, transform=transform)
    valid_ds = IsItEnglishTrainDataset.from_json(VALID_PATH, transform=transform)
    train_loader = DataLoader(train_ds, 
                              batch_size=bsz,
                              shuffle=True,
                              collate_fn=Collate(vocab))
    valid_loader = DataLoader(valid_ds,
                              batch_size=bsz,
                              shuffle=True,
                              collate_fn=Collate(vocab))
    model = CNNForCorruptionClassification(vocab = vocab,
                                           embed_dim = 128,
                                           out_channels = 32,
                                           window_sizes = [4, 8, 16])
    pl_module = BinaryClassification(model=model,
                                     lr=lr)
    trainer = pl.Trainer(log_every_n_steps=1,
                         #gpus=1,
                         max_epochs=10,
                         auto_lr_find='lr',
                         callbacks=[
                             ModelCheckpoint(monitor='valid_acc',
                                                 mode='max')
                         ])
    trainer.fit(pl_module,
                 train_dataloaders=train_loader,
                 val_dataloaders=valid_loader)

if __name__ == '__main__':
    train(bsz=32, lr=1e-2)