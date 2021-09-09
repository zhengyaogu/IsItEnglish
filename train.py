import pytorch_lightning as pl
import torch
from torch import nn
from typing import Any, Dict, List, Union
import torch.nn.functional as F

def count_correct(out, gold):
    preds = out.argmax(dim=-1)
    correct_idx = (preds == gold)
    n_correct = correct_idx.sum().item()
    n_total = correct_idx.shape[0]
    return n_correct, n_total



class BinaryClassification(pl.LightningModule):

    def __init__(self, model: nn.Module,
                 lr: Union[None, float] = None) -> None:
        super(BinaryClassification, self).__init__()

        self.model = model
        self.lr = lr
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        input_ids, gold = batch['input_ids'], batch['gold']
        out = self.model(input_ids)
        loss = F.cross_entropy(out, gold)
        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        input_ids, gold = batch['input_ids'], batch['gold']
        out = self.model(input_ids)
        
        n_correct, n_total = count_correct(out, gold)

        loss = F.cross_entropy(out, gold)
        return {'loss': loss,
                'n_correct': n_correct,
                'n_total': n_total}
    
    def validation_epoch_end(self, step_outputs: List[Any]):
        losses = [d['loss'] for d in step_outputs]
        avg_loss = sum(losses) / len(losses)
        
        acc = (sum([d['n_correct'] for d in step_outputs]) /
               sum([d['n_total'] for d in step_outputs]) )
        self.log('valid_acc', acc, logger=True)
        self.log('valid_avg_loss', avg_loss, logger=True)
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        '''
        By default uses a Adam optimizer for training

        returns: an optimizer for the training
        '''
        if self.lr is None:
            self.lr = 5e-4
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer