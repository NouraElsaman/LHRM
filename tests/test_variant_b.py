import torch
import torch.nn as nn
from lhrm_baseline.model import LHRMModel
from lhrm_baseline.losses import BCELoss
from transformers import BertModel, BertConfig

def test_variant_b_shapes_and_gradients():
    config = BertConfig(hidden_size=768, num_hidden_layers=2, num_attention_heads=2)
    bert = BertModel(config)
    num_labels = 54
    batch_size = 8
    
    model = LHRMModel(bert, num_labels, variant="B")
    
    doc_ids = torch.randint(0, 1000, (batch_size, 128))
    doc_mask = torch.ones(batch_size, 128)
    label_ids = torch.randint(0, 1000, (num_labels, 10))
    label_mask = torch.ones(num_labels, 10)
    
    logits, diag = model(doc_ids, doc_mask, label_ids, label_mask)
    
    # Check shape
    assert logits.shape == (batch_size, num_labels)
    assert not diag  # Should be empty dict
    
    # Check finite loss and backward gradients
    targets = torch.randint(0, 2, (batch_size, num_labels)).float()
    criterion = BCELoss()
    
    loss = criterion(logits, targets)
    assert torch.isfinite(loss)
    
    loss.backward()
    
    # Check that classifier has gradients
    assert model.classifier.weight.grad is not None
    assert model.classifier.bias.grad is not None
    
    # Check that BERT has gradients
    assert model.bert.encoder.layer[0].attention.self.query.weight.grad is not None

if __name__ == "__main__":
    test_variant_b_shapes_and_gradients()
    print("Variant B shape and backward tests passed.")
