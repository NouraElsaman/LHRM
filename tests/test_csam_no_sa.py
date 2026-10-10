import torch
from lhrm_baseline.model import LHRMModel, CSAM_L
from transformers import BertModel, BertConfig

def test_csam_l2_no_sa_mathematical_difference():
    # Setup small BERT for testing
    config = BertConfig(hidden_size=64, num_hidden_layers=1, num_attention_heads=1)
    bert = BertModel(config)
    num_labels = 10
    batch_size = 4
    
    # Instantiate models
    model_l2 = LHRMModel(bert, num_labels, variant="L2")
    model_l2_no_sa = LHRMModel(bert, num_labels, variant="L2_NO_SA")
    
    # Share weights
    model_l2_no_sa.csam.W.weight.data = model_l2.csam.W.weight.data.clone()
    model_l2_no_sa.csam.W.bias.data = model_l2.csam.W.bias.data.clone()
    
    doc_ids = torch.randint(0, 100, (batch_size, 32))
    doc_mask = torch.ones(batch_size, 32)
    label_ids = torch.randint(0, 100, (num_labels, 10))
    label_mask = torch.ones(num_labels, 10)
    
    model_l2.eval()
    model_l2_no_sa.eval()
    
    # Forward passes
    logits_l2, diag_l2 = model_l2(doc_ids, doc_mask, label_ids, label_mask)
    logits_no_sa, diag_no_sa = model_l2_no_sa(doc_ids, doc_mask, label_ids, label_mask)
    
    # Verify shapes
    assert logits_no_sa.shape == (batch_size, num_labels)
    assert logits_l2.shape == (batch_size, num_labels)
    
    # Verify alpha and S are strictly identical between both runs (they share the same BERT + docs + labels)
    assert torch.allclose(diag_l2["alpha"], diag_no_sa["alpha"])
    assert torch.allclose(diag_l2["S"], diag_no_sa["S"])
    
    # Verify the final logits differ (because NO_SA removes A_ij * h_Lj)
    assert not torch.allclose(logits_l2, logits_no_sa), "Logits should differ due to the ablation."
    
    # Mathematical verification for NO_SA:
    # h_final_no_sa = alpha_i * h_Li
    h_L = model_l2.csam.W.weight.new_zeros(num_labels, 64)  # Get actual h_L by encoding
    with torch.no_grad():
        h_L = model_l2.bert(label_ids, attention_mask=label_mask).last_hidden_state[:, 0, :]
    
    alpha = diag_no_sa["alpha"]  # [B, k]
    
    # Expected output for NO_SA manually computed:
    alpha_weighted = alpha.unsqueeze(2) * h_L.unsqueeze(0)  # [B, k, H]
    expected_logits = model_l2_no_sa.csam.W(alpha_weighted).squeeze(-1)
    
    assert torch.allclose(logits_no_sa, expected_logits, atol=1e-5), "L2_NO_SA forward pass did not match expected math."
    
    # Check that grads propagate
    model_l2_no_sa.train()
    logits, _ = model_l2_no_sa(doc_ids, doc_mask, label_ids, label_mask)
    loss = logits.sum()
    loss.backward()
    
    assert model_l2_no_sa.csam.W.weight.grad is not None
    assert model_l2_no_sa.bert.encoder.layer[0].attention.self.query.weight.grad is not None

if __name__ == "__main__":
    test_csam_l2_no_sa_mathematical_difference()
    print("CSAM NO_SA tests passed.")
