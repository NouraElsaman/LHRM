"""
Paper-faithful LHRM model implementation.

Architecture per Eqs. 1-8:
  h_T = BERT_CLS(X)                 [Eq.1: CLS last_hidden_state[:,0]]
  h_Li = BERT_CLS(Y_i)              [Eq.2: same BERT, per label, with grad]
  S_i = h_T . h_Li                  [Eq.4: dot product]
  alpha_i = softmax(S_i over i)     [Eq.5: attention weights]
  A_ij = softmax(h_Li.h_Lj^T/sqrt(d)) [Eq.6: label self-attention]
  h_final^(i) = alpha_i*h_Li + sum_j A_ij*h_Lj  [Eq.7: per-label refined repr]
  y_hat_i = W.h_final^(i) + b       [Eq.8: shared scalar linear]

Variant ladder (A6 / A5b):
  L        = literal (Eqs.4-8 exactly)
  L2       = literal with scaled dot-product S_i/sqrt(768) in Eq.4
  L2_NO_SA = literal L2 with label-label self-attention excised (ablation)
  L2_A5B   = literal L2 with unshared per-label classifier W in R^{k x H} and shared scalar bias b
  R        = repository-style concat+linear (fallback only, not paper-faithful)
  B        = standard BERT multi-label control [CLS] -> Linear(k)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from transformers import BertModel


def get_cls_embedding(bert_output) -> torch.Tensor:
    """
    Extract [CLS] last hidden state.
    Paper Eq.1/2: BERT_CLS(X) = last_hidden_state[:, 0, :]
    NOT pooler_output (which is a learned tanh projection).
    PAPER-DEFINED (A1).
    """
    return bert_output.last_hidden_state[:, 0, :]  # [B, H]


class CSAM_L(nn.Module):
    """
    Literal CSAM implementation (Variant L).
    Implements Eqs. 3-8 exactly as written.
    
    Decision A6: selected if viable (loss decreasing, no NaN, alpha not one-hot).
    Decision A7: no learned projections in Eq.6.
    Decision A5: W is shared R^{1xH} vector (scalar output per label).
    
    The document influences h_final only through alpha_i (A6 note in decision_registry).
    """
    def __init__(self, hidden: int, scale_similarity: bool = False, **kwargs):
        super().__init__()
        self.hidden = hidden
        self.scale_similarity = scale_similarity  # If True: Variant L2
        self.use_label_sa = kwargs.get("use_label_sa", True)  # If False: L2_NO_SA ablation
        # Eq.8: shared weight W in R^{H} and bias b
        self.W = nn.Linear(hidden, 1, bias=True)  # shared across labels
    
    def forward(
        self,
        h_T: torch.Tensor,    # [B, H] document CLS embedding
        h_L: torch.Tensor,    # [k, H] label CLS embeddings (k labels)
        return_alpha: bool = False,  # overrides eval-mode detection
    ) -> Tuple[torch.Tensor, dict]:
        """
        Returns:
            logits: [B, k] — one score per (document, label)
            diagnostics: dict with alpha stats for saturation check

        F1 fix (Phase 1F): diag["alpha"] is set to the detached alpha tensor
        [B, k] whenever the model is in eval mode (model.training == False)
        or when return_alpha=True is explicitly passed.

        Training path: numerically unchanged. The alpha tensor was always
        computed; it is now exposed in the diagnostics dict for the
        compute_diagnostics() reader. No gradient path change.
        """
        B, H = h_T.shape
        k = h_L.shape[0]

        # Eq.4: S_i = h_T . h_Li  (dot product)
        # h_T: [B, H], h_L: [k, H] -> S: [B, k]
        S = h_T @ h_L.T  # [B, k]

        if self.scale_similarity:
            # Variant L2: scale by sqrt(d) to prevent saturation [ASSUMED A6]
            S = S / math.sqrt(H)

        # Eq.5: alpha_i = softmax(S_i) over labels
        alpha = F.softmax(S, dim=1)  # [B, k]

        # Eq.6: A_ij = softmax(h_Li . h_Lj^T / sqrt(d)) [label self-attention]
        # h_L: [k, H] -> A: [k, k]
        A_raw = (h_L @ h_L.T) / math.sqrt(H)  # [k, k]
        A = F.softmax(A_raw, dim=1)  # [k, k], rows sum to 1

        # Eq.7: h_final^(i) = alpha_i * h_Li + sum_j A_ij * h_Lj
        # alpha: [B, k] -> alpha_i for each doc, A: [k, k]
        # alpha_i * h_Li: need shape [B, k, H]
        alpha_weighted = alpha.unsqueeze(2) * h_L.unsqueeze(0)  # [B, k, H]

        if self.use_label_sa:
            # sum_j A_ij * h_Lj: for each i, sum over j
            # A @ h_L: [k, k] @ [k, H] = [k, H] -> broadcast to [B, k, H]
            A_weighted = (A @ h_L).unsqueeze(0).expand(B, -1, -1)  # [B, k, H]
            h_final = alpha_weighted + A_weighted  # [B, k, H]
        else:
            h_final = alpha_weighted  # Ablation: document-label dependency only

        # Eq.8: y_hat_i = W . h_final^(i) + b [shared W in R^H]
        # Apply shared linear layer to each h_final^(i)
        # h_final: [B, k, H] -> logits: [B, k, 1] -> [B, k]
        logits = self.W(h_final).squeeze(-1)  # [B, k]

        # Diagnostics for softmax-saturation check (A6 viability criteria)
        with torch.no_grad():
            max_alpha = alpha.max(dim=1).values.mean().item()
            alpha_entropy = -(alpha * (alpha + 1e-10).log()).sum(dim=1).mean().item()
            one_hot_frac = (alpha.max(dim=1).values > 0.9).float().mean().item()

        diag = {
            "alpha_max_mean":   max_alpha,
            "alpha_entropy_mean": alpha_entropy,
            "alpha_one_hot_frac": one_hot_frac,
            # F1 fix: expose alpha tensor for compute_diagnostics().
            # Set when in eval mode OR when explicitly requested.
            # Training mode: kept None to avoid holding [B,k] tensors per batch.
            "alpha": (alpha.detach() if (not self.training or return_alpha) else None),
            # Expose S scores for distribution diagnostics
            "S": (S.detach() if (not self.training or return_alpha) else None),
        }

        return logits, diag


class CSAM_L_A5B(nn.Module):
    """
    CSAM Variant L2_A5B (Figure 2 Alternative Architectural Interpretation).
    
    Identical to Variant L2 except for classifier head parameterization:
    - Retains scaled document-label similarity S_i / sqrt(H) (Eq. 4-5).
    - Retains label-label self-attention A_ij (Eq. 6-7).
    - Replaces shared classifier nn.Linear(H, 1) with an unshared per-label
      weight parameter W in R^{num_labels x hidden} and a shared scalar bias b in R.
      
    Logit computation:
      logits[b, i] = dot(W[i], h_final[b, i]) + b
      Implemented as: (h_final * self.W.unsqueeze(0)).sum(dim=-1) + self.bias
      Output shape: [B, num_labels]
      
    CRITICAL: Does NOT use nn.Linear(H, k) which produces [B, k, k].
    """
    def __init__(self, hidden: int, num_labels: int, scale_similarity: bool = True, **kwargs):
        super().__init__()
        self.hidden = hidden
        self.num_labels = num_labels
        self.scale_similarity = scale_similarity
        self.use_label_sa = kwargs.get("use_label_sa", True)
        
        # Unshared per-label weight matrix: [num_labels, hidden]
        self.W = nn.Parameter(torch.empty(num_labels, hidden))
        # Shared scalar bias: [1] (preserves strict single-component isolation)
        self.bias = nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.hidden) if self.hidden > 0 else 0.01
        nn.init.uniform_(self.W, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(
        self,
        h_T: torch.Tensor,    # [B, H] document CLS embedding
        h_L: torch.Tensor,    # [k, H] label CLS embeddings (k labels)
        return_alpha: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        B, H = h_T.shape
        k = h_L.shape[0]
        assert k == self.num_labels, f"Expected {self.num_labels} labels, got {k}"

        # Eq.4: S_i = h_T . h_Li (scaled similarity)
        S = h_T @ h_L.T  # [B, k]
        if self.scale_similarity:
            S = S / math.sqrt(H)

        # Eq.5: alpha_i = softmax(S_i) over labels
        alpha = F.softmax(S, dim=1)  # [B, k]

        # Eq.6: A_ij = softmax(h_Li . h_Lj^T / sqrt(d)) [label self-attention]
        A_raw = (h_L @ h_L.T) / math.sqrt(H)  # [k, k]
        A = F.softmax(A_raw, dim=1)  # [k, k]

        # Eq.7: h_final^(i) = alpha_i * h_Li + sum_j A_ij * h_Lj
        alpha_weighted = alpha.unsqueeze(2) * h_L.unsqueeze(0)  # [B, k, H]
        if self.use_label_sa:
            A_weighted = (A @ h_L).unsqueeze(0).expand(B, -1, -1)  # [B, k, H]
            h_final = alpha_weighted + A_weighted  # [B, k, H]
        else:
            h_final = alpha_weighted

        # A5b Classifier Head: logits[b, i] = dot(W[i], h_final[b, i]) + bias
        # h_final: [B, k, H], self.W: [k, H] -> unsqueeze(0) -> [1, k, H]
        # (h_final * self.W.unsqueeze(0)).sum(dim=-1) -> [B, k]
        logits = (h_final * self.W.unsqueeze(0)).sum(dim=-1) + self.bias  # [B, k]

        # Diagnostics matching CSAM_L
        with torch.no_grad():
            max_alpha = alpha.max(dim=1).values.mean().item()
            alpha_entropy = -(alpha * (alpha + 1e-10).log()).sum(dim=1).mean().item()
            one_hot_frac = (alpha.max(dim=1).values > 0.9).float().mean().item()

        diag = {
            "alpha_max_mean":   max_alpha,
            "alpha_entropy_mean": alpha_entropy,
            "alpha_one_hot_frac": one_hot_frac,
            "alpha": (alpha.detach() if (not self.training or return_alpha) else None),
            "S": (S.detach() if (not self.training or return_alpha) else None),
        }

        return logits, diag


class CSAM_R(nn.Module):
    """
    Repository-style CSAM (Variant R) — FALLBACK ONLY.
    Uses the repository's concat([doc, agg])->Linear(2H, k) head.
    NOT paper-faithful (not Eqs.7-8).
    Used only if L and L2 fail viability criteria.
    REPOSITORY-DEFINED.
    """
    def __init__(self, hidden: int, num_labels: int, use_sa: bool = True, heads: int = 4):
        super().__init__()
        self.hidden = hidden
        self.use_sa = use_sa
        if use_sa:
            self.sa = nn.MultiheadAttention(hidden, heads, batch_first=True)
            self.ln = nn.LayerNorm(hidden)
        self.classifier = nn.Linear(2 * hidden, num_labels)
    
    def forward(
        self,
        h_T: torch.Tensor,   # [B, H]
        h_L: torch.Tensor,   # [k, H]
    ) -> Tuple[torch.Tensor, dict]:
        L = h_L  # [k, H]
        if self.use_sa:
            att, _ = self.sa(L.unsqueeze(0), L.unsqueeze(0), L.unsqueeze(0))
            L = self.ln(L + att.squeeze(0))
        w = (h_T @ L.T).softmax(dim=1)   # [B, k]
        agg = w @ L                       # [B, H]
        emb = torch.cat([h_T, agg], dim=1)  # [B, 2H]
        logits = self.classifier(emb)        # [B, k]
        return logits, {}


class LHRMModel(nn.Module):
    """
    Paper-faithful LHRM model.
    
    - Shared BERT for document and label encoding (A2: shared weights)
    - Document: CLS last_hidden_state[:,0] (A1: not pooler_output)
    - Labels: CLS last_hidden_state[:,0] per label, computed with gradients (A3)
    - CSAM: Eqs.4-8, variant selectable (A6)
    
    Args:
        bert_model:        Loaded BertModel (shared for doc and labels)
        label_texts:       List of k label texts Y_1..Y_k
        tokenizer:         BertTokenizer (to encode label texts per batch)
        num_labels:        k (number of labels)
        variant:           'L', 'L2', 'L2_NO_SA', 'L2_A5B', 'R', or 'B'
        max_len:           Max tokenization length for labels
    """
    
    def __init__(
        self,
        bert_model: BertModel,
        num_labels: int,
        variant: str = "L",
        separate_label_encoder: Optional[BertModel] = None,
    ):
        super().__init__()
        self.bert = bert_model
        self.num_labels = num_labels
        self.variant = variant
        self.separate_label_encoder = separate_label_encoder  # None -> shared (A2)
        H = self.bert.config.hidden_size
        
        if variant in ("L", "L2"):
            self.csam = CSAM_L(H, scale_similarity=(variant == "L2"))
        elif variant == "L2_NO_SA":
            self.csam = CSAM_L(H, scale_similarity=True, use_label_sa=False)
        elif variant == "L2_A5B":
            self.csam = CSAM_L_A5B(H, num_labels, scale_similarity=True, use_label_sa=True)
        elif variant == "R":
            self.csam = CSAM_R(H, num_labels, use_sa=True)
        elif variant == "B":
            self.classifier = nn.Linear(H, num_labels)
        else:
            raise ValueError(f"Unknown variant: {variant}. Choices: L, L2, L2_NO_SA, L2_A5B, R, B")
    
    def forward(
        self,
        doc_input_ids: torch.Tensor,       # [B, seq_len]
        doc_attention_mask: torch.Tensor,  # [B, seq_len]
        label_input_ids: torch.Tensor,     # [k, label_seq_len]
        label_attention_mask: torch.Tensor, # [k, label_seq_len]
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass.
        
        Labels are encoded inside the forward pass with gradients (A3).
        Efficiency: encode all k label texts once per batch (same for all docs in batch).
        This is mathematically identical to encoding per document.
        
        Returns:
            logits: [B, k]
            diagnostics: dict with CSAM saturation info
        """
        # Eq.1: Document encoding — [CLS] last hidden state
        doc_output = self.bert(doc_input_ids, attention_mask=doc_attention_mask)
        h_T = get_cls_embedding(doc_output)  # [B, H]
        
        if self.variant == "B":
            logits = self.classifier(h_T)
            return logits, {}
        
        # Eq.2: Label encoding — [CLS] last hidden state, with gradients (A3)
        label_encoder = self.separate_label_encoder if self.separate_label_encoder else self.bert
        label_output = label_encoder(label_input_ids, attention_mask=label_attention_mask)
        h_L = get_cls_embedding(label_output)  # [k, H]
        
        # CSAM (Eqs.3-8)
        logits, diag = self.csam(h_T, h_L)  # [B, k]
        
        return logits, diag


def build_bert_model(bert_name: str, device: torch.device) -> BertModel:
    """Load BERT model from pretrained checkpoint."""
    model = BertModel.from_pretrained(bert_name)
    model = model.to(device)
    return model


def init_a5b_from_l2(l2_model: LHRMModel, a5b_model: LHRMModel) -> None:
    """
    Controlled step-0 initialization for diagnostic parity (DR4 requirement).
    
    1. Copies the baseline shared classifier vector W in R^{1 x H} into all k
       rows of A5b unshared classifier W in R^{k x H}.
    2. Copies the shared scalar classifier bias b in R.
    3. Copies the BERT backbone parameters from l2_model to a5b_model.
    
    Guarantees:
    - Step-0 logits agree within float tolerance (< 1e-6).
    - Step-0 document-label alpha and label-label attention matrices agree identically.
    - Subsequent optimization allows gradients to differ across independent label rows.
    """
    if l2_model.variant not in ("L", "L2", "L2_NO_SA"):
        raise ValueError(f"Expected L/L2 model for source, got variant={l2_model.variant}")
    if a5b_model.variant != "L2_A5B":
        raise ValueError(f"Expected L2_A5B model for target, got variant={a5b_model.variant}")

    # Copy BERT backbone
    a5b_model.bert.load_state_dict(l2_model.bert.state_dict())

    # Copy shared classifier weight vector to all k rows
    shared_w = l2_model.csam.W.weight.data  # [1, H]
    k = a5b_model.num_labels
    a5b_model.csam.W.data.copy_(shared_w.repeat(k, 1))

    # Copy shared classifier bias
    shared_bias = l2_model.csam.W.bias.data  # [1]
    a5b_model.csam.bias.data.copy_(shared_bias)

