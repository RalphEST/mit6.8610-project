"""
Copied from https://github.com/facebookresearch/esm/blob/main/examples/variant-prediction/predict.py
"""
import torch
import logging
import numpy as np
from sklearn.metrics import precision_recall_curve

###############
# Scoring utils
###############
def marginal_score(variant_positions, wt_seq, mt_seq, token_probs, alphabet, offset_idx):
    wt_aas = [wt_seq[pos - offset_idx] for pos in variant_positions]
    mt_aas = [mt_seq[pos - offset_idx] for pos in variant_positions]

    wts_encoded, mts_encoded = [alphabet.get_idx(wt) for wt in wt_aas], [alphabet.get_idx(mt) for mt in mt_aas]

    # add 1 for BOS
    score = (token_probs[1 + variant_positions - offset_idx, mts_encoded] - token_probs[1 + variant_positions - offset_idx, wts_encoded]).sum()  # 1 because positions start from 1
    return score.item()


def compute_pppl(row, sequence, model, alphabet, offset_idx):
    wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
    assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"

    # modify the sequence
    sequence = sequence[:idx] + mt + sequence[(idx + 1) :]

    # encode the sequence
    data = [
        ("protein1", sequence),
    ]

    batch_converter = alphabet.get_batch_converter()

    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)

    # compute probabilities at each position
    log_probs = []
    for i in range(1, len(sequence) - 1):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, i] = alphabet.mask_idx
        with torch.no_grad():
            token_probs = torch.log_softmax(model(batch_tokens_masked.cuda())["logits"], dim=-1)
        log_probs.append(token_probs[0, i, alphabet.get_idx(sequence[i])].item())  # vocab size
    return sum(log_probs)


###############
# logging utils
###############
def get_root_logger(fname=None, file=True):
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")

    if file:
        handler = logging.FileHandler(f"{fname}.txt", mode='w')
        handler.setFormatter(format)
        logger.addHandler(handler)

    logger.addHandler(logging.StreamHandler())
    
    return logger


###############
# Evaluation utils
###############
def fmax_score(y: np.ndarray, preds: np.ndarray, beta = 1.0, pos_label = 1):
    """
    This is only for binary classification.
    Radivojac, P. et al. (2013). A Large-Scale Evaluation of Computational Protein Function Prediction. Nature Methods, 10(3), 221-227.
    """
    precision, recall, thresholds = precision_recall_curve(y, preds, pos_label=pos_label)
    precision += 1e-4
    recall += 1e-4
    f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    return np.nanmax(f1), thresholds[np.argmax(f1)]

