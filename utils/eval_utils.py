import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score, 
    precision_recall_curve
)

def print_metrics(metrics):
    for k,v in metrics.items():
        print(f"\t{k}: {v:.3f}")

def get_metrics(y_true, y_score):
    """
    Get metrics for multilabel classification.
    Parameters:
        y_true: np.array, shape (n_samples, n_labels)
        y_preds: np.array, shape (n_samples, n_labels)
    """
    assert len(y_true) == len(y_score), "Predictions and labels are of different shapes"
    y_preds = torch.sigmoid(y_score).round()
    metrics = {"accuracy"  : accuracy_score(y_true, y_preds),
               "precision" : precision_score(y_true, y_preds, zero_division=0),
               "recall"    : recall_score(y_true, y_preds, zero_division=0),
               "f1"        : f1_score(y_true, y_preds, zero_division=0),
               "AUROC"     : roc_auc_score(y_true, y_score)}
            
    return metrics

# scoring utils

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

# evaluation utils

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


@torch.no_grad()
def unsupervised_variant_score(model, 
                               alphabet, 
                               variant_embeds, 
                               var_positions, 
                               wt_protein_embed, 
                               wt_protein_seq, 
                               var_seqs, 
                               args):
    """
    Calculate scores for variants of one protein (across patients). If args.unsup_scoring_strategy == 
        mt_marginal: \sum_{i\in M}{[\log p(x_i=x_i^{mt}|x^{mt}) - \log p(x_i=x_i^{wt}|x^{mt})]}
        wt_marginal: \sum_{i\in M}{[\log p(x_i=x_i^{mt}|x^{wt}) - \log p(x_i=x_i^{wt}|x^{wt})]}
    """
    if args.variant_head == "unsup_wt_marginal":
        wt_protein_token_probs = model.lm_head(wt_protein_embed).detach().cpu()  # get logits from last layer repr, [num_AA+2, vocab_size]
        scores = [marginal_score(var_position, wt_protein_seq, var_seq, wt_protein_token_probs, alphabet, args.offset_idx) for (_, var_position), var_seq in zip(var_positions.items(), var_seqs)]
    elif args.variant_head == "unsup_mt_marginal":
        variant_token_probs = model.lm_head(variant_embeds).detach().cpu()  # get logits from last layer repr, [batch_size, num_AA+2, vocab_size]
        scores = [marginal_score(var_position, wt_protein_seq, var_seq, variant_token_probs[i], alphabet, args.offset_idx) for i, (_, var_position), var_seq in enumerate(zip(var_positions.items(), var_seqs))]
        
    # elif args.scoring_strategy == "masked-marginals":
    #     all_token_probs = []
    #     for i in tqdm(range(batch_tokens.size(1))):
    #         batch_tokens_masked = batch_tokens.clone()
    #         batch_tokens_masked[0, i] = alphabet.mask_idx
    #         with torch.no_grad():
    #             token_probs = torch.log_softmax(
    #                 model(batch_tokens_masked.cuda())["logits"], dim=-1
    #             )
    #         all_token_probs.append(token_probs[:, i])  # vocab size
    #     token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
    #     scores = df.apply(
    #         lambda row: label_row(
    #             row[args.mutation_col],
    #             args.sequence,
    #             token_probs,
    #             alphabet,
    #             args.offset_idx,
    #         ),
    #         axis=1,
    #     )
    # elif args.scoring_strategy == "pseudo-ppl":
    #     tqdm.pandas()
    #     scores = df.progress_apply(
    #         lambda row: compute_pppl(
    #             row[args.mutation_col], args.sequence, model, alphabet, args.offset_idx
    #         ),
    #         axis=1,
    #     )
    
    else:
        raise NotImplementedError # TODO
    
    return scores


