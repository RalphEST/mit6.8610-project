"""
From [batch_size, seq_len, vocab_size] to [batch_size, 2], where the 2 represents two methods: wt_marginal and mt_marginal.
"""
import torch


def marginal_score(mutation_indicators: torch.BoolTensor, 
                   wt_tokens: torch.LongTensor, 
                   mt_tokens: torch.LongTensor, 
                   logits: torch.FloatTensor):
    
    wt_var_toks = torch.split(wt_tokens[mutation_indicators], 
                              mutation_indicators.sum(dim=1).tolist(), dim=0)
    mt_var_toks = torch.split(mt_tokens[mutation_indicators],
                              mutation_indicators.sum(dim=1).tolist(), dim=0)
    
    scores = []
    for i, individual_logits in enumerate(torch.split(logits[mutation_indicators.bool()], 
                                                      mutation_indicators.bool().sum(dim=1).tolist())):
        # print((individual_logits[torch.arange(mt_var_toks[i].shape[0]), mt_var_toks[i]] - individual_logits[torch.arange(wt_var_toks[i].shape[0]), wt_var_toks[i]]))  # NOTE: Just for testing
        score = (individual_logits[torch.arange(mt_var_toks[i].shape[0]), mt_var_toks[i]] - individual_logits[torch.arange(wt_var_toks[i].shape[0]), wt_var_toks[i]]).sum()  # The only difference in mt- and wt- marginal is whether the context (i.e. the logits) is mt or wt
        scores.append(score)
    
    return torch.FloatTensor(scores)


@torch.no_grad()
def compute_variant_score(wt_logits, mt_logits, mt_indicators, wt_tokens, mt_tokens):
    """
    Calculate scores for variants of one protein (across patients). Scoring methods:
        - mt_marginal: \sum_{i\in M}{[\log p(x_i=x_i^{mt}|x^{mt}) - \log p(x_i=x_i^{wt}|x^{mt})]}
        - wt_marginal: \sum_{i\in M}{[\log p(x_i=x_i^{mt}|x^{wt}) - \log p(x_i=x_i^{wt}|x^{wt})]}
    """
    wt_marginal_scores = marginal_score(mt_indicators.bool(), wt_tokens, mt_tokens, wt_logits)
    mt_marginal_scores = marginal_score(mt_indicators.bool(), wt_tokens, mt_tokens, mt_logits)
    concat_scores = torch.stack([wt_marginal_scores, mt_marginal_scores]).T
    
    return concat_scores


def unsupervised_score(logits, mutation_indicators, tokens):
    """
    This function calculates unsupervised variant score for one gene across mutants. We consider two methods: wt-marginal and mt-marginal. For details, see page 20 in https://www.biorxiv.org/content/10.1101/2021.07.09.450648v2.full.pdf
    Parameters:
        - logits: [N+1, L+2, V]
        - mutation_indicators: [N+1, L]
        - tokens: [N+1, L+2]
    """
    assert mutation_indicators[0].sum() == 0, "First row of mutation indicators is not wild type"
    assert logits.shape[0] == mutation_indicators.shape[0]
    assert tokens.shape[0] == mutation_indicators.shape[0]
    assert logits.shape[1] == tokens.shape[1]
    assert logits.shape[1]-2 == mutation_indicators.shape[1]
    
    mt_logits = logits[1:, 1:-1, :]  # get rid of BOS and EOS in dim 1
    wt_logits = logits[0, 1:-1, :].repeat(mt_logits.shape[0], 1, 1)  # match mt logits dimension
    
    mt_indicators = mutation_indicators[1:, :]
    
    mt_tokens = tokens[1:, 1:-1]  # get rid of BOS and EOS in dim 1
    wt_tokens = tokens[0, 1:-1].repeat(mt_tokens.shape[0], 1)  # match mt tokens dimension
    
    mt_scores = compute_variant_score(wt_logits, mt_logits, mt_indicators, wt_tokens, mt_tokens)  # [N, 2]
    mt_scores = torch.cat([torch.Tensor([[0, 0]]), mt_scores], dim=0)
    
    return mt_scores
