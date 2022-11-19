from data_utils import VariationDataset
from baseline_utils import compute_pppl, marginal_score
from esm import pretrained

import random, argparse, wandb
from typing import List
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD, PCA

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# TODO: @Ralph, specify this
OUTPUT_DIR = None
TOKS_PER_BATCH = 4096
MAX_NUM_EPOCH = 1000


def create_parser():
    parser = argparse.ArgumentParser()
    
    # common args
    parser.add_argument("--pretrained_model", type=str, default='esm1v_t33_650M_UR90S_1', help="pretrained model")
    parser.add_argument("--dim_reduc", type=int, default=0, help="the desired output dimension for dimension reduction using SVD")
    parser.add_argument("--toks_per_batch", type=int, default=TOKS_PER_BATCH, help="maximum batch size")
    parser.add_argument("--repr_layers", type=int, default=[-1], nargs="+", help="layers indices from which to extract embeddings (0 to num_layers, inclusive)")  # Though we only coded for the final embedding
    parser.add_argument("--overwrite", action='store_true', help="whether to regenerate and overwrite existing embeddings or not")
    parser.add_argument("--slide_window", action='store_true', help="if True, use sliding window to generate embeddings instead of truncating at 1022")
    parser.add_argument("--variant_readout", type=str, default='mean', help="variant dim readout")
    parser.add_argument("--patient_readout", type=str, default='mean', help="patient dim readout")
    parser.add_argument("--unsup_patient_score_readout", type=str, default='max', help="summary patient fitness score from multiple genes' scores", choices=['max', 'weighted_mean', 'mean', 'median'])
    
    # parser.add_argument("--seq_data_path", type=str, default=None, help="seq data path")
    # parser.add_argument("--seq_table_path", type=str, default=None, help="seq table path")
    # parser.add_argument("--var_table_path", type=str, default=None, help="variant table path")
    # parser.add_argument("--pat_data_path", type=str, default=None, help="patient data path")
    # parser.add_argument("--phen_data_path", type=str, default=None, help="phen data path")
    # TODO: @Ralph, specify these
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="output directory for results (classifier, rank lists, etc.)")
    
    parser.add_argument('--finetune', action='store_true', help="whether to finetune the pretrained model or not")
    parser.add_argument('--variant_head', type=str, default='unsup', help='how to make predictions for variants', choices=['unsup_wt_marginal', 'unsup_mt_marginal', 'sup_mlp', 'sup_knn', 'sup_lr'])
    parser.add_argument("--use_sklearn", action='store_true', help="use sklearn instead of pytorch")
    parser.add_argument("--classifier", type=str, default='mlp', help="classifier")
    parser.add_argument("--random_state", type=int, default=42, help="random state")
    parser.add_argument("--train_size", type=float, default=0.6, help="size of train set")
    parser.add_argument("--val_size", type=float, default=0.2, help="size of val set")
    parser.add_argument("--hidden_dim_1", type=int, default=512, help="1st hidden dim size")
    parser.add_argument("--hidden_dim_2", type=int, default=256, help="2nd hidden dim size, discard if 0")
    parser.add_argument("--hidden_dim_3", type=int, default=0, help="3rd hidden dim size, discard if 0")
    parser.add_argument("--dropout", type=float, default=0, help="dropout rate")
    parser.add_argument("--norm", type=str, default=None, help="normalization layer")
    parser.add_argument("--actn", type=str, default='relu', help="activation type")
    parser.add_argument("--order", type=str, default='nd', help="order of normalization and dropout")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--num_epoch", type=int, default=MAX_NUM_EPOCH, help="epoch num")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    parser.add_argument("--weigh_sample", action='store_true', help="weighted sampler")
    parser.add_argument("--weigh_loss", action='store_true', help="weighted loss")

    return parser


# No need to change this
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list, output_dim: int, p: float, norm: str, actn: str, order: str = 'nd'):
        super(MLP, self).__init__()
        self.n_layer = len(hidden_dims) - 1
        self.in_dim = in_dim
        
        actn2actfunc = {'relu': nn.ReLU(), 'leakyrelu': nn.LeakyReLU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid(), 'selu': nn.SELU(), 'softplus': nn.Softplus(), 'gelu': nn.GELU()}
        
        try:
            actn = actn2actfunc[actn]
        except:
            print(actn)
            raise NotImplementedError

        # input layer
        layers = [nn.Linear(self.in_dim, hidden_dims[0]), actn]
        # hidden layers
        for i in range(self.n_layer):
            layers += self.compose_layer(
                in_dim=hidden_dims[i], out_dim=hidden_dims[i+1], norm=norm, actn=actn, p=p, order=order
            )
        # output layers
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.fc = nn.Sequential(*layers)

    def compose_layer(
        self,
        in_dim: int,
        out_dim: int,
        norm: str,
        actn: nn.Module,
        p: float = 0.0,
        order: str = 'nd'
    ):
        norm2normlayer = {'bn': nn.BatchNorm1d(in_dim), 'ln': nn.LayerNorm(in_dim), None: None, 'None': None}  # because in_dim is only fixed here
        try:
            norm = norm2normlayer[norm]
        except:
            print(norm)
            raise NotImplementedError
        # norm --> dropout or dropout --> norm
        if order == 'nd':
            layers = [norm] if norm is not None else []
            if p != 0:
                layers.append(nn.Dropout(p))
        elif order == 'dn':
            layers = [nn.Dropout(p)] if p != 0 else []
            if norm is not None:
                layers.append(norm)
        else:
            print(order)
            raise NotImplementedError

        layers.append(nn.Linear(in_dim, out_dim))
        if actn is not None:
            layers.append(actn)
        return layers

    def forward(self, x):
        output = self.fc(x)
        return output


def get_metrics(scores, labels):
    def fmax_score(y: np.ndarray, preds: np.ndarray, beta = 1.0, pos_label = 1):
        """
        Radivojac, P. et al. (2013). A Large-Scale Evaluation of Computational Protein Function Prediction. Nature Methods, 10(3), 221-227.
        """
        precision, recall, thresholds = precision_recall_curve(y, preds, pos_label)
        precision += 1e-4
        recall += 1e-4
        f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
        return np.nanmax(f1), thresholds[np.argmax(f1)]
    
    
        
    return metrics


@torch.no_grad()
def evaluate():
    pass


def sup_train_epoch():
    pass


def sup_train(protein_embed_data, phen_data, args, model_type='mlp'):
    pass


@torch.no_grad()
def unsupervised_variant_score(model, alphabet, variant_embeds, var_positions, wt_protein_embed, wt_protein_seq, var_seqs, args):
    """
    Calculate scores for variants of one protein (across patients). If args.unsup_scoring_strategy == 
        mt_marginal: \sum_{i\in M}{[\log p(x_i=x_i^{mt}|x^{mt}) - \log p(x_i=x_i^{wt}|x^{mt})]}
        wt_marginal: \sum_{i\in M}{[\log p(x_i=x_i^{mt}|x^{wt}) - \log p(x_i=x_i^{wt}|x^{wt})]}
    """
    variant_token_probs = model.lm_head(variant_embeds).detach().cpu()  # get logits from last layer repr, [batch_size, num_AA+2, vocab_size]
    wt_protein_token_probs = model.lm_head(wt_protein_embed).detach().cpu()  # get logits from last layer repr, [num_AA+2, vocab_size]
    
    if args.variant_head == "unsup_wt_marginal":
        scores = [marginal_score(var_position, wt_protein_seq, var_seq, wt_protein_token_probs, alphabet, args.offset_idx) for var_position, var_seq in zip(var_positions, var_seqs)]
    elif args.variant_head == "unsup_mt_marginal":
        scores = [marginal_score(var_position, wt_protein_seq, var_seq, variant_token_probs[i], alphabet, args.offset_idx) for i, var_position, var_seq in enumerate(zip(var_positions, var_seqs))]
        
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


def unsupervised_score(variant_embed_data, variant_positions, wt_protein_embeds, wt_protein_seqs, variant_seqs, args):
    """
    For now, consider only the mutant marginal probability.
    If using masked scoring schemes (masked marginal probability or pseudo ppl), we need to compute the scores for each variant separately. We'll need to get variant embeddings for every variant's every masked version (masking one position at a time), which is computationally too expensive. We also need to think of a way to combine the scores from all sequences.
    For details, see page 20 in https://www.biorxiv.org/content/10.1101/2021.07.09.450648v2.full.pdf
    """
    model, alphabet = pretrained.load_model_and_alphabet(args.pretained_model)
    model.eval()
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = model.cuda()
        print("Transferred model to GPU")

    # batch_converter = alphabet.get_batch_converter()    
    # data = None
    # batch_labels, batch_strs, batch_tokens = batch_converter(data)  # [batch_size, seq_len]

    prot_scores = {}
    for prot, variant_embeds in variant_embed_data.items():
        # variant_embeds: [num_patients, seq_len, embed_dim]
        var_positions = variant_positions[prot]
        var_seqs = variant_seqs[prot]
        wt_protein_seq = wt_protein_seqs[prot]
        wt_protein_embed = wt_protein_embeds[prot]
        scores = unsupervised_variant_score(model, alphabet, variant_embeds.to(device), var_positions, wt_protein_embed, wt_protein_seq, var_seqs, args)        
        prot_scores[prot] = scores
    
    if args.unsup_patient_score_readout == 'max':
        patient_scores = np.array([max(scores) for scores in prot_scores.values()])
    else:
        raise NotImplementedError  # TODO
    
    return patient_scores


def main(args):
    ################# THIS PART SHOULD BE ADAPTED TO FIT THE REAL DATA #################
    # TODO: @Ralph, specify this and convert this into some way
    # VariationDataset()
    NUM_PROTEINS = 20
    NUM_PATIENTS = 100
    EMBED_DIM = 768
    patients = np.arange(NUM_PATIENTS)  # all patients
    proteins = np.arange(NUM_PROTEINS)  # all unique protein names, NOT VARIANTS!
    wt_protein_seqs = {prot:np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), np.random.randint(1022)).tolist() for prot in proteins}  # all wildtype protein sequences (first 1022 AAs)
    variant_seqs = {prot:[np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), np.random.randint(1022)).tolist()] * NUM_PATIENTS for prot in proteins}  # all mutant protein sequences (first 1022 AAs), here we just let everyone has the same mutant sequence
    
    # TODO: @ Ralph, please also specify this
    # For now I just use randomly generated data.
    # Let's say we have 100 patients, 20 proteins, each protein with a random number of AAs, each AA with 768-dim features
    # Somehow get embeddings using those frozen ones generated by ESM2_inf 
    # Generate {protein: {torch.tensor([num_patient, num_AA+2, feature_dim])}}
    variant_embed_data = {prot:torch.randn(NUM_PATIENTS, len(wt_protein_seqs[prot]), EMBED_DIM) for prot in proteins}  # NOTE: Each tensor follows the same order of patients in the first dimension
    
    # Also get wild type protein embeddings
    wt_protein_embeds = {prot:torch.randn(len(wt_protein_seqs[prot]), EMBED_DIM) for prot in proteins}
    
    phenotypes = np.random.choice([0, 1], NUM_PATIENTS)  # all patients' phenotypes (it can be easily adapted to multilabel)
    phen_data = pd.DataFrame.from_dict({'patient':patients, 'phen_1':phenotypes, 'phen_2':phenotypes}).set_index('patient')
    
    variant_positions = {prot:{patient:[np.random.randint(np.arange(variant_embed_data[prot].shape[1]))] * 3 for patient in patients} for prot in proteins}  # {protein: {patient: [variant_position]}}, here assuming all proteins have 3 mutated AAs.  NOTE: POSITION should be set to start from 0
    ####################################################################################
    
    # Random split of patients: 6-2-2
    train_val_ind, test_ind = train_test_split(np.arange(NUM_PATIENTS), test_size=0.2, random_state=args.random_state)
    train_ind, val_ind = train_test_split(range(len(train_val_ind)), test_size=0.25, random_state=args.random_state)
    train_ind = train_val_ind[train_ind]
    val_ind = train_val_ind[val_ind]
    
    # TODO: Needs further discussion for the implementation of masked approaches
    if args.variant_head.startswith('unsup'):
        scores = unsupervised_score(variant_embed_data, variant_positions, wt_protein_embeds, wt_protein_seqs, variant_seqs, args)
        metrics = get_metrics(scores, phen_data)
        
        train_metrics = metrics[train_ind]
        val_metrics = metrics[val_ind]
        test_metrics = metrics[test_ind]
        
        
    
    # In SUP setting, needs to go from AA embeddings to variant embedding
    # From {protein: {torch.tensor([num_patient, num_AA+2, feature_dim])}} to {protein: {torch.tensor([num_patient, feature_dim])}}
    if args.variant_readout == 'mean':
        variant_embed_data = {prot:data[:, 1:-1, :].mean(dim=1) for prot, data in variant_embed_data.items()}    
    elif args.variant_readout == 'bos':
        variant_embed_data = {prot:data[:, 0, :] for prot, data in variant_embed_data.items()}
    elif args.variant_readout == 'mean_mean':
        # TODO: other ways to get variant embedding, e.g. concatenate sequence-wise mean repr and mean repr across variant positions
        raise NotImplementedError
    else:
        raise NotImplementedError
    
    # From multiple protein embeddings to patient embedding
    # From {protein: {torch.tensor([num_patient, feature_dim])}} to torch.tensor([num_patient, feature_dim])
    if args.patient_reaout == 'mean':
        protein_name_ind_map = list(variant_embed_data.keys())
        patient_embed_data = torch.stack(list(variant_embed_data.values())).mean(0)  # [num_patient, feature_dim]
    elif args.patient_readout == 'concat':
        protein_name_ind_map = list(variant_embed_data.keys())
        patient_embed_data = torch.cat(list(variant_embed_data.values()), dim=1)  # [num_patient, num_protein*feature_dim]
    else:
        raise NotImplementedError
    
    patient_embed_data_train = patient_embed_data[train_ind]
    patient_embed_data_val = patient_embed_data[val_ind]
    patient_embed_data_test = patient_embed_data[test_ind]
    phen_data_train = phen_data.iloc[train_ind]
    phen_data_val = phen_data.iloc[val_ind]
    phen_data_test = phen_data.iloc[test_ind]
        
    # Train supervised model
    if args.variant_head == 'sup_mlp':
        predictor = sup_train(patient_embed_data_train, patient_embed_data_val, phen_data_train, phen_data_val, args, model_type='mlp')
    elif args.variant_head == 'sup_lr':
        predictor = sup_train(patient_embed_data_train, patient_embed_data_val, phen_data_train, phen_data_val, args, model_type='lr')
    elif args.variant_head == 'sup_knn':
        predictor = sup_train(patient_embed_data_train, patient_embed_data_val, phen_data_train, phen_data_val, args, model_type='knn')
    
    # Score test variants
    predictor.eval()
    train_metrics = evaluate
    


if __name__ == "__init__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)

