import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from data.data_utils import VariationDataset
from baseline_utils import *
from esm import pretrained

import random, argparse
from typing import List
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# TODO: @Ralph, specify this
OUTPUT_DIR = '.'
TOKS_PER_BATCH = 4096
MAX_NUM_EPOCH = 500


def parse_args():
    parser = argparse.ArgumentParser()
    
    # common args
    parser.add_argument("--pretrained_model", type=str, default='esm1v_t33_650M_UR90S_1', help="pretrained model")
    parser.add_argument("--toks_per_batch", type=int, default=TOKS_PER_BATCH, help="maximum batch size")
    parser.add_argument("--repr_layers", type=int, default=[-1], nargs="+", help="layers indices from which to extract embeddings (0 to num_layers, inclusive)")  # Though we only coded for the final embedding
    parser.add_argument("--overwrite", action='store_true', help="whether to regenerate and overwrite existing embeddings or not")
    parser.add_argument("--slide_window", action='store_true', help="if True, use sliding window to generate embeddings instead of truncating at 1022")
    parser.add_argument("--variant_readout", type=str, default='mean', help="variant dim readout", choices=['mean', 'mean_mean', 'bos', 'eos'])
    parser.add_argument("--patient_readout", type=str, default='mean', help="patient dim readout", choices=['concat', 'mean', 'max', 'attention'])
    parser.add_argument("--dim_reduc", type=int, default=0, help="the desired output dimension for dimension reduction using SVD")
    parser.add_argument("--unsup_patient_score_readout", type=str, default='max', help="summary patient fitness score from multiple genes' scores", choices=['max', 'weighted_mean', 'mean', 'median'])
    
    # parser.add_argument("--seq_data_path", type=str, default=None, help="seq data path")
    # parser.add_argument("--seq_table_path", type=str, default=None, help="seq table path")
    # parser.add_argument("--var_table_path", type=str, default=None, help="variant table path")
    # parser.add_argument("--pat_data_path", type=str, default=None, help="patient data path")
    # parser.add_argument("--phen_data_path", type=str, default=None, help="phen data path")
    # TODO: @Ralph, specify these
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="output directory for results (classifier, rank lists, etc.)")
    parser.add_argument('--log_file', type='str', default='log', help='log file name')
    
    # parser.add_argument('--finetune', action='store_true', help="whether to finetune the pretrained model or not")  # NOTE: FIX ALL EMBEDDINGS
    parser.add_argument('--variant_head', type=str, default='unsup_wt_marginal', help='how to make predictions for variants', choices=['unsup_wt_marginal', 'unsup_mt_marginal', 'sup_mlp', 'sup_knn', 'sup_lr'])
    parser.add_argument('--offset_idx', type=int, default=1, help='offset index for positioning of sequence variants')
    parser.add_argument("--use_sklearn", action='store_true', help="use sklearn instead of pytorch")
    parser.add_argument("--classifier", type=str, default='mlp', help="classifier")
    parser.add_argument("--random_state", type=int, default=42, help="random state")
    parser.add_argument("--train_size", type=float, default=0.6, help="size of train set")
    parser.add_argument("--val_size", type=float, default=0.2, help="size of val set")
    parser.add_argument("--hidden_dim_1", type=int, default=512, help="1st hidden dim size")  # separating layers for wandb
    parser.add_argument("--hidden_dim_2", type=int, default=256, help="2nd hidden dim size, discard if 0")
    parser.add_argument("--hidden_dim_3", type=int, default=0, help="3rd hidden dim size, discard if 0")
    parser.add_argument("--dropout", type=float, default=0, help="dropout rate")
    parser.add_argument("--norm", type=str, default=None, help="normalization layer")
    parser.add_argument("--actn", type=str, default='relu', help="activation type")
    parser.add_argument("--order", type=str, default='nd', help="order of normalization and dropout")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-2, help="weight decay")
    parser.add_argument("--num_epoch", type=int, default=MAX_NUM_EPOCH, help="epoch num")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size")
    parser.add_argument("--weigh_sample", action='store_true', help="weighted sampler")
    parser.add_argument("--weigh_loss", action='store_true', help="weighted loss")

    args = parser.parse_args()
    
    return args


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
    """
    Get metrics for multilabel classification.
    Parameters:
        scores: np.array, shape (n_samples, n_labels)
        labels: np.array, shape (n_samples, n_labels)
    """
    if scores.shape != labels.shape:
        scores = np.tile(scores, (labels.shape[1], 1)).T
    
    auprc_macro = average_precision_score(labels, scores, average='macro')
    auprc_micro = average_precision_score(labels, scores, average='micro')
    auprc_weighted = average_precision_score(labels, scores, average='weighted')
    auroc_macro = roc_auc_score(labels, scores, average='macro')
    auroc_micro = roc_auc_score(labels, scores, average='micro')
    auroc_weighted = roc_auc_score(labels, scores, average='weighted')
    
    fmax = 0
    for i in range(labels.shape[1]):
        fmax += fmax_score(labels[:, i], scores[:, i])[0]
    fmax /= labels.shape[1]  # fmax macro
    
    metrics = {'auroc_macro': auroc_macro, 'auroc_micro': auroc_micro, 'auroc_weighted': auroc_weighted, 'auprc_macro': auprc_macro, 'auprc_micro': auprc_micro, 'auprc_weighted': auprc_weighted, 'fmax_macro': fmax}
    
    # TODO: generate ranked prediction indices list
        
    return metrics


@torch.no_grad()
def evaluate(predictor, patient_embed_data, phen_data, args):
    if args.use_sklearn:
        y_preds = predictor.predict_proba(patient_embed_data.numpy())
    else:
        y_preds = F.sigmoid(predictor(patient_embed_data)).detach().numpy()
        
    # Evaluation
    metrics = get_metrics(y_preds, phen_data)

    # pd.DataFrame({'y':sorted_y, 'preds':sorted_y_preds, 'patient':sorted_patient}).to_csv(f'{args.output_dir}/{args.pretrained_model}_{args.variant_head}_test_preds.csv', index=False)  # Save the test predictions

    ## save model stuff
    # if args.use_sklearn:
    #     mod_hparams = predictor.best_estimator_.get_params()
    #     hparam_save_path = args.model_output_prefix + "_hparams.pkl"
    #     with open(hparam_save_path, 'wb') as f:
    #         pickle.dump(mod_hparams, f)
    #     model_save_path = args.model_output_prefix + ".pkl"
    #     with open(model_save_path, 'wb') as f:
    #         pickle.dump(predictor, f)
    # else:    
    #     model_save_path = args.model_output_prefix + ".pt"
    #     torch.save({'epoch':best_epoch, 'model_state_dict':predictor.state_dict(), 'val_auprc':best_val_auprc}, model_save_path)

    return metrics


def sup_train_epoch(model, train_loader, optim, loss_func, batch_size, logger, device):
    model.train()
    train_size = len(train_loader.dataset)
    total_sample = total_loss = 0
    all_y = torch.tensor([])
    all_preds = torch.tensor([])
    for i, (X, y) in enumerate(train_loader):
        all_y = torch.cat([all_y, y])
        X, y = X.to(device), y.to(device)
        optim.zero_grad()

        preds = model(X)
        loss = loss_func(preds, y.float())
        
        loss.backward()
        optim.step()

        all_preds = torch.cat([all_preds, preds.cpu()])
        total_sample += batch_size
        total_loss += float(loss) * batch_size

        if i % 20 == 0:
            loss, current = loss.item(), i * len(X)
            logger.warning(f'train loss: {loss:.4f} [{current}/{train_size}]')
    
    all_y = all_y.detach().numpy().astype(int)
    all_preds = F.sigmoid(all_preds).detach().numpy()
    
    auroc_macro, auroc_micro, auroc_weighted, auprc_macro, auprc_micro, auprc_weighted, fmax_macro = tuple(get_metrics(all_preds, all_y).values())
    
    logger.warning('Train metrics:')
    logger.warning(f'auroc_macro = {auroc_macro}, auroc_micro = {auroc_micro}, auroc_weighted = {auroc_weighted}, auprc_macro = {auprc_macro}, auprc_micro = {auprc_micro}, auprc_weighted = {auprc_weighted}, fmax_macro = {fmax_macro}')
    
    total_loss = total_loss / total_sample  # weighted total train loss
    
    return total_loss, auprc_macro, all_y, all_preds


def sup_validate_epoch(model, val_loader, loss_func, logger, device):
    val_size = len(val_loader.dataset)
    model.eval()
    val_loss = 0
    all_y = torch.tensor([])
    all_preds = torch.tensor([])
    with torch.no_grad():
        for X, y in val_loader:
            all_y = torch.cat([all_y, y], dim=0)
            X, y = X.to(device), y.to(device)
            preds = model(X)
            all_preds = torch.cat([all_preds, preds.cpu()], dim=0)
            val_loss += loss_func(preds, y.float()).item() * X.shape[0]
        val_loss /= val_size
        y, preds = all_y.detach().numpy(), F.sigmoid(all_preds).detach().numpy()
        auroc_macro, auroc_micro, auroc_weighted, auprc_macro, auprc_micro, auprc_weighted, fmax_macro = tuple(get_metrics(preds, y).values())
        
        logger.warning('Val metrics:')
        logger.warning(f'auroc_macro = {auroc_macro}, auroc_micro = {auroc_micro}, auroc_weighted = {auroc_weighted}, auprc_macro = {auprc_macro}, auprc_micro = {auprc_micro}, auprc_weighted = {auprc_weighted}, fmax_macro = {fmax_macro}')
    
    return val_loss, auprc_macro, y, preds


def sup_train(patient_embed_data_train, patient_embed_data_val, phen_data_train, phen_data_val, args, logger):
    logger.warning(f'Using model: {args.variant_head[4:]}')
    if args.use_sklearn:
        if args.variant_head == 'sup_lr':
            param_grid = {'C':[1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100], 'l1_ratio':np.linspace(0, 1, 20)}
            # param_grid = {'l1_ratio':[0.5]}  # for testing only
            model = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=20000, n_jobs=-1)
        elif args.variant_head == 'sup_knn':
            param_grid = {'n_neighbor':[1, 5, 10, 20, 50, 100], 'metric':['euclidean', 'manhattan', 'cosine']}
            model = KNeighborsClassifier(n_jobs=-1)
        else:
            raise NotImplementedError

        X = torch.cat([patient_embed_data_train, patient_embed_data_val], dim=0).numpy()
        y = torch.cat([phen_data_train, phen_data_val], dim=0).numpy()
        clf = GridSearchCV(model, param_grid, scoring='average_precision', cv=round(X.shape[0]/patient_embed_data_val.shape[0]), n_jobs=-1)  # auprc macro 
        clf.fit(X, y)
        # clf.predict_proba(X_test)

    else:
        best_val_auprc = 0       
        train_dataset = TensorDataset(patient_embed_data_train, phen_data_train)
        val_dataset = TensorDataset(patient_embed_data_val, phen_data_val)
        train_size = len(train_dataset)

        drop_last = False
        if (args.norm in {'bn', 'ln'}) and len(train_dataset) % args.batch_size < 3:
            drop_last = True 
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=drop_last)
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if args.hidden_dim_2 == 0:
            hidden_dims = [args.hidden_dim_1]
        elif args.hidden_dim_3 == 0:
            hidden_dims = [args.hidden_dim_1, args.hidden_dim_2]
        else:
            hidden_dims = [args.hidden_dim_1, args.hidden_dim_2, args.hidden_dim_3]
        
        model = MLP(in_dim=patient_embed_data_train.shape[1], hidden_dims=hidden_dims, output_dim = phen_data_train.shape[1], p=args.dropout, norm=args.norm, actn=args.actn, order=args.order)
        try:
            model = nn.DataParallel(model).to(device)
        except:
            model = model.to(device)
        
        # if use_focal_loss:
        #     loss_func = FocalLoss(alpha=hparams['alpha'], gamma=hparams['gamma'])
        loss_func = nn.BCEWithLogitsLoss()
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

        for i in range(args.num_epoch):
            logger.warning(f"Epoch {i+1}/{args.num_epoch}\n---------------")
            _, _, train_y, train_preds = sup_train_epoch(model, train_loader, optim, loss_func, args.batch_size, logger, device)
            _, val_auprc, val_y, val_preds = sup_validate_epoch(model, val_loader, loss_func, logger, device)
            if val_auprc > best_val_auprc:
                clf = deepcopy(model)
                best_val_auprc = val_auprc
                best_epoch = i
                best_val_y = val_y.copy()
                best_val_preds = val_preds.copy()
                best_train_y = train_y.copy()
                best_train_preds = train_preds.copy()
    
    return clf.module.to('cpu')


@torch.no_grad()
def unsupervised_variant_score(model, alphabet, variant_embeds, var_positions, wt_protein_embed, wt_protein_seq, var_seqs, args):
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


def unsupervised_score(variant_embed_data, variant_positions, wt_protein_embeds, wt_protein_seqs, variant_seqs, args):
    """
    For now, consider only the mutant marginal probability.
    If using masked scoring schemes (masked marginal probability or pseudo ppl), we need to compute the scores for each variant separately. We'll need to get variant embeddings for every variant's every masked version (masking one position at a time), which is computationally too expensive. We also need to think of a way to combine the scores from all sequences.
    For details, see page 20 in https://www.biorxiv.org/content/10.1101/2021.07.09.450648v2.full.pdf
    """
    model, alphabet = pretrained.load_model_and_alphabet(args.pretrained_model)
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
        scores = unsupervised_variant_score(model, alphabet, variant_embeds.to(device), var_positions, wt_protein_embed.to(device), wt_protein_seq, var_seqs, args)        
        prot_scores[prot] = scores
    
    if args.unsup_patient_score_readout == 'max':
        patient_scores = np.array(list(prot_scores.values())).max(axis=0)
    else:
        raise NotImplementedError  # TODO
    
    patient_scores = MinMaxScaler().fit_transform(np.array(list(prot_scores.values())).max(axis=0)[:, None]).T[0]  # Scale scores back to [0, 1]
    
    return patient_scores


def main(args):
    logger = get_root_logger(args.log_fie)
    logger.warning('Loading data...')
    
    ################# THIS PART SHOULD BE ADAPTED TO FIT THE REAL DATA #################
    # TODO: @Ralph, specify this and convert this into some way
    # Also make sure the patient order is always the same
    # VariationDataset()
    NUM_PROTEINS = 20
    NUM_PATIENTS = 100
    EMBED_DIM = 1280
    patients = np.arange(NUM_PATIENTS)  # all patients
    proteins = np.arange(NUM_PROTEINS)  # all unique protein names, NOT VARIANTS!
    wt_protein_seqs = {prot:np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), np.random.randint(1022)+1).tolist() for prot in proteins}  # all wildtype protein sequences (first 1022 AAs)
    variant_seqs = {prot:[np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), len(wt_protein_seqs[prot])).tolist() for _ in patients] for prot in proteins}  # all mutant protein sequences (first 1022 AAs), here we just let everyone has the same mutant sequence, not considering deletions or insertions
    
    # TODO: @ Ralph, please also specify this
    # For now I just use randomly generated data.
    # Let's say we have 100 patients, 20 proteins, each protein with a random number of AAs, each AA with 768-dim features
    # Somehow get embeddings using those frozen ones generated by ESM2_inf 
    # Generate {protein: {torch.tensor([num_patient, num_AA+2, feature_dim])}}
    variant_embed_data = {prot:torch.randn(NUM_PATIENTS, len(wt_protein_seqs[prot])+2, EMBED_DIM) for prot in proteins}  # incl. BOS and EOS.  NOTE: Each tensor follows the same order of patients in the first dimension
    
    # Also get wild type protein embeddings
    wt_protein_embeds = {prot:torch.randn(len(wt_protein_seqs[prot])+2, EMBED_DIM) for prot in proteins}  # incl. BOS and EOS
    
    phenotypes = np.random.choice([0, 1], NUM_PATIENTS)  # all patients' phenotypes (it can be easily adapted to multilabel)
    phen_data = pd.DataFrame.from_dict({'patient':patients, 'phen_1':phenotypes, 'phen_2':phenotypes}).set_index('patient')
    
    variant_positions = {prot:{patient:np.random.choice(np.arange(len(wt_protein_seqs[prot]))+1, 3) for patient in patients} for prot in proteins}  # {protein: {patient: [variant_position]}}, here assuming all proteins have 3 mutated AAs.  NOTE: POSITION should be set to start from 1, not 0
    ####################################################################################
    
    # Random split of patients: 6-2-2
    train_val_ind, test_ind = train_test_split(np.arange(NUM_PATIENTS), test_size=0.2, random_state=args.random_state)
    train_ind, val_ind = train_test_split(range(len(train_val_ind)), test_size=0.25, random_state=args.random_state)
    train_ind = train_val_ind[train_ind]
    val_ind = train_val_ind[val_ind]
    
    # TODO: Needs further discussion for the implementation of masked approaches
    if args.variant_head.startswith('unsup'):
        logger.warning(f'Computing **unsupervised** scores... ')
        logger.warning(f'Calculation method: {args.variant_head[6:]}')
        logger.warning(f'Multi-gene readout method: {args.unsup_patient_score_readout}')
        
        scores = unsupervised_score(variant_embed_data, variant_positions, wt_protein_embeds, wt_protein_seqs, variant_seqs, args)
        
        train_metrics = get_metrics(scores[train_ind], phen_data.values[train_ind, :])
        val_metrics = get_metrics(scores[val_ind], phen_data.values[val_ind, :])
        test_metrics = get_metrics(scores[test_ind], phen_data.values[test_ind, :])
    
    else:
        logger.warning(f'Computing **supervised** scores... ')
        logger.warning(f'Calculation method: {args.variant_head[4:]}')
        logger.warning(f'Variant readout method: {args.variant_readout}')
        logger.warning(f'Patient readout method: {args.patient_readout}')
                
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
        if args.patient_readout == 'mean':
            protein_name_ind_map = list(variant_embed_data.keys())  # keep track because we might be interested in specific proteins
            patient_embed_data = torch.stack(list(variant_embed_data.values())).mean(0)  # [num_patient, feature_dim]
        elif args.patient_readout == 'concat':
            protein_name_ind_map = list(variant_embed_data.keys())
            patient_embed_data = torch.cat(list(variant_embed_data.values()), dim=1)  # [num_patient, num_protein*feature_dim]
        else:
            raise NotImplementedError
        
        patient_embed_data_train = patient_embed_data[train_ind]
        patient_embed_data_val = patient_embed_data[val_ind]
        patient_embed_data_test = patient_embed_data[test_ind]
        phen_data_train = torch.from_numpy(phen_data.iloc[train_ind].values).type(torch.int)
        phen_data_val = torch.from_numpy(phen_data.iloc[val_ind].values).type(torch.int)
        phen_data_test = torch.from_numpy(phen_data.iloc[test_ind].values).type(torch.int)
            
        # Train supervised model
        predictor = sup_train(patient_embed_data_train, patient_embed_data_val, phen_data_train, phen_data_val, args, logger)
        
        # Score test variants
        predictor.eval()
        train_metrics = evaluate(predictor, patient_embed_data_train, phen_data_train, args)
        val_metrics = evaluate(predictor, patient_embed_data_val, phen_data_val, args)
        test_metrics = evaluate(predictor, patient_embed_data_test, phen_data_test, args)
        
        # TODO: Save predictions, save best model
    
    logger.warning('\nBest model performance ---------------------')
    logger.warning('\nEvaluation on training set:')
    for k, v in train_metrics.items():
        logger.warning("train %s: %g" % (k, v))
        
    logger.warning('\nEvaluation on validation set:')
    for k, v in val_metrics.items():
        logger.warning("val %s: %g" % (k, v))
    
    logger.warning('\nEvaluation on test set:')
    for k, v in test_metrics.items():
        logger.warning("test %s: %g" % (k, v))
    

if __name__ == "__main__":
    args = parse_args()
    main(args)

