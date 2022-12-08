import logging

# logging utils

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

# (supervised) training

def train_epoch(model, 
                train_loader, 
                optimizer, 
                loss_fn, 
                batch_size, 
                logger, 
                device):
    model.train()
    train_size = len(train_loader.dataset)
    total_sample = total_loss = 0
    all_y = torch.tensor([])
    all_preds = torch.tensor([])
    for i, (X, y) in enumerate(train_loader):
        all_y = torch.cat([all_y, y])
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        preds = model(X)
        loss = loss_fn(preds, y.float())
        
        loss.backward()
        optimizer.step()

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