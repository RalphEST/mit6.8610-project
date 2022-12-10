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

def train_epoch(model, train_loader, optimizer, loss_fn, log_every=10):
    model.train()
    total_loss = 0
    
    all_labels, all_preds = [],[]
    for i, batch in enumerate(train_loader):
        # move batch dictionary to device
        data_utils.batch_dict_to_device(batch, device)
        labels, features = batch['labels'], batch['seq-var-matrix']
        
        # compute prediction and loss
        preds = model(features)
        loss = loss_fn(preds, labels)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # tracking
        total_loss += loss.item()
        all_labels.append(labels.cpu())
        all_preds.append(preds.flatten().detach().cpu())
        
        # logging
        if (i % log_every == 0):
            print(f"\tBatch {i} | BCE Loss: {loss.item():.4f}")
    
    metrics = eval_utils.get_metrics(torch.cat(all_labels), 
                                     torch.cat(all_preds))
    metrics['loss'] = total_loss
    
    return metrics

def test(model, test_loader, loss_fn):
    model.eval()
    total_loss = 0
    all_labels, all_preds = [],[]
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader)):
            # move batch dictionary to device
            data_utils.batch_dict_to_device(batch, device)
            labels, features = batch['labels'], batch['seq-var-matrix']

            # compute prediction and loss
            preds = model(features)
            loss = loss_fn(preds, labels)

            # tracking
            total_loss += loss.item()
            
            all_labels.append(labels.cpu())
            all_preds.append(preds.flatten().detach().cpu())
    
    metrics = eval_utils.get_metrics(torch.cat(all_labels), 
                                     torch.cat(all_preds))
    metrics['loss'] = total_loss
    
    return metrics

def train(model, 
          train_dataset,
          test_dataset, 
          lr=1e-3, 
          n_epochs=10,
          batch_size=256):
    
    train_loader = DataLoader(
        dataset = train_dataset, 
        batch_size = batch_size,
        sampler = WeightedRandomSampler(train_dataset.weights('131338-0.0'), 
                                        num_samples = len(train_dataset)),
        num_workers=12
    )
    
    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = batch_size,
        num_workers=12
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    track_metrics = {'train':{i:None for i in range(n_epochs)}, 
                     'test': {i:None for i in range(n_epochs)}}
    
    for epoch in range(n_epochs):
        print(f"Epoch #{epoch}:")
        train_metrics = train_epoch(model, 
                                    train_loader, 
                                    optimizer, 
                                    loss_fn, 
                                    log_every=10)
        print("Train metrics:")
        eval_utils.print_metrics(train_metrics)
        test_metrics = test(model, 
                            test_loader, 
                            loss_fn)
        print("Test metrics:")
        eval_utils.print_metrics(test_metrics)
        
        track_metrics['train'][epoch] = train_metrics
        track_metrics['test'][epoch] = test_metrics
    
    return track_metrics

