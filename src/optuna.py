import optuna
import torch
import torch.nn as nn
from tqdm import tqdm
from src.train import eval, train

def objective(
    trial,
    model,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    epochs=5,
    metric=None, 
    valid_loader=None
):

    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.3)

    #1. TODO: model = Transformer(dropout=dropout)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):

        #TODO: verificar train_epoch
        train_epoch = train(
            model=model,
            epochs=epochs,
            data_loader=data_loader,
            criterion=criterion,
            optimizer=optimizer,
            metric=metric,
            valid_loader=valid_loader
        )

        val_score = eval(
            model=model,
            metric=metric,
            valid_loader=valid_loader
        )

        trial.report(val_score, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return val_score

#2. TODO: adicionar os dados
#train_loader = 
#valid_loader =

study = optuna.create_study(
    direction='minimiza'
)

# 3. TODO: habilitar functools
# func_otimizacao = functools.partial(
#    objective, 
#    data_loader=data_loader,
#    valid_loader=valid_loader,
#    epochs=5, 
# )

#study.optimize(func_otimizacao, n_trials=10)

# print("BEST PARAMS:", study.best_params)
    




            