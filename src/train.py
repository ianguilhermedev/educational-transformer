import torch
import torch.nn as nn
from tqdm import tqdm
import tokenizers

tokenizer = tokenizers.Tokenizer.from_file(
    path = '../data/tokenizer/en-pt-cased-pont.json'
)

#avaliação utilizando o torchmetris
def eval(
    model,
    metric,
    valid_loader,
    device,
    pad_id=0
):
    with torch.no_grad():

        for input, target in tqdm(valid_loader,
                                     desc="VAL"):
            src_ids = input.src_ids.to(device) # [ B , L , Vocab(dim)]

            #pegando id do <bos>
            bos_id = tokenizer.token_to_id("<bos>")

            #criando tensor contando <bos> para cada linha ( frase do batch)
            tgt_ids = torch.full(
                size=(src_ids.size(0), 1),
                fill_value=(bos_id),
                device=device
            )

            
            finished = torch.zeros(
                src_ids.size(0),
                dtype=torch.bool,
                device=device
            )
            max_len = model.max_len
            d_pred = 1
            eos_id = tokenizer.token_to_id("<eos>")

            while (not finished.all()) and (d_pred < max_len):

                logits = model(
                    src=src_ids,
                    tgt=tgt_ids,
                    pad_id=pad_id,
                    device=device
                )

                preds = torch.argmax(logits[: , -1, :], dim=-1)

                #verificando se chegou ao final
                reached_eos = (preds == eos_id)
                
                #criando mascara para frases já finalizadas
                preds = preds.masked_fill(finished, pad_id)

                finished = (finished | reached_eos) # ou ja esta finalizado ou finalizou agora

                #adicionando ao target mais um token
                tgt_ids = torch.cat([tgt_ids, preds.unsqueeze(1)], dim=1)
                #incrementando dimensões da previsão
                d_pred += 1

            #pegando o id do maior logit e fazendo decoding, ou seja, formando a frase principal.
            predict_text = tokenizer.decode_batch(tgt_ids.tolist())
            tgt_text = [[tokenizer.decode(tgt.tolist())] for tgt in target]

            metric.update(
                #n dimensões do tgt depende do vocabulario
                #cada proxima previsão gera um logit
                predict_text, 
                tgt_text
            )
    return metric.compute()

#trainando o modelo
def train(
    model,
    epochs: int,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device,
    metric=None, 
    valid_loader=None,
    schedule=None,
    warmup_schedule=None,
    pad_id=0
):

    best_eval = torch.inf  
    patience_limit = 3
    patience_counter = 0

    for epoch in range(epochs):

        model.train()

        #TODO: corrigir warmup e scheduler
        if warmup_schedule is not None:
            warmup_schedule.step()

        progress = tqdm(
            data_loader, desc=f'EPOCH: {epoch+1}/{epochs}'
        )

        for input, target in progress:
            src_ids = input.src_ids.to(device)
            tgt_ids = input.tgt_ids.to(device)

            logits = model(
                src=src_ids,
                tgt=tgt_ids,
                pad_id=pad_id,
                device=device
            )

            #TODO: comparando logits com ids
            loss = criterion(
                logits.reshape(-1, logits.shape[-1]),
                target.reshape(-1).to(device)
            )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            optimizer.step()

            optimizer.zero_grad()

            progress.set_postfix(
                loss=f'{loss.item():.2f}'
            )


        model.eval()
        val_score = eval(
            model=model, 
            metric=metric, 
            valid_loader=valid_loader,
            pad_id=pad_id,
            device=device
        )   
        metric.reset()

        if schedule is not None and epoch >= 3: #TODO: remover epocas de warmup hardcoded
            schedule.step(val_score.item())

        print(f'EVAL: {val_score}')

        # Early Stopping
        if val_score > best_eval:
            patience_counter = 0
            best_eval = val_score
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f'EARLY STOPPING! Without improve in the last {patience_limit} epochs.')
                break


