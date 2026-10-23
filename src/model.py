import torch
import torch.nn as nn
import math
from src.functional import softmax

#1. Embedding
class Embedding(nn.Module):
    def __init__(
            self,
            num_embeddings,
            embedding_dim,
            device
        ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings, #numero de embeddings -> mesmo tamanho do vocabulário ( com n dimensões CADA)
            embedding_dim,
            device=device
        )
        # matriz weight foi inicializada com valores iniciais aleatórios ( sem significado semântico útil ) -> ajustados no treinamento 
        
        self.embedding_dim = embedding_dim

    def forward(self, idxs):
        return self.embedding(idxs) * math.sqrt(self.embedding_dim) # o paper "Attention is All You Need" faz a implementação multiplicando pela raiz de d_model.]
        #retorna uma matriz [B, L, D]

#2. Positional Embedding
class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model, device, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        #criando matriz das posições
        pos_matrix = (torch.arange(
            0, max_len,
            device=device # [L]
        ).unsqueeze(1)) #[L, D]

        #criando matriz pe
        pe = torch.zeros(
                    max_len,
                    d_model,
                    device=device
                ) # [L, D]
 
        #termo de divisão
        div_term = (
            10000.0 ** (
                torch.arange(0, d_model, 2) # 2i 
                /d_model
            )
        )

        #aplicando seno aos pares
        pe[:, 0::2] = torch.sin(pos_matrix / div_term)

        #aplicando cosseno nos ímpares
        pe[:, 1::2] = torch.cos(pos_matrix / div_term)

        #para deixar no padrão esperado pelo embedding de [B, L, D]
        pe = pe.unsqueeze(0)

        #levar self.pe para buffer, pois ele não será  treinado. 
        self.register_buffer("pe", pe)

    def forward(self, x): # esse x represneta a matriz de embeddings
        L = x.shape[1]
        return self.dropout(x + self.pe[:,:L,:])


#3. Scaled Dot Procut ( Attention )
def scaled_dot_product_attention(
    q: torch.Tensor,
    v: torch.Tensor,
    k: torch.Tensor,
    attn_mask=None,
) -> torch.Tensor:

    d_k = k.size(-1) # [B, H, Lk, dk]
    
    attn_scores = (
        torch.matmul(
            q,
            k.transpose(-2, -1)
        ) / math.sqrt(d_k)
    ) # [B, H, Lq, Lk]

    if attn_mask is not None:
        attn_scores = attn_scores.masked_fill(
            attn_mask == 0,
            float('-inf')
        )

    prob_attn = softmax(
        attn_scores,
        dim=-1
    )

    attn_output = torch.matmul(
        prob_attn, v
    )

    return attn_output, prob_attn 


#4. FFN
class PositionWiseFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout,
        device: torch.device
    ):
        
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(
                d_model,
                d_ff,
                device=device
            ),
            nn.ReLU(),
            nn.Linear(
                d_ff,
                d_model,
                device=device
            )
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.model(x))


#5. MultiHead Attention
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads:int,
        device: torch.device,
        dropout=0.1
    ):
        #definindo o numero de dimensões de cada head
        super().__init__()

        assert d_model % num_heads == 0, "d_model deve ser divisível por num_heads"

        self.d_head = d_model // num_heads
        self.num_heads = num_heads

        #fazendo as camadas lineares
        self.q_layer = nn.Linear(
            d_model,
            d_model,
            device=device
        )

        self.k_layer = nn.Linear(
            d_model,
            d_model,
            device=device
        )

        self.v_layer = nn.Linear(
            d_model,
            d_model,
            device=device
        )

        self.o_layer = nn.Linear(
            d_model,
            d_model,
            device=device
        )

        self.dropout = nn.Dropout(dropout)

    def _splitting(self, x):
        return x.view(x.size(0), x.size(1), self.num_heads, self.d_head).transpose(1,2)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask=None
    ):

        #criando as camadas lineares
        query = self._splitting(self.q_layer(query))
        key = self._splitting(self.k_layer(key))
        value = self._splitting(self.v_layer(value))

        #aplicando o attention
        attn_output, prob_attn = scaled_dot_product_attention(
            q=query,
            k=key,
            v=value,
            attn_mask=attn_mask
        )

        #gerando o shape correto final (juntado os heads)
        #.contiguous() reorganiza, se necessário, os dados na memória. Pois o transpose não necessariamente reorganiza os dados na memória.
        x = attn_output.transpose(1,2).contiguous() #[B, L, h, d]
       
        x = x.view(
            x.size(0),
            x.size(1),
            self.num_heads * self.d_head
        ) #[B, L, h * d]

        return self.dropout(self.o_layer(x)) # [B, L, d_model]

#6. Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            num_heads: int,
            device: torch.device,
            dropout # padrão do paper 2048
        ):

        super().__init__()

        self.mha = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            device=device,
            dropout=dropout
        )

        self.ffn = PositionWiseFFN(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dropout=dropout
        )

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(
                d_model,
                device=device
            ),
            nn.LayerNorm(
                d_model,
                device=device
            )
        ])

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None
    ):
        
        mha_output = self.mha(x, x, x, attn_mask=attn_mask)
        x = self.layer_norms[0](x + mha_output)

        ffn_output = self.ffn(x)
        x = self.layer_norms[1](x + ffn_output)

        return x



#7. Encoder
class Encoder(nn.Module):
    def __init__(
        self, 
        d_model:int,
        d_ff:int,
        num_heads:int,
        device: torch.device,
        num_layers=6,
        dropout=0.1
    ):
        
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                d_ff=d_ff,
                num_heads=num_heads,
                device=device,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None
    ):

        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)

        return x

#8. Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            num_heads: int,
            device: torch.device,
            dropout: float # padrão do paper 2048
        ):

        super().__init__()

        self.masked_mha = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            device=device,
            dropout=dropout
        )

        self.mha = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            device=device,
            dropout=dropout
        )

        self.ffn = PositionWiseFFN(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dropout=dropout
        )

        num_layer_norms = 3
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(
                d_model,
                device=device
            ) 
            for _ in range(num_layer_norms)
        ])


    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor | None = None, # padding mask
        causal_mask: torch.Tensor | None = None, # mascara causal ou combinação de mascaras
        tgt_padding: torch.Tensor | None = None

    ):
        

        #masked attention
        masked_mha_output = self.masked_mha(
            x, x, x,
            attn_mask=causal_mask & tgt_padding
        )
        x = self.layer_norms[0](x + masked_mha_output)

        #cross attention
        '''
            x: [Batch, Target Lenght, d_model]      -> entrada do decoder
            memory: [Batch, Source Lenght, d_model] -> output do encoder
        '''
        mha_output = self.mha(
            x, memory, memory,
            attn_mask=src_mask
        ) #Q vem do decoder -> faz a requisição / K e V vem do encoder
        x = self.layer_norms[1](x + mha_output)

        #feed forward
        ffn_output = self.ffn(x)
        x = self.layer_norms[2](x + ffn_output)
        return x


#9. Decoder
class Decoder(nn.Module):
    def __init__(
        self, 
        d_model:int,
        d_ff:int,
        num_heads:int,
        device: torch.device,
        num_layers=6,
        dropout=0.1
    ):
        
        super().__init__()

        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                d_ff=d_ff,
                num_heads=num_heads,
                device=device,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
        tgt_padding: torch.Tensor | None = None,
    ):

        for layer in self.layers:
            x = layer(
                x, memory,
                src_mask=src_mask,
                causal_mask=causal_mask,
                tgt_padding=tgt_padding
            )

        return x

#10. Transformer 

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        max_len,
        device: torch.device,
        d_ff = 2048,
        d_model = 512,
        num_heads = 8,
        num_encoder_layers = 6,
        num_decoder_layers = 6,
        dropout=0.1
    ):
    
        super().__init__()

        self.max_len = max_len
        self.device = device

        self.src_embedding = Embedding(
            src_vocab_size,
            d_model,
            device=device
        )

        self.tgt_embedding = Embedding(
            tgt_vocab_size,
            d_model,
            device=device
        )

        self.pos_emb = PositionalEncoding(
            max_len=self.max_len,
            d_model=d_model,
            device=device
        )

        self.encoder = Encoder(
            d_model=d_model,
            d_ff=d_ff,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            device=device,
            dropout=dropout
        )

        self.decoder = Decoder(
            d_model=d_model,
            d_ff=d_ff,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            device=device,
            dropout=dropout      
        )

        self.output = nn.Linear(
            in_features=d_model,
            out_features=tgt_vocab_size,
            device=device
        )

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        pad_id: int,
        device: torch.device,
        *,
        output_prob=False
    ):

        #mascara causal
        tgt_len = tgt.size(1)
        causal_mask = torch.tril(
            torch.ones(
                tgt_len,
                tgt_len,
                dtype=torch.bool,
                device=device
            )
        )

        #src padding mask
        src_pad_mask = (src != pad_id).unsqueeze(1).unsqueeze(2).to(device) #shape para Attention

        #tgt padding mask
        tgt_pad_mask = (tgt != pad_id).unsqueeze(1).unsqueeze(2).to(device) #shape para Attention       
        
        #encoder input
        src_pos_embed = self.pos_emb(self.src_embedding(src))
        memory = self.encoder(
            src_pos_embed,
            attn_mask=src_pad_mask
        )

        #decoder input
        tgt_pos_embed = self.pos_emb(self.tgt_embedding(tgt))

        output = self.decoder(
            x=tgt_pos_embed,
            memory=memory,
            src_mask=src_pad_mask,
            causal_mask=causal_mask,
            tgt_padding=tgt_pad_mask
        )

        #nesse ponto armazena os logits
        output = self.output(
            output
        )

        if output_prob:
            output = softmax(
                output,
                dim=-1
            )

        return output

        
