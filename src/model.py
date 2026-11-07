import torch
import torch.nn as nn
import math
from src.functional import softmax

#1. Embedding
class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        #criando tensor inicial vazio com shape [num_embeddings, embedding_dim]
        weights = torch.empty(
            num_embeddings,
            embedding_dim,
            device=device
        )

        #transformando esse tensor em parâmetros treinaveis
        self.weights = nn.Parameter(weights, requires_grad=True)

        #inicializando o tensor com uma distribuiçãoo uniforme
        #com valores entre -0.1 e 0.1
        nn.init.uniform_(self.weights, a=-0.1, b=0.1)

    def forward(
        self,
        idxs: torch.Tensor,
    ):
        #criando tensor dos embeddings correspondentes aos ids fornecidos
        embeddings_from_ids = self.weights[idxs]

        #escalando os embeddings por sqrt(d_model) seguindo recomendação do paper
        scaled_embeddings = embeddings_from_ids * math.sqrt(self.embedding_dim) 

        return scaled_embeddings #[batch, n_embeddings, embedding_dim]


#2. Positional Embedding
class PositionalEncoding(nn.Module):
    def __init__(
        self,
        max_len: int,
        d_model: int,
        device: torch.device | None = None,
        dropout: float = 0.1,
    ):
        
        super().__init__()
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)

    # -----------------------------------------------

        pos_matrix = (
            torch.arange(
                0,
                max_len
            ).unsqueeze(1) #[L, 1]
        ).to(device) 

        # matriz que deve receber apenas o número das posições que vamos códificar
        # esse arrange cria um tensor unididmensional que parte do elemento 0 até max_len - 1
        # .unsqueeze(1) transforma o tensor de [L] para [L, 1]
        #
        # Exemplo para max_len = 6:
        # Output: tensor([[0],
        #                 [1],
        #                 [2],
        #                 [3],
        #                 [4],
        #                 [5]])

    # ----------------------------------------------

        #criando matriz que vai receber os positional encodings
        pe = torch.zeros(
                    max_len,
                    d_model
                ).to(device) # [L, D]

        # linha  -> posição na sequencia
        # coluna -> dimensão do encoding
 
    # ----------------------------------------------

        # 2i / d_model
        exp_div_term = (
                torch.arange(0, d_model, 2) # 2i 
                /d_model
                
            ).to(device)

        #10000^(2i/d_model)
        div_term = (
            10000.0 ** exp_div_term
        ).to(device)

        # Exemplo de div_term para d_model = 6:
        # Output: tensor([1.0000e+00, 4.6416e+02, 2.1544e+05])
        # esses valores que alteram o denominador a seguir e geram períodos diferentes
    
    # ------------------------------------------------

        
        pe[:, 0::2] = torch.sin(pos_matrix / div_term) #aplicando seno aos pares        
        pe[:, 1::2] = torch.cos(pos_matrix / div_term) #aplicando cosseno nos ímpares

        pe = pe.unsqueeze(0) #padrão esperado pelo embedding de [B, L, D]

        #levar self.pe para buffer, pois ele não será  treinado. 
        self.register_buffer(
            "pe", pe
        )

    def forward(self, x): # esse x represneta a matriz de embeddings
        _, seq_len, _ = x.shape

        if seq_len > self.max_len:
            raise ValueError(f"tamanho da sequencia ({seq_len}) maior do que o maximo permitido: {self.max_len}")

        #somando informação posicional a todos os tokens
        x = x + self.pe[:, :seq_len, :] 
        # [:, :seq_len, :] -> [sobre todos os batches, até o tamanho da sequencia de input, sobre todas as dimensões]
        
        return self.dropout(x)




#3. Scaled Dot Procut ( Attention )
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    device: torch.device,
    is_causal: bool = False,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:

    d_k = key.size(-1) # [B, H, Lk, dk]
    scale_factor = math.sqrt(d_k)

# --------------------------------

    # Shapes de entrada
    # Q = [B, H, Lq, d_k]
    # K = [B, H, Lk, d_k]
    
    attn_scores = (
        torch.matmul(
            query,                # Q   -> [B, H, Lq, d_k]
            key.transpose(-2, -1) # K^T -> [B, H, d_k, Lk]
        ) 
        / scale_factor            # sqrt(d_k)
    ) 

    # Shape de Output do Attention Score Matrix:
    # [B, H, Lq, Lk]

    # No anexo abaixo tem uma implementação mais limpa utilizando einsum.

# -------------------------------

    if is_causal:
        Lq = query.size(-2) # comprimento da sequencia de queries
        Lk = key.size(-2)   # comprimento da sequencia de keys

        #criando uma matriz triangular inferior booleana -> 1 = True | 0 = False
        causal_mask = torch.tril(
            torch.ones(
                Lq,
                Lk,
                device=device,
                dtype=torch.bool
            )
        )
        attn_scores = attn_scores.masked_fill(
            causal_mask.logical_not(), float('-inf') #(-inf) substitui False
        )

    if attn_mask is not None:
        # aplicando outras mascaras como as de padding
        attn_scores = attn_scores.masked_fill(
            attn_mask == 0, float('-inf') #False -> (-inf) 
        )

    prob_attn = softmax(
        attn_scores,
        dim=-1
    )

    return torch.matmul(prob_attn, value) 


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
        self.device = device

        if d_model % num_heads != 0:
            raise ValueError("d_model deve ser divisível por num_heads")

        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        #fazendo as camadas lineares
        self.q_proj = nn.Linear(
            d_model,
            d_model,
            device=device
        )

        self.k_proj = nn.Linear(
            d_model,
            d_model,
            device=device
        )

        self.v_proj = nn.Linear(
            d_model,
            d_model,
            device=device
        )
 
        self.o_proj = nn.Linear(
            d_model,
            d_model,
            device=device
        )


    def _split_heads(self, x): # [B, L, d]
        batch_size, seq_len, _ = x.shape

        x = x.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ) # [B, L, h, d]

        return x.transpose(1,2) # [B, h, L, d]

    def _merge_heads(self, x): # [B, h, L, d]
        batch_size, _  , seq_len,  _ = x.shape

        x = x.transpose(1,2).contiguous() #[B, L, h, d]

        x = x.view(
            batch_size,
            seq_len,
            self.num_heads * self.head_dim 
        )

        return x # [B, L, h * d]

    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool | None = None,
        attn_mask: torch.Tensor | None = None,
    ):

        #criando as camadas lineares e separando heads
        query = self._split_heads(self.q_proj(query))
        key = self._split_heads(self.k_proj(key))
        value = self._split_heads(self.v_proj(value))

        #aplicando o attention
        attn_output = scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            is_causal=is_causal,
            attn_mask=attn_mask,
            device=self.device
        )

        #concatenando heads
        x = self._merge_heads(attn_output) #[B, L, h * d]

        #projetando a camada de output
        x = self.o_proj(x)

        return self.dropout(x) # [B, L, d_model]

#6. Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,        
            d_ff: int,           
            num_heads: int,     
            device: torch.device,
            dropout: float = 0.1 
        ):

        super().__init__()
        self.device = device
        
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

        #layer norm
        self.norm_mha = nn.LayerNorm(
            d_model,
            device=device
        )
        
        self.norm_ffn = nn.LayerNorm(
            d_model,
            device=device    
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None
    ):

        #Multi-Head Attention + Add & Norm
        mha_output = self.mha(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask
        )
        x = self.norm_mha(x + mha_output)

        #Position Wise FFN + Add & Norm
        ffn_output = self.ffn(x)
        x = self.norm_ffn(x + ffn_output)

        return x


#7. Encoder
class Encoder(nn.Module):
    def __init__(
        self, 
        d_model:int,         
        d_ff:int,            
        num_heads:int,      
        num_layers: int = 6,
        dropout: float = 0.1, 
        device: torch.device | None = None,
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
            x = layer(
                x,
                attn_mask=attn_mask
            )

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

        #layer norms
        self.norm_masked_mha = nn.LayerNorm(
            d_model,
            device=device
        )

        self.norm_mha = nn.LayerNorm(
            d_model,
            device=device
        )
                
        self.norm_ffn = nn.LayerNorm(
            d_model,
            device=device    
        )


    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        is_causal: bool = True,
        src_mask: torch.Tensor | None = None, # padding mask
        tgt_padding: torch.Tensor | None = None

    ):
        

        #masked attention
        masked_mha_output = self.masked_mha(
            x, x, x,
            is_causal=is_causal,
            attn_mask=tgt_padding
        )
        x = self.norm_masked_mha(x + masked_mha_output)

        #cross attention
        '''
            x: [Batch, Target Lenght, d_model]      -> entrada do decoder
            memory: [Batch, Source Lenght, d_model] -> output do encoder
        '''
        mha_output = self.mha(
            x, memory, memory,
            attn_mask=src_mask
        ) #Q vem do decoder -> faz a requisição / K e V vem do encoder
        x = self.norm_mha(x + mha_output)

        #feed forward
        ffn_output = self.ffn(x)
        x = self.norm_ffn(x + ffn_output)
        return x


#9. Decoder
class Decoder(nn.Module):
    def __init__(
        self, 
        d_model:int,
        d_ff:int,
        num_heads:int,
        num_layers: int = 6,
        device: torch.device | None = None,
        dropout: float = 0.1
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
        is_causal: bool = True,
        src_mask: torch.Tensor | None = None,
        tgt_padding: torch.Tensor | None = None
    ):

        for layer in self.layers:
            x = layer(
                x, memory,
                src_mask=src_mask,
                is_causal=is_causal,
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
        d_ff: int = 2048,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        device: torch.device | None = None,
        dropout: float = 0.1
    ):
    
        super().__init__()
        self.max_len = max_len

        self.src_embedding = Embedding(
            num_embeddings=src_vocab_size,
            embedding_dim=d_model,
            device=device
        )

        self.tgt_embedding = Embedding(
            num_embeddings=tgt_vocab_size,
            embedding_dim=d_model,
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
        is_causal: bool | None = True,
        *,
        output_prob=False
    ):

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
            is_causal=is_causal,
            src_mask=src_pad_mask,
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

        

        
