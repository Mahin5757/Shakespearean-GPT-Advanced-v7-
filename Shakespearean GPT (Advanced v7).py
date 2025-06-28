import torch
import torch.nn as nn
from torch.nn import functional as F
import requests
import os
import time
import math
from dataclasses import dataclass, field, fields
from typing import Optional, Tuple, List, Dict, Union

from torch.utils.checkpoint import checkpoint as gradient_checkpoint_fn

try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
    from tokenizers.normalizers import NFKC, Sequence
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.trainers import BpeTrainer
except ImportError:
    print("The 'tokenizers' library by Hugging Face is not installed.")
    print("Please install it by running: pip install tokenizers")
    print("This script cannot run without it for BPE tokenization.")
    exit(1)

# --- Design Philosophy & Goals ---
# This model aims to be an improvement over simpler GPT architectures (like a basic nanoGPT)
# by incorporating modern techniques for better performance and parameter efficiency,
# with a focus on generating high-quality Shakespearean English, understanding modern inputs,
# and exhibiting robust stylistic control.

# --- Core Dataset Configuration ---
DATA_URL = 'https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt'
# USER ACTION: Ensure this DATA_PATH points to your BEST CLEANED Shakespeare dataset.
DATA_PATH = 'shakespeare_cleaned.txt'
# USER ACTION: Use a new tokenizer path if you've cleaned data and want to retrain the tokenizer.
TOKENIZER_JSON_PATH = 'bpe_shakespeare_cleaned_larger_model_tokenizer.json'

# --- Model Hyperparameters (LARGER capacity model) ---
DEFAULT_VOCAB_SIZE_BPE = 8000
MIN_FREQUENCY_BPE = 2
VAL_SPLIT = 0.1
DEFAULT_BLOCK_SIZE = 384
DEFAULT_N_EMBD = 512      # Embedding dimension
DEFAULT_N_LAYER = 8        # Number of transformer layers
DEFAULT_N_HEAD = 8         # Number of query heads for attention (512 % 8 == 0)
DEFAULT_NUM_KV_HEADS = 2   # GQA setting
DEFAULT_DROPOUT = 0.10
DEFAULT_ROPE_THETA = 10000.0
RMSNORM_EPS = 1e-5
DEFAULT_FFN_DIM_MULTIPLE_OF = 256

# --- Training Hyperparameters (Adjusted for fine-tuning the LARGER model) ---
BATCH_SIZE = 16
GRAD_ACCUMULATION_STEPS = 12
MAX_LEARNING_RATE = 3e-5 # Reduced for fine-tuning from the 1e-4 initial run
MIN_LEARNING_RATE = MAX_LEARNING_RATE / 10
ADAMW_BETA1 = 0.9
ADAMW_BETA2 = 0.95
MAX_ITERS = 15000 # Target total iterations (can be increased if time allows and loss improves)
WARMUP_ITERS = 200 # Shorter warmup for fine-tuning from a checkpoint
LR_DECAY_ITERS = MAX_ITERS
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.1
LABEL_SMOOTHING_EPSILON = 0.1

# --- Evaluation and Device Settings ---
EVAL_INTERVAL = 250
EVAL_ITERS = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
AMP_ENABLED = True if DEVICE == 'cuda' else False

# --- Compilation and Checkpointing ---
TORCH_COMPILE_ENABLED = False # Disabled to ensure smooth run without Triton errors.
GRADIENT_CHECKPOINTING_ENABLED = True

# USER ACTION: Keep the same model save path to continue fine-tuning the larger model.
MODEL_SAVE_PATH = 'shakespeare_gpt_bpe_advanced_v7_larger_model_final.pth'

# --- Generation Parameters ---
GENERATION_MAX_NEW_TOKENS = 200
GENERATION_TEMPERATURE = 0.7
GENERATION_TOP_K = 50
GENERATION_TOP_P = 0.92
GENERATION_REPETITION_PENALTY = 1.15
GENERATION_MIROSTAT_MODE = 0
GENERATION_MIROSTAT_TAU = 5.0
GENERATION_MIROSTAT_ETA = 0.1

# --- Chatbot Specific ---
CHATBOT_HISTORY_LENGTH = 3

# --- GPT Configuration Dataclass ---
@dataclass
class GPTConfig:
    block_size: int = DEFAULT_BLOCK_SIZE
    vocab_size: int = DEFAULT_VOCAB_SIZE_BPE
    n_layer: int = DEFAULT_N_LAYER
    n_head: int = DEFAULT_N_HEAD
    num_kv_heads: Optional[int] = DEFAULT_NUM_KV_HEADS
    n_embd: int = DEFAULT_N_EMBD
    dropout: float = DEFAULT_DROPOUT
    bias_enabled: bool = False
    rope_theta: float = DEFAULT_ROPE_THETA
    rmsnorm_eps: float = RMSNORM_EPS
    ffn_hidden_multiplier: float = 8/3
    ffn_dim_multiple_of: int = DEFAULT_FFN_DIM_MULTIPLE_OF

    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.n_head

        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head."
        assert self.n_head % self.num_kv_heads == 0, \
            "n_head must be divisible by num_kv_heads for Grouped Query Attention (GQA/MQA)."

# --- Data Handling ---
def download_full_shakespeare_dataset():
    if not os.path.exists(DATA_PATH):
        print(f"Downloading Shakespeare dataset from {DATA_URL}...")
        try:
            response = requests.get(DATA_URL)
            response.raise_for_status()
            text_content = response.content.decode('utf-8', errors='replace')
            with open(DATA_PATH, 'w', encoding='utf-8') as f:
                f.write(text_content)
            print(f"Shakespeare dataset saved to {DATA_PATH}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading Shakespeare dataset: {e}")
            exit(1)
        except UnicodeDecodeError as e:
            print(f"Error decoding Shakespeare dataset (tried UTF-8): {e}.")
            exit(1)
    else:
        print(f"Shakespeare dataset {DATA_PATH} already exists.")

def load_data_raw():
    with open(DATA_PATH, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    print(f"Loaded raw Shakespeare dataset with {len(text)} characters.")
    return text

# --- BPE Tokenizer Wrapper ---
class BPETokenizerWrapper:
    def __init__(self, tokenizer_json_path=TOKENIZER_JSON_PATH, train_data_path=None,
                 vocab_size=DEFAULT_VOCAB_SIZE_BPE, min_frequency=MIN_FREQUENCY_BPE):
        self.tokenizer_json_path = tokenizer_json_path
        self.tokenizer = None
        self._vocab_size = 0
        
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        
        self.user_token = "[USER]"
        self.bard_token = "[BARD]"
        self.style_sonnet_token = "[STYLE:SONNET]"
        self.style_play_token = "[STYLE:PLAY]"

        self.special_tokens_list = sorted(list(set([
            self.bos_token, self.eos_token, self.pad_token, self.unk_token,
            self.user_token, self.bard_token, self.style_sonnet_token, self.style_play_token
        ])))

        if os.path.exists(self.tokenizer_json_path):
            print(f"Loading BPE tokenizer from {self.tokenizer_json_path}...")
            self.tokenizer = Tokenizer.from_file(self.tokenizer_json_path)
            print("BPE Tokenizer loaded.")
        elif train_data_path and os.path.exists(train_data_path):
            print(f"Training BPE tokenizer from {train_data_path}...")
            bpe_model = BPE(unk_token=self.unk_token)
            self.tokenizer = Tokenizer(bpe_model)
            self.tokenizer.normalizer = Sequence([NFKC()])
            self.tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=True)
            self.tokenizer.decoder = ByteLevelDecoder()
            
            trainer = BpeTrainer(
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                show_progress=True,
                special_tokens=self.special_tokens_list,
            )
            self.tokenizer.train(files=[train_data_path], trainer=trainer)
            
            self.save_tokenizer_model(self.tokenizer_json_path)
            print(f"BPE Tokenizer trained and saved to {self.tokenizer_json_path}")
        else:
            raise ValueError(
                f"Tokenizer file {self.tokenizer_json_path} not found AND "
                f"no valid train_data_path ({train_data_path}) provided for training."
            )
        self._vocab_size = self.tokenizer.get_vocab_size(with_added_tokens=True)
        print(f"BPE Tokenizer vocabulary size (incl. special tokens): {self._vocab_size}")

    def encode(self, text_or_batch: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        if isinstance(text_or_batch, str):
            return self.tokenizer.encode(text_or_batch).ids
        elif isinstance(text_or_batch, list):
            return [enc.ids for enc in self.tokenizer.encode_batch(text_or_batch)]
        else:
            raise TypeError("Input must be a string or a list of strings.")

    def decode(self, token_ids_or_batch: Union[List[int], List[List[int]], torch.Tensor], skip_special_tokens=True) -> Union[str, List[str]]:
        if isinstance(token_ids_or_batch, torch.Tensor):
            token_ids_or_batch = token_ids_or_batch.cpu().tolist()
        
        if not token_ids_or_batch:
            return "" if not (isinstance(token_ids_or_batch, list) and len(token_ids_or_batch) > 0 and isinstance(token_ids_or_batch[0], list)) else []

        is_likely_batched = isinstance(token_ids_or_batch, list) and \
                             len(token_ids_or_batch) > 0 and \
                             isinstance(token_ids_or_batch[0], list) and \
                             all(isinstance(item, int) for item in token_ids_or_batch[0])
        
        is_single_sequence_of_ints = isinstance(token_ids_or_batch, list) and \
                                      (not token_ids_or_batch or isinstance(token_ids_or_batch[0], int))

        if is_likely_batched:
             return self.tokenizer.decode_batch(token_ids_or_batch, skip_special_tokens=skip_special_tokens)
        elif is_single_sequence_of_ints:
            return self.tokenizer.decode(token_ids_or_batch, skip_special_tokens=skip_special_tokens)
        else:
            try:
                if isinstance(token_ids_or_batch, list) and len(token_ids_or_batch) == 1 and isinstance(token_ids_or_batch[0], list):
                    return self.tokenizer.decode(token_ids_or_batch[0], skip_special_tokens=skip_special_tokens)
                return self.tokenizer.decode(token_ids_or_batch, skip_special_tokens=skip_special_tokens)
            except Exception as e:
                print(f"Warning: Tokenizer decode encountered an issue with input format: {type(token_ids_or_batch)}. Error: {e}")
                return ""

    def save_tokenizer_model(self, path):
        dir_name = os.path.dirname(path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        self.tokenizer.save(path)

    @property
    def vocab_size(self):
        return self._vocab_size
    
    def token_to_id(self, token_str: str) -> Optional[int]:
        return self.tokenizer.token_to_id(token_str)
    
    def id_to_token(self, token_id: int) -> Optional[str]:
        return self.tokenizer.id_to_token(token_id)

# --- Rotary Positional Embedding (RoPE) ---
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis_broadcastable = freqs_cis.unsqueeze(0).unsqueeze(0)
    
    xq_rotated = xq_complex * freqs_cis_broadcastable
    xk_rotated = xk_complex * freqs_cis_broadcastable
    
    xq_out = torch.view_as_real(xq_rotated).flatten(3)
    xk_out = torch.view_as_real(xk_rotated).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# --- RMSNorm Layer ---
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# --- Model Components ---
class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.num_kv_heads = config.num_kv_heads
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim = config.n_embd // config.n_head
        self.bias_enabled = config.bias_enabled

        kv_embd_dim = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=self.bias_enabled)
        self.k_proj = nn.Linear(config.n_embd, kv_embd_dim, bias=self.bias_enabled)
        self.v_proj = nn.Linear(config.n_embd, kv_embd_dim, bias=self.bias_enabled)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=self.bias_enabled)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        B, T, C = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        current_seq_len = q.shape[2]
        q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis=freqs_cis[:current_seq_len])

        if self.num_kv_heads < self.n_head:
            num_repeats = self.n_head // self.num_kv_heads
            k_rot = k_rot.unsqueeze(2).repeat(1, 1, num_repeats, 1, 1).view(B, self.n_head, T, self.head_dim)
            v = v.unsqueeze(2).repeat(1, 1, num_repeats, 1, 1).view(B, self.n_head, T, self.head_dim)
        
        if hasattr(F, 'scaled_dot_product_attention') and (not self.training or self.dropout == 0.0):
              y = F.scaled_dot_product_attention(
                  q_rot, k_rot, v,
                  attn_mask=None,
                  dropout_p=0.0,
                  is_causal=True
              )
        else:
            att = (q_rot @ k_rot.transpose(-2, -1)) * (1.0 / math.sqrt(k_rot.size(-1)))
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            att = att.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            att = F.softmax(att, dim=-1)
            if self.training and self.dropout > 0.0:
                att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y))
        return y

class SwiGLUFFN(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        base_hidden_dim = int(config.ffn_hidden_multiplier * config.n_embd)
        multiple_of = config.ffn_dim_multiple_of
        hidden_dim = ((base_hidden_dim + multiple_of - 1) // multiple_of) * multiple_of
        self.bias_enabled = config.bias_enabled

        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=self.bias_enabled)
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=self.bias_enabled)
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=self.bias_enabled)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        swish_gate = F.silu(self.w1(x))
        gate_val = self.w3(x)
        hidden = swish_gate * gate_val
        output = self.w2(hidden)
        output = self.dropout(output)
        return output

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffwd = SwiGLUFFN(config)
        self.ln1 = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        self.ln2 = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        self.config = config
    
    def _block_forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x), freqs_cis)
        x = x + self.ffwd(self.ln2(x))
        return x
    
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        if GRADIENT_CHECKPOINTING_ENABLED and self.training:
            return gradient_checkpoint_fn(self._block_forward, x, freqs_cis, use_reentrant=False)
        else:
            return self._block_forward(x, freqs_cis)

class GPTLanguageModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.token_embedding_table.weight = self.lm_head.weight

        freqs_cis_data = precompute_freqs_cis(
            config.n_embd // config.n_head,
            config.block_size * 2,
            config.rope_theta
        )
        self.register_buffer("freqs_cis", freqs_cis_data, persistent=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight') or pn.endswith('w2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        assert T <= self.config.block_size, \
            f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        tok_emb = self.token_embedding_table(idx)
        
        if self.freqs_cis.device != idx.device:
            self.freqs_cis = self.freqs_cis.to(idx.device)
        
        current_freqs_cis = self.freqs_cis[:T]
        
        x = tok_emb
        for block in self.blocks:
            x = block(x, current_freqs_cis)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            logits_view = logits.view(B * T, self.config.vocab_size)
            targets_view = targets.view(B * T)
            if LABEL_SMOOTHING_EPSILON > 0.0 and self.training:
                loss = F.cross_entropy(logits_view, targets_view, label_smoothing=LABEL_SMOOTHING_EPSILON)
            else:
                loss = F.cross_entropy(logits_view, targets_view)
        
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: Optional[int] = None, top_p: Optional[float] = None,
                 repetition_penalty: float = 1.0,
                 mirostat_mode: int = 0, mirostat_tau: float = 5.0, mirostat_eta: float = 0.1,
                 eos_token_id: Optional[int] = None
                 ) -> torch.Tensor:
        self.eval()
        mu_mirostat = None
        
        if mirostat_mode == 2:
            mu_mirostat = torch.full((idx.size(0),), 2.0 * mirostat_tau, device=idx.device, dtype=torch.float32)
            if idx.size(0) > 1 and mirostat_mode > 0:
                print("Warning: Batched Mirostat 2.0 k-selection is simplified. For optimal batched Mirostat, k-selection should also be batched per item.")

        generated_tokens_history = [idx[i].tolist() for i in range(idx.size(0))]

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            
            if repetition_penalty != 1.0:
                for i in range(idx.size(0)):
                    if generated_tokens_history[i]:
                        for token_id_to_penalize in set(generated_tokens_history[i]):
                            if 0 <= token_id_to_penalize < logits.size(1):
                                if logits[i, token_id_to_penalize] > 0:
                                    logits[i, token_id_to_penalize] /= repetition_penalty
                                else:
                                    logits[i, token_id_to_penalize] *= repetition_penalty
            
            if mirostat_mode == 0:
                if temperature <= 0: temperature = 1.0
                if temperature != 1.0:
                    logits = logits / temperature
                
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                if top_p is not None and 0.0 < top_p < 1.0:
                    for i in range(logits.size(0)):
                        sorted_logits_item, sorted_indices_item = torch.sort(logits[i], descending=True)
                        cumulative_probs_item = torch.cumsum(F.softmax(sorted_logits_item, dim=-1), dim=-1)
                        sorted_indices_to_remove_item = cumulative_probs_item > top_p
                        sorted_indices_to_remove_item[..., 1:] = sorted_indices_to_remove_item[..., :-1].clone()
                        sorted_indices_to_remove_item[..., 0] = 0
                        indices_to_remove_item = sorted_indices_to_remove_item.scatter(0, sorted_indices_item, sorted_indices_to_remove_item)
                        logits[i, indices_to_remove_item] = -float('Inf')

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            
            elif mirostat_mode == 1 or mirostat_mode == 2:
                idx_next_list = []
                for i in range(idx.size(0)):
                    logits_item = logits[i]
                    probs_item_orig = F.softmax(logits_item, dim=-1)
                    sorted_probs_item, sorted_indices_item = torch.sort(probs_item_orig, descending=True)
                    
                    current_mu_item = mu_mirostat[i].item() if mirostat_mode == 2 and mu_mirostat is not None else 2.0 * mirostat_tau
                    
                    k = 0
                    if mirostat_mode == 1:
                        for prob_idx in range(len(sorted_probs_item)):
                            p = sorted_probs_item[prob_idx].item()
                            if p == 0: break
                            if -math.log(p) > mirostat_tau * 1.5 : break
                            k +=1
                        k = max(1,k); k = min(k, len(sorted_probs_item))
                    
                    elif mirostat_mode == 2:
                        for prob_idx in range(len(sorted_probs_item)):
                            p_val = sorted_probs_item[prob_idx].item()
                            if p_val == 0: break
                            surprise_val = -math.log(p_val)
                            if surprise_val <= current_mu_item:
                                k += 1
                            else:
                                break
                        k = max(1, k); k = min(k, len(sorted_probs_item))
                    
                    logits_miro_item = logits_item[sorted_indices_item[:k]]
                    probs_miro_item = F.softmax(logits_miro_item, dim=-1)
                    sampled_local_idx = torch.multinomial(probs_miro_item, num_samples=1)
                    actual_token_idx = sorted_indices_item[sampled_local_idx.item()]
                    idx_next_list.append(actual_token_idx.unsqueeze(0))
                    
                    if mirostat_mode == 2 and mu_mirostat is not None:
                        prob_of_chosen_token_item = probs_item_orig[actual_token_idx]
                        observed_surprise_item = -torch.log(prob_of_chosen_token_item + 1e-9)
                        error_item = observed_surprise_item - mirostat_tau
                        mu_mirostat[i] = mu_mirostat[i] - mirostat_eta * error_item
                
                idx_next = torch.cat(idx_next_list, dim=0).unsqueeze(1)

            idx = torch.cat((idx, idx_next), dim=1)
            for i in range(idx.size(0)):
                generated_tokens_history[i].append(idx_next[i, 0].item())
            
            if eos_token_id is not None and idx_next.item() == eos_token_id:
                break
            
        return idx

    def get_param_groups(self, weight_decay: float):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = []
        nodecay_params = []

        for pn, p in param_dict.items():
            if p.dim() >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        return optim_groups

# --- Training Setup ---
train_token_ids: Optional[torch.Tensor] = None
val_token_ids: Optional[torch.Tensor] = None
current_block_size_global: int = DEFAULT_BLOCK_SIZE
_torch_compile_actually_enabled = TORCH_COMPILE_ENABLED

def prepare_data_for_bpe(raw_text: str, tokenizer: BPETokenizerWrapper, val_split_ratio: float):
    global train_token_ids, val_token_ids
    print("Tokenizing entire dataset with BPE tokenizer...")
    all_token_ids = tokenizer.encode(raw_text)
    print(f"Dataset tokenized into {len(all_token_ids)} BPE tokens.")
    
    n = len(all_token_ids)
    split_idx = int(n * (1 - val_split_ratio))
    
    train_token_ids = torch.tensor(all_token_ids[:split_idx], dtype=torch.long)
    val_token_ids = torch.tensor(all_token_ids[split_idx:], dtype=torch.long)
    print(f"Training tokens: {len(train_token_ids):,}, Validation tokens: {len(val_token_ids):,}")

def get_batch(data_split: str) -> Tuple[torch.Tensor, torch.Tensor]:
    global current_block_size_global
    data = train_token_ids if data_split == 'train' else val_token_ids
    if data is None or len(data) < current_block_size_global + 1:
        raise ValueError(
            f"{data_split} data not prepared or too short. "
            f"Block size: {current_block_size_global}, Data len: {len(data) if data is not None else 0}"
        )
    
    ix = torch.randint(len(data) - current_block_size_global, (BATCH_SIZE,))
    x = torch.stack([data[i : i + current_block_size_global] for i in ix])
    y = torch.stack([data[i + 1 : i + current_block_size_global + 1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

@torch.no_grad()
def estimate_loss(model: GPTLanguageModel) -> dict:
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        perplexities = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            amp_dtype = torch.bfloat16 if DEVICE == 'cuda' and hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast(device_type=DEVICE, dtype=amp_dtype, enabled=AMP_ENABLED):
                _, loss = model(X, Y)
            losses[k] = loss.item()
            if loss.item() < 700:
                perplexities[k] = math.exp(loss.item())
            else:
                perplexities[k] = float('inf')
        out[split + '_loss'] = losses.mean()
        out[split + '_perplexity'] = perplexities.mean()
    model.train()
    return out

def get_lr(it: int) -> float:
    if it < WARMUP_ITERS:
        return MAX_LEARNING_RATE * it / WARMUP_ITERS
    if it > LR_DECAY_ITERS:
        return MIN_LEARNING_RATE
    
    decay_progress = it - WARMUP_ITERS
    total_decay_duration = LR_DECAY_ITERS - WARMUP_ITERS
    
    if total_decay_duration <= 0:
        return MIN_LEARNING_RATE
        
    decay_ratio = decay_progress / total_decay_duration
    decay_ratio = max(0.0, min(1.0, decay_ratio))
    
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LEARNING_RATE + coeff * (MAX_LEARNING_RATE - MIN_LEARNING_RATE)

# --- Training Loop ---
checkpoint_data_global = {}

def train_model(model: GPTLanguageModel, optimizer: torch.optim.Optimizer,
                tokenizer: BPETokenizerWrapper, scaler: torch.cuda.amp.GradScaler):
    global current_block_size_global, checkpoint_data_global, _torch_compile_actually_enabled
    current_block_size_global = model.config.block_size
    
    best_val_loss = checkpoint_data_global.get('best_val_loss', float('inf'))
    iter_num_at_load = checkpoint_data_global.get('iter_num', 0)
    iters_since_best_val_loss = checkpoint_data_global.get('iters_since_best_val_loss', 0)

    patience_epochs = 10
    
    print(f"Starting training on {DEVICE} for {MAX_ITERS} iterations (resuming from iter {iter_num_at_load})...")
    print(f"Effective Batch Size: {BATCH_SIZE * GRAD_ACCUMULATION_STEPS} (Micro Batch: {BATCH_SIZE}, Accum Steps: {GRAD_ACCUMULATION_STEPS})")
    if AMP_ENABLED: print("Automatic Mixed Precision (AMP) enabled.")
    
    if _torch_compile_actually_enabled and hasattr(model, '_orig_mod'):
        print("Torch.compile() is active on the model.")
    elif TORCH_COMPILE_ENABLED and not _torch_compile_actually_enabled:
        print("Torch.compile() was enabled but compilation failed or was skipped; model is running in eager mode.")
    elif not TORCH_COMPILE_ENABLED:
        print("Torch.compile() is disabled by global TORCH_COMPILE_ENABLED flag. Model will run in eager mode.")

    if GRADIENT_CHECKPOINTING_ENABLED: print("Gradient Checkpointing is enabled.")
    print(f"LR Scheduler: Warmup {WARMUP_ITERS} iters, Cosine Decay. Max LR: {MAX_LEARNING_RATE}, Min LR: {MIN_LEARNING_RATE}")
    print(f"Optimizer Betas: ({ADAMW_BETA1}, {ADAMW_BETA2}), Weight Decay: {WEIGHT_DECAY}")
    print(f"Model Config: {model.config}")
    print(f"Label Smoothing Epsilon: {LABEL_SMOOTHING_EPSILON if LABEL_SMOOTHING_EPSILON > 0 else 'Disabled'}")
    print(f"Eval Interval: {EVAL_INTERVAL} iters")
    print(f"Early stopping patience: {patience_epochs * EVAL_INTERVAL} iterations without validation loss improvement.")
    
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)
    
    for iter_num_offset in range(MAX_ITERS - iter_num_at_load + 1):
        iter_num = iter_num_at_load + iter_num_offset
        if iter_num > MAX_ITERS:
            break
        
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        model.train()
        current_micro_batch_loss_sum = 0.0
        
        for micro_step in range(GRAD_ACCUMULATION_STEPS):
            xb, yb = get_batch('train')
            amp_dtype = torch.bfloat16 if DEVICE == 'cuda' and hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast(device_type=DEVICE, dtype=amp_dtype, enabled=AMP_ENABLED):
                logits, loss = model(xb, yb)
                loss = loss / GRAD_ACCUMULATION_STEPS
            
            if micro_step == GRAD_ACCUMULATION_STEPS - 1:
                current_micro_batch_loss_sum = loss.item() * GRAD_ACCUMULATION_STEPS
            
            if AMP_ENABLED:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        
        total_norm = 0.0
        if GRAD_CLIP > 0:
            if AMP_ENABLED: scaler.unscale_(optimizer)
            params_to_clip = model._orig_mod.parameters() if hasattr(model, '_orig_mod') and _torch_compile_actually_enabled else model.parameters()
            total_norm = torch.nn.utils.clip_grad_norm_(params_to_clip, GRAD_CLIP).item()
        else:
            if AMP_ENABLED: scaler.unscale_(optimizer)
            params_to_check = model._orig_mod.parameters() if hasattr(model, '_orig_mod') and _torch_compile_actually_enabled else model.parameters()
            for p in params_to_check:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

        if AMP_ENABLED:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        optimizer.zero_grad(set_to_none=True)
        
        if iter_num % EVAL_INTERVAL == 0 or iter_num == MAX_ITERS:
            eval_metrics = estimate_loss(model)
            elapsed_time = time.time() - start_time
            print(
                f"Iter {iter_num}/{MAX_ITERS} | LR: {lr:.1e} | "
                f"Train Loss (approx): {current_micro_batch_loss_sum:.3f} | "
                f"Val Loss: {eval_metrics['val_loss']:.3f} (PPL: {eval_metrics['val_perplexity']:.2f}) | "
                f"Grad Norm: {total_norm:.3f} | Time: {elapsed_time:.1f}s"
            )
            
            current_val_loss = eval_metrics['val_loss']
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                iters_since_best_val_loss = 0
                model_to_save = model._orig_mod if hasattr(model, '_orig_mod') and _torch_compile_actually_enabled else model
                torch.save({
                    'model_state_dict': model_to_save.state_dict(),
                    'config': model_to_save.config,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if AMP_ENABLED else None,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'tokenizer_vocab_size': tokenizer.vocab_size,
                    'iters_since_best_val_loss': iters_since_best_val_loss
                }, MODEL_SAVE_PATH)
                print(f"Model saved to {MODEL_SAVE_PATH} (New best val_loss: {best_val_loss:.4f}, PPL: {eval_metrics['val_perplexity']:.2f})")
            else:
                iters_since_best_val_loss += 1
                print(f"Val loss did not improve. Best: {best_val_loss:.4f}. Iters since best: {iters_since_best_val_loss * EVAL_INTERVAL}")
                if iter_num % (EVAL_INTERVAL * 2) == 0 :
                    model_to_save = model._orig_mod if hasattr(model, '_orig_mod') and _torch_compile_actually_enabled else model
                    torch.save({
                        'model_state_dict': model_to_save.state_dict(),
                        'config': model_to_save.config,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scaler_state_dict': scaler.state_dict() if AMP_ENABLED else None,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'tokenizer_vocab_size': tokenizer.vocab_size,
                        'iters_since_best_val_loss': iters_since_best_val_loss
                    }, MODEL_SAVE_PATH + ".latest")
                    print(f"Periodic checkpoint saved to {MODEL_SAVE_PATH}.latest")

                if iters_since_best_val_loss >= patience_epochs:
                    print(f"Early stopping triggered after {patience_epochs * EVAL_INTERVAL} iterations without improvement.")
                    break
            
            if iter_num > 0 and iter_num < MAX_ITERS and iter_num % (EVAL_INTERVAL * 4) == 0 :
                model.eval()
                print("\n--- Sample Output (Iter {}) ---".format(iter_num))
                bos_token_id = tokenizer.token_to_id(tokenizer.bos_token)
                if bos_token_id is None: bos_token_id = 0
                
                context_str = tokenizer.bos_token
                context_tokens = tokenizer.encode(context_str)
                if not context_tokens: context_tokens = [0]
                
                context = torch.tensor([context_tokens], dtype=torch.long, device=DEVICE)
                model_for_gen = model._orig_mod if hasattr(model, '_orig_mod') and _torch_compile_actually_enabled else model
                
                generated_indices = model_for_gen.generate(
                    context, max_new_tokens=100, temperature=GENERATION_TEMPERATURE,
                    top_k=GENERATION_TOP_K, top_p=GENERATION_TOP_P,
                    repetition_penalty=GENERATION_REPETITION_PENALTY, mirostat_mode=GENERATION_MIROSTAT_MODE,
                    mirostat_tau=GENERATION_MIROSTAT_TAU, mirostat_eta=GENERATION_MIROSTAT_ETA,
                    eos_token_id=tokenizer.token_to_id(tokenizer.eos_token)
                )
                generated_text = tokenizer.decode(generated_indices[0].tolist(), skip_special_tokens=True)
                print(f"Sample: {generated_text.strip()}")
                print("---------------------------------\n")
                model.train()
                
    print("Training finished.")
    print(f"Total training time: {(time.time() - start_time):.2f}s")
    print(f"Final model (best val_loss {best_val_loss:.4f}) saved to {MODEL_SAVE_PATH}")

# --- Chatbot and Generation Functions ---
def run_chatbot(model: GPTLanguageModel, tokenizer: BPETokenizerWrapper):
    global current_block_size_global, _torch_compile_actually_enabled
    current_block_size_global = model.config.block_size
    
    print(f"\n--- Shakespearean Chatbot (Model: {MODEL_SAVE_PATH}) ---")
    print("Thou art Bard, a witty and eloquent chatbot who speaks in the grand style of Master William Shakespeare.")
    print("Respond to the user's modern parlance with Shakespearean flair, incorporating rich vocabulary, metaphors, and poetic rhythm where appropriate.")
    print("Type 'exit' to quit, or 'new' for a fresh discourse.")
    
    model_for_chat = model._orig_mod if hasattr(model, '_orig_mod') and _torch_compile_actually_enabled else model
    model_for_chat.eval()
    
    conversation_log: List[Dict[str, List[int]]] = []
    
    system_prompt_str = (
        "Thou art Bard, a most eloquent and witty companion, well-versed in the grandiloquent style of Master William Shakespeare. "
        "When the User doth speak in their modern tongue, thou shalt respond with verse and prose befitting the Globe Theatre itself. "
        "Employ rich vocabulary, striking metaphors, and, where thy muse doth inspire, the gentle cadence of iambic pentameter. "
        "Converse thusly:\n\n"
    )
    user_token_str = tokenizer.user_token
    bard_token_str = tokenizer.bard_token


    while True:
        try:
            user_input_text = input(f"{user_token_str}: ")
            if user_input_text.lower() == 'exit':
                break
            if user_input_text.lower() == 'new':
                print(f"{bard_token_str}: Anon! A fresh parchment for our thoughts!")
                conversation_log = []
                continue
            
            current_user_turn_prompt_str = f"{user_token_str}: {user_input_text.strip()}"
            user_input_tokens = tokenizer.encode(user_input_text.strip())

            prompt_construction_tokens: List[int] = tokenizer.encode(system_prompt_str)
            
            temp_history_render_tokens = []
            for turn_data in reversed(conversation_log):
                role_str = user_token_str if turn_data["role"] == "User" else bard_token_str
                turn_tokens_for_length_check = tokenizer.encode(f"{role_str}: ") + turn_data["content_tokens"]
                
                potential_len = len(prompt_construction_tokens) + len(temp_history_render_tokens) + \
                                len(turn_tokens_for_length_check) + len(tokenizer.encode("\n")) + \
                                len(tokenizer.encode(current_user_turn_prompt_str)) + \
                                len(tokenizer.encode(f"\n{bard_token_str}:")) + 10

                if potential_len < model.config.block_size:
                    temp_history_render_tokens = turn_tokens_for_length_check + tokenizer.encode("\n") + temp_history_render_tokens
                else:
                    break
            
            prompt_construction_tokens.extend(temp_history_render_tokens)
            prompt_construction_tokens.extend(tokenizer.encode(current_user_turn_prompt_str))
            prompt_construction_tokens.extend(tokenizer.encode(f"\n{bard_token_str}:"))

            if len(prompt_construction_tokens) > model.config.block_size:
                prompt_construction_tokens = prompt_construction_tokens[-model.config.block_size:]

            context = torch.tensor([prompt_construction_tokens], dtype=torch.long, device=DEVICE)
            
            print(f"{bard_token_str} (musing...):")
            generated_indices = model_for_chat.generate(
                context,
                max_new_tokens=GENERATION_MAX_NEW_TOKENS,
                temperature=GENERATION_TEMPERATURE,
                top_k=GENERATION_TOP_K,
                top_p=GENERATION_TOP_P,
                repetition_penalty=GENERATION_REPETITION_PENALTY,
                mirostat_mode=GENERATION_MIROSTAT_MODE,
                mirostat_tau=GENERATION_MIROSTAT_TAU,
                mirostat_eta=GENERATION_MIROSTAT_ETA,
                eos_token_id=tokenizer.token_to_id(tokenizer.eos_token)
            )
            
            num_prompt_tokens_in_batch = context.size(1)
            generated_only_token_ids = generated_indices[0, num_prompt_tokens_in_batch:].tolist()
            
            eos_id = tokenizer.token_to_id(tokenizer.eos_token)
            if eos_id is not None and generated_only_token_ids and generated_only_token_ids[-1] == eos_id:
                generated_only_token_ids = generated_only_token_ids[:-1]

            generated_text = tokenizer.decode(generated_only_token_ids, skip_special_tokens=True).strip()
            
            if generated_text:
                if not any(p in generated_text for p in ['.', '?', '!']) and len(generated_text.split()) > 3:
                    last_space = generated_text.rfind(' ')
                    if last_space != -1 and len(generated_text) - last_space < 20 :
                        generated_text = generated_text[:last_space] + "..."
                    elif len(generated_text) > 30 and len(generated_text.split()) > 5 :
                        generated_text = generated_text + "..."
            else:
                generated_text = "Alack, my tongue is tied, and words do fail me!"


            print(f"{bard_token_str}: {generated_text}")

            conversation_log.append({"role": "User", "content_tokens": user_input_tokens})
            conversation_log.append({"role": "Bard", "content_tokens": generated_only_token_ids})
            
            if len(conversation_log) > CHATBOT_HISTORY_LENGTH * 2:
                conversation_log = conversation_log[-(CHATBOT_HISTORY_LENGTH * 2):]


        except KeyboardInterrupt:
            print("\nExiting chatbot.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()
            if "CUDA out of memory" in str(e):
                print("CUDA out of memory. Try a shorter prompt or restart with a smaller model/batch size.")

# --- Main Execution ---
if __name__ == '__main__':
    if TORCH_COMPILE_ENABLED and DEVICE == 'cuda':
        try:
            import torch._dynamo
            # torch._dynamo.config.suppress_errors = True
            print("torch._dynamo.config.suppress_errors is available (currently not suppressing by default).")
        except ImportError:
            print("torch._dynamo not found, cannot set suppress_errors. This is fine for older PyTorch versions.")
        except Exception as e:
            print(f"Could not access torch._dynamo.config.suppress_errors: {e}")
        
        if hasattr(torch, 'set_float32_matmul_precision'):
            try:
                torch.set_float32_matmul_precision('high')
                print("torch.set_float32_matmul_precision('high') set.")
            except Exception as e:
                print(f"Could not set float32 matmul precision: {e}")

    print(f"--- Shakespearean GPT (Advanced v7 - GQA, RoPE, RMSNorm, SwiGLU) ---")
    print(f"Using device: {DEVICE}")
    if AMP_ENABLED: print("Automatic Mixed Precision (AMP) is enabled for CUDA.")
    if GRADIENT_CHECKPOINTING_ENABLED: print("Gradient Checkpointing is configured for training.")
    
    _torch_compile_actually_enabled = TORCH_COMPILE_ENABLED

    if _torch_compile_actually_enabled:
        print("Torch.compile() is configured (will attempt to compile and fallback on error).")
        print("NOTE: For torch.compile to work effectively on CUDA, a working Triton installation is often required.")
        print("If you see 'BackendCompilerFailed' errors related to Triton, please ensure Triton is installed correctly")
        print("for your PyTorch version and GPU architecture (pip install -U triton).")
        print("The script will attempt to fall back to eager execution if compilation fails.")
    else:
        print("Torch.compile() is DISABLED by global TORCH_COMPILE_ENABLED flag. Model will run in eager mode.")
    
    print(f"Default generation uses Mirostat mode: {GENERATION_MIROSTAT_MODE}, Temp: {GENERATION_TEMPERATURE}, Top-K: {GENERATION_TOP_K}, Top-P: {GENERATION_TOP_P}")
    print(f"Gradient Accumulation Steps: {GRAD_ACCUMULATION_STEPS}, Effective Batch Size: {BATCH_SIZE * GRAD_ACCUMULATION_STEPS}")

    download_full_shakespeare_dataset()
    raw_text_data = load_data_raw()
    tokenizer = BPETokenizerWrapper(
        tokenizer_json_path=TOKENIZER_JSON_PATH,
        train_data_path=DATA_PATH,
        vocab_size=DEFAULT_VOCAB_SIZE_BPE,
        min_frequency=MIN_FREQUENCY_BPE
    )
    
    # Tokenizer Sanity Check
    print("\n--- Tokenizer Sanity Check ---")
    sanity_text = "To be, or not to be -- that is the question."
    print(f"Original text: '{sanity_text}'")
    try:
        sanity_tokens = tokenizer.encode(sanity_text)
        print(f"Encoded tokens: {sanity_tokens}")
        sanity_decoded = tokenizer.decode(sanity_tokens)
        print(f"Decoded text: '{sanity_decoded}'")
        assert sanity_text.strip() == sanity_decoded.strip(), "Tokenizer encode/decode mismatch!"
        print("Tokenizer sanity check PASSED.")
    except Exception as e_tok_sanity:
        print(f"Tokenizer sanity check FAILED: {e_tok_sanity}")
    print("----------------------------\n")

    prepare_data_for_bpe(raw_text_data, tokenizer, VAL_SPLIT)
    del raw_text_data
    
    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        num_kv_heads=DEFAULT_NUM_KV_HEADS
    )
    current_block_size_global = model_config.block_size
    
    model = GPTLanguageModel(model_config)
    model = model.to(DEVICE)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if model_config.num_kv_heads < model_config.n_head:
        mha_config_dict = model_config.__dict__.copy()
        mha_config_dict['num_kv_heads'] = model_config.n_head
        mha_config_temp = GPTConfig(**mha_config_dict)
        mha_model_temp = GPTLanguageModel(mha_config_temp)
        mha_num_params = sum(p.numel() for p in mha_model_temp.parameters() if p.requires_grad)
        del mha_model_temp
        print(f"GQA/MQA Model initialized with {num_params/1e6:.2f}M parameters (Q_heads={model_config.n_head}, KV_heads={model_config.num_kv_heads}).")
        print(f"Equivalent MHA model would have approx. {mha_num_params/1e6:.2f}M parameters (saving of {(mha_num_params-num_params)/1e6:.2f}M params).")
    else:
        print(f"MHA Model initialized with {num_params/1e6:.2f}M parameters.")
    print(f"Full Model Config: {model_config}")
    
    uncompiled_model_for_param_groups = model
    
    if _torch_compile_actually_enabled and DEVICE == 'cuda':
        if hasattr(torch, 'compile'):
            compile_mode = "reduce-overhead"
            print(f"Attempting to compile the model with torch.compile(mode='{compile_mode}')...")
            try:
                from torch._dynamo.exc import BackendCompilerFailed

                compiled_model_candidate = torch.compile(model, mode=compile_mode)
                print("Initial torch.compile call successful. Performing a dry run with a dummy batch...")
                
                dummy_batch_size = 1
                dummy_seq_len = min(16, model_config.block_size)
                dummy_input_ids = torch.randint(0, model_config.vocab_size, (dummy_batch_size, dummy_seq_len), device=DEVICE)
                dummy_targets = torch.randint(0, model_config.vocab_size, (dummy_batch_size, dummy_seq_len), device=DEVICE)
                
                _ = compiled_model_candidate(dummy_input_ids, targets=dummy_targets)
                
                model = compiled_model_candidate
                print("Model compiled and dry run successful.")
            except BackendCompilerFailed as bcf_error:
                print(f"WARNING: torch.compile backend compiler failed during dry run (likely Triton issue): {bcf_error}")
                print("Proceeding without model compilation (eager mode). This will be slower.")
                model = uncompiled_model_for_param_groups
                _torch_compile_actually_enabled = False
            except Exception as e:
                print(f"WARNING: torch.compile() or dry run failed with an unexpected error: {e}")
                print("Proceeding without model compilation (eager mode). This will be slower.")
                model = uncompiled_model_for_param_groups
                _torch_compile_actually_enabled = False
        else:
            print("torch.compile() not available in this PyTorch version. Proceeding without compilation.")
            _torch_compile_actually_enabled = False
    
    param_groups = uncompiled_model_for_param_groups.get_param_groups(WEIGHT_DECAY)
    
    fused_adam_possible = False
    if DEVICE == 'cuda' and AMP_ENABLED:
        try:
            temp_param = torch.nn.Parameter(torch.randn(1,1, device=DEVICE))
            _ = torch.optim.AdamW([temp_param], lr=0.001, fused=True)
            fused_adam_possible = True
            print("Fused AdamW seems available.")
        except RuntimeError:
            fused_adam_possible = False
            print("Fused AdamW not available or encountered an issue during test.")

    optimizer = torch.optim.AdamW(param_groups, lr=MAX_LEARNING_RATE, betas=(ADAMW_BETA1, ADAMW_BETA2), fused=fused_adam_possible)
    if fused_adam_possible: print("Using Fused AdamW optimizer.")
    else: print("Using standard AdamW optimizer (Fused not available/enabled or AMP disabled).")
    
    scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)

    print("\nWhat would you like to do?")
    print("1. Train a new model (or continue training)")
    print("2. Load pre-trained model & run chatbot")
    print("3. Load pre-trained model & generate sample text")
    
    while True:
        choice = input("Enter choice (1, 2, or 3): ")
        if choice in ['1', '2', '3']:
            break
        print("Invalid choice.")

    if choice == '1':
        print("\n--- Training Mode ---")
        iter_num_at_load = 0
        best_val_loss_at_load = float('inf')
        iters_since_best_val_loss_at_load = 0

        if os.path.exists(MODEL_SAVE_PATH):
            load_q = input(f"Model file '{MODEL_SAVE_PATH}' found. Load and continue? (yes/no): ").lower()
            if load_q == 'yes':
                try:
                    loaded_checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
                    loaded_config_data = loaded_checkpoint.get('config')
                    
                    target_model_config_from_chkpt = model_config
                    if isinstance(loaded_config_data, GPTConfig):
                        target_model_config_from_chkpt = loaded_config_data
                        print("Loaded GPTConfig object from checkpoint.")
                    elif isinstance(loaded_config_data, dict):
                        print("Checkpoint config is a dict. Reconstructing GPTConfig.")
                        cfg_fields = {f.name: f.default for f in fields(GPTConfig)}
                        cfg_args = {}
                        for fname, fdefault in cfg_fields.items():
                            cfg_args[fname] = loaded_config_data.get(fname, fdefault)
                        cfg_args['vocab_size'] = loaded_config_data.get('vocab_size', tokenizer.vocab_size)
                        if 'num_kv_heads' not in loaded_config_data:
                            cfg_args['num_kv_heads'] = loaded_config_data.get('n_head', model_config.num_kv_heads)
                        if 'ffn_dim_multiple_of' not in loaded_config_data:
                             cfg_args['ffn_dim_multiple_of'] = DEFAULT_FFN_DIM_MULTIPLE_OF
                        try:
                            target_model_config_from_chkpt = GPTConfig(**cfg_args)
                            print(f"Reconstructed GPTConfig: {target_model_config_from_chkpt}")
                        except TypeError as e_cfg:
                            print(f"Error reconstructing GPTConfig: {e_cfg}. Using current script's config.")
                            target_model_config_from_chkpt = model_config
                    else:
                        print(f"Cannot interpret config from checkpoint. Using current script's config: {model_config}")
                        target_model_config_from_chkpt = model_config
                    
                    if target_model_config_from_chkpt.vocab_size != tokenizer.vocab_size:
                        print(f"Warning: Checkpoint vocab_size ({target_model_config_from_chkpt.vocab_size}) "
                              f"differs from current tokenizer ({tokenizer.vocab_size}). "
                              f"Forcing current tokenizer's vocab_size for model re-initialization.")
                        target_model_config_from_chkpt.vocab_size = tokenizer.vocab_size
                    
                    if 'tokenizer_vocab_size' in loaded_checkpoint and \
                       loaded_checkpoint['tokenizer_vocab_size'] != tokenizer.vocab_size:
                        print(f"CRITICAL WARNING: Checkpoint was saved with tokenizer_vocab_size "
                              f"({loaded_checkpoint['tokenizer_vocab_size']}) but current tokenizer has "
                              f"({tokenizer.vocab_size}). This can lead to serious issues if not intended.")

                    reinit_needed = any([
                        not hasattr(model, 'config'),
                        model.config.n_embd != target_model_config_from_chkpt.n_embd,
                        model.config.n_layer != target_model_config_from_chkpt.n_layer,
                        model.config.n_head != target_model_config_from_chkpt.n_head,
                        model.config.num_kv_heads != target_model_config_from_chkpt.num_kv_heads,
                        model.config.block_size != target_model_config_from_chkpt.block_size,
                        model.config.vocab_size != target_model_config_from_chkpt.vocab_size,
                        model.config.ffn_dim_multiple_of != target_model_config_from_chkpt.ffn_dim_multiple_of
                    ])

                    if reinit_needed:
                        print(f"Re-initializing model based on checkpoint config: {target_model_config_from_chkpt}")
                        model = GPTLanguageModel(target_model_config_from_chkpt).to(DEVICE)
                        current_block_size_global = model.config.block_size
                        uncompiled_model_for_param_groups = model
                        
                        if _torch_compile_actually_enabled and DEVICE == 'cuda' and hasattr(torch, 'compile'):
                            print("Re-compiling re-initialized model (with dry run)...")
                            try:
                                compiled_model_candidate = torch.compile(model, mode="reduce-overhead")
                                dummy_input_ids = torch.randint(0, target_model_config_from_chkpt.vocab_size, (1, min(16, target_model_config_from_chkpt.block_size)), device=DEVICE)
                                dummy_targets = torch.randint(0, target_model_config_from_chkpt.vocab_size, (1, min(16, target_model_config_from_chkpt.block_size)), device=DEVICE)
                                _ = compiled_model_candidate(dummy_input_ids, targets=dummy_targets)
                                model = compiled_model_candidate
                                print("Re-initialized model compiled and dry run successful.")
                            except Exception as comp_e:
                                print(f"torch.compile() or dry run failed for re-initialized model: {comp_e}. Proceeding uncompiled.")
                                model = uncompiled_model_for_param_groups
                                _torch_compile_actually_enabled = False
                        
                        param_groups = uncompiled_model_for_param_groups.get_param_groups(WEIGHT_DECAY)
                        optimizer = torch.optim.AdamW(param_groups, lr=MAX_LEARNING_RATE, betas=(ADAMW_BETA1, ADAMW_BETA2), fused=fused_adam_possible)
                        print("Optimizer re-initialized for new model structure.")

                    model.load_state_dict(loaded_checkpoint['model_state_dict'])
                    checkpoint_data_global.update(loaded_checkpoint)
                    print(f"Model state loaded. Active config: {model.config}")

                    if not reinit_needed:
                        if 'optimizer_state_dict' in checkpoint_data_global and checkpoint_data_global['optimizer_state_dict']:
                            try:
                                optimizer.load_state_dict(checkpoint_data_global['optimizer_state_dict'])
                                print("Optimizer state loaded.")
                            except Exception as e_optim:
                                print(f"Could not load optimizer state (may be due to model structure change or other issues): {e_optim}. Optimizer will start fresh.")
                        if AMP_ENABLED and 'scaler_state_dict' in checkpoint_data_global and checkpoint_data_global['scaler_state_dict']:
                            scaler.load_state_dict(checkpoint_data_global['scaler_state_dict'])
                            print("AMP GradScaler state loaded.")
                    else:
                        print("Optimizer was re-initialized due to model structure change; not loading its state from checkpoint.")
                    
                    iter_num_at_load = checkpoint_data_global.get('iter_num', 0)
                    best_val_loss_at_load = checkpoint_data_global.get('best_val_loss', float('inf'))
                    iters_since_best_val_loss_at_load = checkpoint_data_global.get('iters_since_best_val_loss', 0)
                    print(f"Resuming training from iter {iter_num_at_load}, best_val_loss: {best_val_loss_at_load:.4f}")

                    if not hasattr(model, '_orig_mod') and _torch_compile_actually_enabled and DEVICE == 'cuda' and hasattr(torch, 'compile'):
                        if not reinit_needed: print("Loaded model was not compiled. Attempting to compile now (with dry run)...")
                        try:
                            compiled_model_candidate = torch.compile(model, mode="reduce-overhead")
                            dummy_input_ids = torch.randint(0, model.config.vocab_size, (1, min(16, model.config.block_size)), device=DEVICE)
                            dummy_targets = torch.randint(0, model.config.vocab_size, (1, min(16, model.config.block_size)), device=DEVICE)
                            _ = compiled_model_candidate(dummy_input_ids, targets=dummy_targets)
                            model = compiled_model_candidate
                            print("Model compiled successfully after loading.")
                        except Exception as e:
                            print(f"torch.compile() or dry run failed after loading: {e}. Proceeding uncompiled.")
                            model = uncompiled_model_for_param_groups
                            _torch_compile_actually_enabled = False
                    print("Continuing training.")
                except Exception as e:
                    print(f"Error loading checkpoint: {e}. Starting fresh training.")
                    model = GPTLanguageModel(model_config).to(DEVICE)
                    current_block_size_global = model.config.block_size
                    uncompiled_model_for_param_groups = model
                    if _torch_compile_actually_enabled and DEVICE == 'cuda' and hasattr(torch, 'compile'):
                        try:
                            compiled_model_candidate = torch.compile(model, mode="reduce-overhead")
                            dummy_input_ids = torch.randint(0, model_config.vocab_size, (1, min(16, model_config.block_size)), device=DEVICE)
                            dummy_targets = torch.randint(0, model_config.vocab_size, (1, min(16, model_config.block_size)), device=DEVICE)
                            _ = compiled_model_candidate(dummy_input_ids, targets=dummy_targets)
                            model = compiled_model_candidate
                            print("New model compiled and dry run successful.")
                        except Exception as comp_e:
                            print(f"torch.compile() or dry run failed for new model: {comp_e}. Proceeding uncompiled.")
                            model = uncompiled_model_for_param_groups
                            _torch_compile_actually_enabled = False
                    param_groups = uncompiled_model_for_param_groups.get_param_groups(WEIGHT_DECAY)
                    optimizer = torch.optim.AdamW(param_groups, lr=MAX_LEARNING_RATE, betas=(ADAMW_BETA1, ADAMW_BETA2), fused=fused_adam_possible)
                    iter_num_at_load = 0; best_val_loss_at_load = float('inf'); checkpoint_data_global.clear()
                    iters_since_best_val_loss_at_load = 0
            else:
                print("Starting fresh training (user chose not to load existing checkpoint).")
                current_block_size_global = model.config.block_size; iter_num_at_load = 0; best_val_loss_at_load = float('inf'); checkpoint_data_global.clear()
                iters_since_best_val_loss_at_load = 0
        else:
            print(f"No model found at {MODEL_SAVE_PATH}. Starting fresh training.")
            current_block_size_global = model.config.block_size; iter_num_at_load = 0; best_val_loss_at_load = float('inf'); checkpoint_data_global.clear()
            iters_since_best_val_loss_at_load = 0
        
        checkpoint_data_global['iter_num'] = iter_num_at_load
        checkpoint_data_global['best_val_loss'] = best_val_loss_at_load
        checkpoint_data_global['iters_since_best_val_loss'] = iters_since_best_val_loss_at_load
        train_model(model, optimizer, tokenizer, scaler)
        
        if input("Training done. Run chatbot? (yes/no): ").lower() == 'yes':
            final_model_for_chat = model._orig_mod if hasattr(model, '_orig_mod') and _torch_compile_actually_enabled else model
            run_chatbot(final_model_for_chat, tokenizer)

    elif choice == '2' or choice == '3':
        model_file_to_load = MODEL_SAVE_PATH
        if not os.path.exists(model_file_to_load):
            print(f"Model '{model_file_to_load}' not found. Please train a model first (choice 1).")
            exit()
        
        print(f"\n--- Loading Pre-trained Model from {model_file_to_load} ---")
        try:
            checkpoint = torch.load(model_file_to_load, map_location=DEVICE, weights_only=False)
            loaded_config_data = checkpoint.get('config')
            
            target_model_config_from_chkpt = model_config
            if isinstance(loaded_config_data, GPTConfig):
                target_model_config_from_chkpt = loaded_config_data
            elif isinstance(loaded_config_data, dict):
                print("Checkpoint config is dict. Reconstructing GPTConfig for inference.")
                cfg_fields = {f.name: f.default for f in fields(GPTConfig)}
                cfg_args = {}
                for fname, fdefault in cfg_fields.items():
                    cfg_args[fname] = loaded_config_data.get(fname, fdefault)
                cfg_args['vocab_size'] = loaded_config_data.get('vocab_size', tokenizer.vocab_size)
                if 'num_kv_heads' not in loaded_config_data:
                    cfg_args['num_kv_heads'] = loaded_config_data.get('n_head', model_config.num_kv_heads)
                if 'ffn_dim_multiple_of' not in loaded_config_data:
                    cfg_args['ffn_dim_multiple_of'] = DEFAULT_FFN_DIM_MULTIPLE_OF
                try:
                    target_model_config_from_chkpt = GPTConfig(**cfg_args)
                except TypeError as e_cfg:
                    print(f"Error reconstructing GPTConfig for inference: {e_cfg}. Using current script's config.")
                    target_model_config_from_chkpt = model_config
            else:
                print(f"Warning: Loaded config not recognized. Using current script's config for inference: {model_config}")
                target_model_config_from_chkpt = model_config

            if target_model_config_from_chkpt.vocab_size != tokenizer.vocab_size:
                print(f"Warning: Checkpoint vocab_size ({target_model_config_from_chkpt.vocab_size}) "
                      f"differs from current tokenizer ({tokenizer.vocab_size}). "
                      f"Forcing current tokenizer's vocab_size for inference model.")
                target_model_config_from_chkpt.vocab_size = tokenizer.vocab_size
            if 'tokenizer_vocab_size' in checkpoint and checkpoint['tokenizer_vocab_size'] != tokenizer.vocab_size:
                 print(f"CRITICAL WARNING (Inference): Checkpoint tokenizer_vocab_size "
                       f"({checkpoint['tokenizer_vocab_size']}) mismatch with current ({tokenizer.vocab_size}).")

            model_for_inference = GPTLanguageModel(target_model_config_from_chkpt).to(DEVICE)
            model_for_inference.load_state_dict(checkpoint['model_state_dict'])
            current_block_size_global = model_for_inference.config.block_size
            print(f"Model loaded successfully. Active config for inference: {model_for_inference.config}")
            
            uncompiled_model_for_inference = model_for_inference
            if _torch_compile_actually_enabled and DEVICE == 'cuda' and hasattr(torch, 'compile'):
                compile_mode_inference = "reduce-overhead"
                print(f"Attempting to compile loaded model for inference (mode='{compile_mode_inference}', with dry run)...")
                try:
                    compiled_model_candidate = torch.compile(model_for_inference, mode=compile_mode_inference)
                    dummy_input_ids = torch.randint(0, target_model_config_from_chkpt.vocab_size, (1, min(16, target_model_config_from_chkpt.block_size)), device=DEVICE)
                    _ = compiled_model_candidate(dummy_input_ids)
                    model_for_inference = compiled_model_candidate
                    print("Model compiled and dry run successful for inference.")
                except Exception as e:
                    print(f"torch.compile() or dry run failed for inference: {e}. Proceeding uncompiled.")
                    model_for_inference = uncompiled_model_for_inference
            
            model_for_inference.eval()
            
            if choice == '2':
                run_chatbot(model_for_inference, tokenizer)
            else: # choice == '3'
                print("\n--- Sample Text Generation ---")
                original_start_prompt = input("Enter a starting prompt (e.g., 'To be, or not to be:' or leave blank for BOS token): ")
                model_input_string = ""
                if not original_start_prompt.strip():
                    print(f"Using BOS token ({tokenizer.bos_token}) as start.")
                    model_input_string = tokenizer.bos_token
                else:
                    model_input_string = tokenizer.bos_token + " " + original_start_prompt
                
                context_tokens = tokenizer.encode(model_input_string)
                if not context_tokens:
                    print("Warning: Prompt tokenized to empty. Using BOS token.")
                    context_tokens = tokenizer.encode(tokenizer.bos_token)
                if not context_tokens: context_tokens = [0]
                
                context = torch.tensor([context_tokens], dtype=torch.long, device=DEVICE)
                
                print("Generating text...")
                generated_indices = model_for_inference.generate(
                    context,
                    max_new_tokens=GENERATION_MAX_NEW_TOKENS,
                    temperature=GENERATION_TEMPERATURE,
                    top_k=GENERATION_TOP_K,
                    top_p=GENERATION_TOP_P,
                    repetition_penalty=GENERATION_REPETITION_PENALTY,
                    mirostat_mode=GENERATION_MIROSTAT_MODE,
                    mirostat_tau=GENERATION_MIROSTAT_TAU,
                    mirostat_eta=GENERATION_MIROSTAT_ETA,
                    eos_token_id=tokenizer.token_to_id(tokenizer.eos_token)
                )
                
                num_prompt_tokens = len(context_tokens)
                generated_only_token_ids = generated_indices[0, num_prompt_tokens:].tolist()
                generated_text = tokenizer.decode(generated_only_token_ids, skip_special_tokens=True)
                
                print("\n--- Generated Shakespearean Text ---")
                if original_start_prompt.strip():
                    print(f"Prompt: {original_start_prompt}")
                    print(f"Continuation: {generated_text.strip()}")
                else:
                    print(generated_text.strip())
                print("--- End of Generation ---")
        except Exception as e:
            print(f"Error loading model or during generation: {e}")
            print(f"Ensure model file '{MODEL_SAVE_PATH}' exists and is compatible. "
                  f"Consider deleting '{TOKENIZER_JSON_PATH}' to retrain tokenizer if issues persist with vocabulary mismatches.")

    print("\nExiting Shakespearean GPT.")
