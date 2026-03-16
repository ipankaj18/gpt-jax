from typing import Any, Optional, Tuple
from dataclasses import dataclass
from functools import partial
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict


distribution = nn.initializers.normal(stddev=0.02)


@dataclass(frozen=True)
class GPTConfig:
    # ~1B parameter configuration:
    #   num_layers=24, num_heads=16, num_embeds=2048
    #   Approx param count:
    #     Embedding: vocab_size * num_embeds = 50304 * 2048 ≈ 103M (weight-tied)
    #     Each Block:
    #       Attn QKV:  2048 * (3*2048) = 12.6M
    #       Attn proj: 2048 * 2048     =  4.2M
    #       MLP fc:    2048 * 8192     = 16.8M
    #       MLP proj:  8192 * 2048     = 16.8M
    #       LayerNorms: ~negligible
    #       Per block ≈ 50.4M
    #     24 blocks: 24 * 50.4M ≈ 1,210M
    #     Total (with embeddings, weight-tied output) ≈ ~1.2B
    block_size: int = 2048          # longer context for large model
    vocab_size: int = 50304         # GPT-2 vocab rounded to nearest 64 for efficiency
    num_layers: int = 24            # depth (GPT-2 XL uses 48; GPT-3 1.3B uses 24)
    num_heads: int = 16             # attention heads
    num_embeds: int = 2048          # hidden dimension
    mlp_ratio: int = 4              # MLP hidden dim = mlp_ratio * num_embeds
    dropout_rate: float = 0.0       # typically 0 for inference / large pretrained models
    use_bias: bool = True
    dtype: Optional[Any] = jnp.bfloat16   # bfloat16 for memory efficiency at 1B scale


class SelfAttention(nn.Module):
    num_heads: int
    dtype: Any = jnp.bfloat16
    dropout_rate: float = 0.0
    deterministic: Optional[bool] = None
    use_proj_bias: bool = True

    @nn.compact
    def __call__(self, x, mask, deterministic=None):
        B, T, C = x.shape
        assert C % self.num_heads == 0, f"Embedding dim {C} must be divisible by num_heads {self.num_heads}"
        head_dim = C // self.num_heads
        deterministic = nn.merge_param('deterministic', self.deterministic, deterministic)

        # Fused QKV projection
        qkv = nn.Dense(
            3 * C,
            use_bias=self.use_proj_bias,
            dtype=self.dtype,
            name='c_attn',
            kernel_init=distribution,
        )(x)
        qkv = qkv.reshape(B, T, 3 * self.num_heads, head_dim)
        q, k, v = jnp.array_split(qkv, 3, axis=2)

        # Scaled dot-product attention
        scale = 1.0 / jnp.sqrt(head_dim).astype(self.dtype)
        attn = jnp.einsum('...qhd,...khd->...hqk', q, k) * scale
        attn = jnp.where(mask, attn, jnp.finfo(self.dtype).min)
        attn = jax.nn.softmax(attn).astype(self.dtype)
        attn = nn.Dropout(self.dropout_rate)(attn, deterministic=deterministic)

        # Weighted sum over values
        x = jnp.einsum('...hqk,...khd->...qhd', attn, v).reshape(B, T, C)
        x = nn.Dense(
            C,
            use_bias=self.use_proj_bias,
            dtype=self.dtype,
            name='c_proj',
            kernel_init=distribution,
        )(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        return x


class MLP(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, x, deterministic=None):
        B, T, C = x.shape
        hidden_dim = self.config.mlp_ratio * C
        x = nn.Dense(
            hidden_dim,
            dtype=self.config.dtype,
            use_bias=self.config.use_bias,
            name='c_fc',
            kernel_init=distribution,
        )(x)
        x = nn.gelu(x, approximate=True)
        x = nn.Dense(
            C,
            dtype=self.config.dtype,
            use_bias=self.config.use_bias,
            name='c_proj',
            kernel_init=distribution,
        )(x)
        x = nn.Dropout(self.config.dropout_rate)(x, deterministic)
        return x


class Block(nn.Module):
    config: GPTConfig

    def setup(self):
        self.ln_1 = nn.LayerNorm(
            epsilon=1e-5,
            dtype=self.config.dtype,
            use_bias=self.config.use_bias,
        )
        self.attn = SelfAttention(
            num_heads=self.config.num_heads,
            dtype=self.config.dtype,
            dropout_rate=self.config.dropout_rate,
            use_proj_bias=self.config.use_bias,
        )
        self.ln_2 = nn.LayerNorm(
            epsilon=1e-5,
            dtype=self.config.dtype,
            use_bias=self.config.use_bias,
        )
        self.mlp = MLP(self.config)

    def __call__(self, x, mask=None, deterministic=None):
        x = x + self.attn(self.ln_1(x), mask, deterministic)
        x = x + self.mlp(self.ln_2(x), deterministic)
        return x


class GPT(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, idx, deterministic=None):
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds block_size {self.config.block_size}"
        )

        pos = jnp.arange(0, T)[None]                        # [1, T]
        attn_mask = nn.make_causal_mask(idx, dtype=bool)    # [B, 1, T, T]

        wte = nn.Embed(
            self.config.vocab_size,
            self.config.num_embeds,
            dtype=self.config.dtype,
            name='wte',
        )
        wpe = nn.Embed(
            self.config.block_size,
            self.config.num_embeds,
            dtype=self.config.dtype,
            name='wpe',
        )

        token_embed = wte(idx)    # [B, T, num_embeds]
        pos_embed = wpe(pos)      # [1, T, num_embeds]
        x = nn.Dropout(self.config.dropout_rate)(token_embed + pos_embed, deterministic)

        for i in range(self.config.num_layers):
            x = Block(self.config, name=str(i))(x, attn_mask, deterministic=deterministic)

        x = nn.LayerNorm(
            epsilon=1e-5,
            dtype=self.config.dtype,
            use_bias=self.config.use_bias,
            name='ln_f',
        )(x)

        # Weight-tied output projection (shares weights with token embedding)
        logits = wte.attend(x)   # [B, T, vocab_size]
        return logits

    def init(self, rng):
        """
        JIT-compiled init to avoid materialising the full parameter set eagerly.
        Uses a short dummy sequence to trace shapes.
        """
        tokens = jnp.zeros((2, self.config.block_size), dtype=jnp.uint32)
        params = jax.jit(super().init, static_argnums=(2,))(rng, tokens, True)
        return params

    @staticmethod
    def param_count(config: GPTConfig) -> int:
        """Estimate total trainable parameter count."""
        C = config.num_embeds
        V = config.vocab_size
        T = config.block_size
        L = config.num_layers
        H = config.mlp_ratio

        embed_params   = V * C + T * C          # wte + wpe (output head is weight-tied)
        attn_per_block = C * (3 * C) + C * C    # QKV + proj kernels
        attn_bias      = (3 * C) + C             # QKV + proj biases
        mlp_per_block  = C * (H * C) + (H * C) * C
        mlp_bias       = (H * C) + C
        ln_per_block   = 4 * C                  # 2x LayerNorm (scale + bias each)
        per_block      = attn_per_block + attn_bias + mlp_per_block + mlp_bias + ln_per_block

        final_ln = 2 * C
        total = embed_params + L * per_block + final_ln
        return total


# ---------------------------------------------------------------------------
# Predefined scale configurations
# ---------------------------------------------------------------------------

CONFIGS = {
    # Original GPT-2 sizes (kept for reference)
    'gpt2':        GPTConfig(num_layers=12, num_heads=12, num_embeds=768,  block_size=1024, dtype=jnp.float32),
    'gpt2-medium': GPTConfig(num_layers=24, num_heads=16, num_embeds=1024, block_size=1024, dtype=jnp.float32),
    'gpt2-large':  GPTConfig(num_layers=36, num_heads=20, num_embeds=1280, block_size=1024, dtype=jnp.float32),
    'gpt2-xl':     GPTConfig(num_layers=48, num_heads=25, num_embeds=1600, block_size=1024, dtype=jnp.float32),

    # ~1B parameter model
    'gpt-1b': GPTConfig(
        num_layers=24,
        num_heads=16,
        num_embeds=2048,
        mlp_ratio=4,
        block_size=2048,
        vocab_size=50304,
        dropout_rate=0.0,
        use_bias=True,
        dtype=jnp.bfloat16,
    ),
}


def convert_hf_params(hf_params: FrozenDict, num_heads: int, num_embeds: int) -> FrozenDict:
    """
    Convert HuggingFace GPT-2 checkpoint parameters to this model's layout.
    Transposes Conv1D kernels (HF uses [in, out]; we use [out, in] via nn.Dense).
    """
    params = unfreeze(hf_params['transformer'])
    for k, v in params.pop('h', {}).items():
        params[k] = v

    params = flatten_dict(params, sep='.')
    for k in list(params.keys()):
        if k.endswith('attn.c_attn.kernel'):
            params[k] = params[k].T
        elif k.endswith('attn.c_proj.kernel'):
            params[k] = params[k].T
        elif k.split('.')[1] == 'mlp' and k.endswith('kernel'):
            params[k] = params[k].T

    params = unflatten_dict({f'params.{k}': v for k, v in params.items()}, sep='.')
    return freeze(params)


def get_pretrained_params(model_type: str) -> Tuple[GPTConfig, FrozenDict]:
    """
    Load pretrained weights from HuggingFace for GPT-2 family models.
    Note: only gpt2/medium/large/xl checkpoints are publicly available;
    the 'gpt-1b' config must be trained from scratch or loaded from a
    compatible checkpoint.
    """
    assert model_type in CONFIGS, f"Unknown model type '{model_type}'. Choose from: {list(CONFIGS)}"
    assert model_type.startswith('gpt2'), (
        "Pretrained HuggingFace weights are only available for gpt2/medium/large/xl. "
        "The 'gpt-1b' config must be trained from scratch."
    )

    from transformers import FlaxGPT2LMHeadModel
    print(f"Loading weights from pretrained checkpoint: {model_type}")

    config = CONFIGS[model_type]
    model_hf = FlaxGPT2LMHeadModel.from_pretrained(model_type)
    hf_params = model_hf.params['transformer']
    params = convert_hf_params(hf_params, config.num_heads, config.num_embeds)
    return config, params


if __name__ == '__main__':
    cfg = CONFIGS['gpt-1b']
    n = GPT.param_count(cfg)
    print(f"GPT-1B config: {cfg}")
    print(f"Estimated parameter count: {n / 1e9:.3f}B")