# DEV Specifications

## Project Goal
- Provide a flexible, model-agnostic feature extraction toolkit for transformer architectures, enabling consistent retrieval and saving of internal representations (e.g., embeddings, hidden states, attention, MLP, and norm outputs) across model families.

## Supported Model Types
- Positional Encoding
    - RoPE
    - Absolute
- Attention
    - c_attn
        - only in GPT2
        - Use Conv1d (not the one in torch.nn, but the one in the original gpt2 codebase) to linear project input to q, k, v.
    - Standard (one linear projection for each of q, k, v)
    - GQA (shares key and value vectors over some heads)
    - (Maybe softmax1 in gpt-oss?)
- MLP
    - Standard (one linear layer, one activation function)
    - Gated (SwiGLU or GEGLU style gating)
- Normalization
    - LayerNorm
    - RMSNorm
- Transformer Layer
    - Pre-LN (norm -> attn -> add -> norm -> mlp -> add)
    - Post-LN (attn -> add -> norm -> mlp -> add -> norm)
    - Both (used in recent gemma) (norm -> attn -> ... (not rure..))

## Features that can be retrieved
- Embedding: the output of the token embedding layer
- Hidden States: the output of each transformer layer
- Attention:
    - Query Vectors for each head
    - Key vectors for each head (if gqa, duplicate for heads that share keys)
    - Value vectors for each head (if gqa, duplicate for heads that share values)
    - Attention weights for each head (after softmax)
- MLP:
    - Activation output for each layer (after nonlinearity, before output projection)
- Normalization:
    - Input to each norm layer
    - Output of each norm layer

## Saving
- save in torch .pt format
- save path should be `{output_dir}/{dataset_name}/{model_name}/{feature_name}/{sentence_id}.pt`
