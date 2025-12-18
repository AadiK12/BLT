package com.blt.transformer;

import com.blt.nn.Module;
import com.blt.tensor.Tensor;

/**
 * The Byte Latent Transformer (GPT) Model.
 *
 * ASSIGNMENT:
 * Assemble the full architecture!
 * 1. Token Embeddings (Vocab size 256 for bytes).
 * 2. Positional Embeddings (Learnable).
 * 3. Stack of Transformer Blocks.
 * 4. Final LayerNorm.
 * 5. Language Model Head (Linear to Vocab size).
 */
public class GPT extends Module {

    // TODO: Define Embeddings, Blocks, LN_f, and Head.

    public GPT(int vocabSize, int dModel, int numLayers, int numHeads) {
        // TODO: Initialize everything.
    }

    @Override
    public Tensor forward(Tensor idx) {
        // idx is a Tensor of indices [batch, seq_len] (integers).
        // TODO: Lookup embeddings.
        // TODO: Add positional embeddings.
        // TODO: Pass through blocks.
        // TODO: Final norm and projection.
        return null;
    }

    /**
     * Generate text!
     * 
     * @param startBytes   Initial prompt
     * @param maxNewTokens How many new bytes to generate
     */
    public void generate(byte[] startBytes, int maxNewTokens) {
        // TODO: Implement the generation loop.
        // 1. Forward pass.
        // 2. Pick next token (greedy or sample).
        // 3. Append to sequence and repeat.
    }
}
