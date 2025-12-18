package com.blt.transformer;

import com.blt.nn.Module;
import com.blt.tensor.Tensor;

/**
 * A Single Transformer Block.
 * Contains:
 * 1. MultiHeadAttention
 * 2. FeedForward Network (MLP)
 * 3. LayerNorms and Residual Connections.
 *
 * Structure (Pre-Norm):
 * x = x + Attention(LayerNorm(x))
 * x = x + MLP(LayerNorm(x))
 */
public class Block extends Module {

    // TODO: Define Attention, FeedForward, and LayerNorm modules.

    public Block(int dModel, int numHeads) {
        // TODO: Initialize components.
    }

    @Override
    public Tensor forward(Tensor input) {
        // TODO: Implement the residual block logic.
        return null;
    }
}
