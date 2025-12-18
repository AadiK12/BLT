package com.blt.transformer;

import com.blt.nn.Module;
import com.blt.tensor.Tensor;

/**
 * Multi-Head Scaled Dot-Product Attention.
 * The core of the Transformer.
 *
 * ASSIGNMENT:
 * This is the most complex part. Break it down:
 * 1. Linear projections for Query, Key, Value.
 * 2. Split heads (reshape/transpose).
 * 3. Scaled Dot-Product Attention: softmax(QK^T / sqrt(d_k)) * V
 * 4. Masking! (Causal mask needed for GPT).
 * 5. Concatenate heads and final linear projection.
 */
public class MultiHeadAttention extends Module {

    private int numHeads;
    private int dModel;
    private int dHead;

    // TODO: Define Linear layers for q, k, v, and output.

    public MultiHeadAttention(int dModel, int numHeads) {
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.dHead = dModel / numHeads;

        // TODO: Initialize your Linear layers here.
    }

    @Override
    public Tensor forward(Tensor input) {
        // TODO: 1. Project input to Q, K, V.
        // TODO: 2. Calculate attention scores.
        // TODO: 3. Apply causal mask (set future positions to -infinity).
        // TODO: 4. Softmax.
        // TODO: 5. Multiply by V.
        // TODO: 6. Merge heads and project output.
        return null;
    }
}
