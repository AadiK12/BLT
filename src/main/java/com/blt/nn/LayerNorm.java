package com.blt.nn;

import com.blt.tensor.Tensor;

/**
 * Layer Normalization.
 * Normalizes input across the last dimension.
 *
 * ASSIGNMENT:
 * Implement standard LayerNorm: (x - mean) / sqrt(var + eps)
 * Then scale by 'gamma' and shift by 'beta' (learnable parameters).
 */
public class LayerNorm extends Module {

    private Tensor gamma; // Scale parameter
    private Tensor beta; // Shift parameter
    private float epsilon = 1e-5f;

    public LayerNorm(int features) {
        // TODO: Initialize gamma to ones, beta to zeros.
    }

    @Override
    public Tensor forward(Tensor input) {
        // TODO: Calculate mean and variance along the last dimension.
        // TODO: Normalize.
        // TODO: Apply gamma and beta.
        return null; // Placeholder
    }
}
