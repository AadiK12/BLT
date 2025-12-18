package com.blt.nn;

import com.blt.tensor.Tensor;

/**
 * A Linear (Fully Connected) Layer.
 * y = xA^T + b
 *
 * ASSIGNMENT:
 * Implement the forward pass for a dense layer.
 * 1. Initialize weights (randomly) and bias (zeros) in the constructor.
 * 2. Perform the affine transformation in forward().
 */
public class Linear extends Module {

    private Tensor weights; // Shape: [out_features, in_features]
    private Tensor bias; // Shape: [out_features]

    public Linear(int inFeatures, int outFeatures) {
        // TODO: Initialize weights with small random values.
        // TODO: Initialize bias with zeros.
        this.weights = new Tensor(outFeatures, inFeatures);
        this.bias = new Tensor(new float[outFeatures]);

    }

    @Override
    public Tensor forward(Tensor input) {
        // TODO: Implement y = input * weights^T + bias
        // Hint: You might need to broadcast the bias addition.
        return null;
    }
}
