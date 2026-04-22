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

    private Tensor weights;
    private Tensor bias;

    public Linear(int inFeatures, int outFeatures) {
        this.weights = new Tensor(outFeatures, inFeatures);
        this.weights.fillRandom();
        this.bias = new Tensor(1, outFeatures);
    }

    @Override
    public Tensor forward(Tensor input) {
        if (input.getCols() != weights.getCols()) {
            throw new IllegalArgumentException(
                    "Linear expected input width " + weights.getCols() + " but got " + input.getCols() + ".");
        }

        return input.matmul(weights.transpose()).add(bias);
    }

    public static void main(String[] args) {
        System.out.println("Verifying Linear Layer...");
        Linear linear = new Linear(2, 3);
        Tensor input = new Tensor(new float[][] { { 1, 2 }, { 3, 4 } });
        Tensor output = linear.forward(input);
        System.out.println("Output shape should be 2x3: " + output);
    }
}
