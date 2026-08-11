package com.blt.nn;

import com.blt.tensor.Tensor;

import java.util.Random;

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

    private final Tensor weights;
    private final Tensor bias;

    public Linear(int inFeatures, int outFeatures) {
        this(inFeatures, outFeatures, new Random(0L));
    }

    public Linear(int inFeatures, int outFeatures, long seed) {
        this(inFeatures, outFeatures, new Random(seed));
    }

    public Linear(int inFeatures, int outFeatures, Random rng) {
        if (inFeatures <= 0 || outFeatures <= 0) {
            throw new IllegalArgumentException("Linear dimensions must be positive.");
        }
        this.weights = new Tensor(outFeatures, inFeatures);
        this.weights.fillRandom(rng, 0.1f);
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
