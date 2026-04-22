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

    private final Tensor gamma;
    private final Tensor beta;
    private final float epsilon = 1e-5f;

    public LayerNorm(int features) {
        this.gamma = new Tensor(1, features);
        this.beta = new Tensor(1, features);
        this.gamma.fill(1.0f);
    }

    @Override
    public Tensor forward(Tensor input) {
        float[][] x = input.getData();
        int rows = input.getRows();
        int cols = input.getCols();
        float[][] out = new float[rows][cols];
        float[] gammaData = gamma.getData()[0];
        float[] betaData = beta.getData()[0];

        if (cols != gamma.getCols()) {
            throw new IllegalArgumentException(
                    "LayerNorm expected input width " + gamma.getCols() + " but got " + cols + ".");
        }

        for (int i = 0; i < rows; i++) {
            float sum = 0;
            for (int j = 0; j < cols; j++) {
                sum += x[i][j];
            }
            float mean = sum / cols;
            float sumDiffSq = 0;
            for (int j = 0; j < cols; j++) {
                float diff = x[i][j] - mean;
                sumDiffSq += diff * diff;
            }
            float var = sumDiffSq / cols;
            float std = (float) Math.sqrt(var + epsilon);
            for (int j = 0; j < cols; j++) {
                out[i][j] = ((x[i][j] - mean) / std) * gammaData[j] + betaData[j];
            }
        }
        return new Tensor(out);
    }
}
