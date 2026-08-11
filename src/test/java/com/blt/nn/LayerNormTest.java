package com.blt.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.blt.tensor.Tensor;
import org.junit.jupiter.api.Test;

class LayerNormTest {

    @Test
    void outputHasApproximatelyZeroMeanAndUnitVariance() {
        Tensor output = new LayerNorm(4).forward(
                new Tensor(new float[][] {{1, 2, 3, 4}, {4, 3, 2, 1}}));
        for (float[] row : output.getData()) {
            float mean = 0.0f;
            for (float value : row) {
                mean += value;
            }
            mean /= row.length;
            float variance = 0.0f;
            for (float value : row) {
                variance += (value - mean) * (value - mean);
            }
            variance /= row.length;
            assertEquals(0.0f, mean, 1e-5f);
            assertEquals(1.0f, variance, 2e-5f);
        }
    }

    @Test
    void invalidFeatureWidthFailsExplicitly() {
        LayerNorm layerNorm = new LayerNorm(4);
        assertThrows(
                IllegalArgumentException.class,
                () -> layerNorm.forward(new Tensor(new float[][] {{1, 2, 3}})));
    }
}
