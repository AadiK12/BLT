package com.blt.tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class TensorTest {

    @Test
    void matrixMultiplicationMatchesKnownValues() {
        Tensor left = new Tensor(new float[][] {{1, 2}, {3, 4}});
        Tensor right = new Tensor(new float[][] {{5, 6}, {7, 8}});

        Tensor output = left.matmul(right);

        assertArrayEquals(new float[] {19, 22}, output.getData()[0], 1e-6f);
        assertArrayEquals(new float[] {43, 50}, output.getData()[1], 1e-6f);
    }

    @Test
    void invalidMatrixMultiplicationFailsExplicitly() {
        Tensor left = new Tensor(2, 3);
        Tensor right = new Tensor(2, 2);
        assertThrows(IllegalArgumentException.class, () -> left.matmul(right));
    }

    @Test
    void rowBiasBroadcastsAcrossRows() {
        Tensor values = new Tensor(new float[][] {{1, 2}, {3, 4}});
        Tensor bias = new Tensor(new float[][] {{10, 20}});
        Tensor output = values.add(bias);
        assertArrayEquals(new float[] {11, 22}, output.getData()[0], 1e-6f);
        assertArrayEquals(new float[] {13, 24}, output.getData()[1], 1e-6f);
    }

    @Test
    void stableSoftmaxRowsSumToOne() {
        Tensor probabilities = new Tensor(
                new float[][] {{1000, 1001, 1002}, {-1000, -999, -998}}).softmax();
        for (float[] row : probabilities.getData()) {
            assertEquals(1.0f, row[0] + row[1] + row[2], 1e-6f);
        }
    }

    @Test
    void seededRandomInitializationIsReproducible() {
        Tensor first = new Tensor(3, 4);
        Tensor second = new Tensor(3, 4);
        first.fillRandom(42L, 0.1f);
        second.fillRandom(42L, 0.1f);
        for (int row = 0; row < first.getRows(); row++) {
            assertArrayEquals(first.getData()[row], second.getData()[row], 0.0f);
        }
    }
}
