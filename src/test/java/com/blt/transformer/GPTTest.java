package com.blt.transformer;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.blt.tensor.Tensor;
import java.util.Arrays;
import org.junit.jupiter.api.Test;

class GPTTest {

    @Test
    void sameSeedProducesIdenticalLogits() {
        GPT first = new GPT(256, 16, 2, 4, 32, 123L);
        GPT second = new GPT(256, 16, 2, 4, 32, 123L);
        Tensor input = new Tensor(new float[][] {{'B', 'L', 'T'}});
        Tensor firstLogits = first.forward(input);
        Tensor secondLogits = second.forward(input);
        for (int row = 0; row < firstLogits.getRows(); row++) {
            assertArrayEquals(firstLogits.getData()[row], secondLogits.getData()[row], 0.0f);
        }
    }

    @Test
    void differentSeedsAffectModelLogits() {
        GPT first = new GPT(256, 16, 2, 4, 32, 123L);
        GPT second = new GPT(256, 16, 2, 4, 32, 124L);
        Tensor input = new Tensor(new float[][] {{'B', 'L', 'T'}});
        assertFalse(
                Arrays.equals(
                        first.forward(input).getData()[0],
                        second.forward(input).getData()[0]));
    }

    @Test
    void causalPrefixDoesNotChangeWhenFutureBytesAreAppended() {
        GPT model = new GPT(256, 16, 2, 4, 32, 456L);
        Tensor prefix = model.forward(new Tensor(new float[][] {{10, 20, 30}}));
        Tensor longer = model.forward(new Tensor(new float[][] {{10, 20, 30, 40, 50}}));
        for (int row = 0; row < prefix.getRows(); row++) {
            assertArrayEquals(prefix.getData()[row], longer.getData()[row], 1e-6f);
        }
    }

    @Test
    void logitsUseByteVocabularyAndSequenceLength() {
        GPT model = new GPT(256, 16, 1, 4, 32, 1L);
        Tensor logits = model.forward(new Tensor(new float[][] {{1, 2, 3, 4}}));
        assertEquals(4, logits.getRows());
        assertEquals(256, logits.getCols());
    }

    @Test
    void invalidConfigurationAndSequenceFailExplicitly() {
        assertThrows(IllegalArgumentException.class, () -> new GPT(256, 15, 1, 4));
        GPT model = new GPT(256, 16, 1, 4, 2, 1L);
        assertThrows(
                IllegalArgumentException.class,
                () -> model.forward(new Tensor(new float[][] {{1, 2, 3}})));
    }
}
