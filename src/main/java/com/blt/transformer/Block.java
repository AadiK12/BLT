package com.blt.transformer;

import com.blt.nn.LayerNorm;
import com.blt.nn.Linear;
import com.blt.nn.Module;
import com.blt.tensor.Tensor;

import java.util.Random;

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

    private final MultiHeadAttention attention;
    private final LayerNorm ln1;
    private final LayerNorm ln2;
    private final Linear feedForwardIn;
    private final Linear feedForwardOut;

    public Block(int dModel, int numHeads) {
        this(dModel, numHeads, new Random(0L));
    }

    public Block(int dModel, int numHeads, Random rng) {
        this.attention = new MultiHeadAttention(dModel, numHeads, rng);
        this.ln1 = new LayerNorm(dModel);
        this.ln2 = new LayerNorm(dModel);
        this.feedForwardIn = new Linear(dModel, dModel * 4, rng);
        this.feedForwardOut = new Linear(dModel * 4, dModel, rng);
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor attended = attention.forward(ln1.forward(input));
        Tensor x = input.add(attended);
        Tensor feedForward = feedForwardOut.forward(gelu(feedForwardIn.forward(ln2.forward(x))));
        return x.add(feedForward);
    }

    private Tensor gelu(Tensor input) {
        float[][] data = input.getData();
        float[][] out = new float[input.getRows()][input.getCols()];
        float coefficient = (float) Math.sqrt(2.0 / Math.PI);

        for (int i = 0; i < input.getRows(); i++) {
            for (int j = 0; j < input.getCols(); j++) {
                float x = data[i][j];
                float inner = coefficient * (x + 0.044715f * x * x * x);
                out[i][j] = 0.5f * x * (1.0f + (float) Math.tanh(inner));
            }
        }

        return new Tensor(out);
    }
}
