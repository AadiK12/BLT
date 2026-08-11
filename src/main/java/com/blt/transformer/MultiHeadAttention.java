package com.blt.transformer;

import com.blt.nn.Module;
import com.blt.tensor.Tensor;
import com.blt.nn.Linear;

import java.util.Random;

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

    private final int numHeads;
    private final int dModel;
    private final int dHead;

    private final Linear q;
    private final Linear k;
    private final Linear v;
    private final Linear out;

    public MultiHeadAttention(int dModel, int numHeads) {
        this(dModel, numHeads, new Random(0L));
    }

    public MultiHeadAttention(int dModel, int numHeads, Random rng) {
        if (dModel <= 0 || numHeads <= 0) {
            throw new IllegalArgumentException("dModel and numHeads must be positive.");
        }
        if (dModel % numHeads != 0) {
            throw new IllegalArgumentException("dModel must be divisible by numHeads.");
        }

        this.dModel = dModel;
        this.numHeads = numHeads;
        this.dHead = dModel / numHeads;

        this.q = new Linear(dModel, dModel, rng);
        this.k = new Linear(dModel, dModel, rng);
        this.v = new Linear(dModel, dModel, rng);
        this.out = new Linear(dModel, dModel, rng);
    }

    @Override
    public Tensor forward(Tensor input) {
        if (input.getCols() != dModel) {
            throw new IllegalArgumentException(
                    "Attention expected input width " + dModel + " but got " + input.getCols() + ".");
        }

        Tensor query = q.forward(input);
        Tensor key = k.forward(input);
        Tensor value = v.forward(input);

        float[][] qData = query.getData();
        float[][] kData = key.getData();
        float[][] vData = value.getData();
        int seqLen = input.getRows();
        float[][] context = new float[seqLen][dModel];
        float scale = (float) (1.0 / Math.sqrt(dHead));

        for (int head = 0; head < numHeads; head++) {
            int offset = head * dHead;

            for (int position = 0; position < seqLen; position++) {
                float[] scores = new float[seqLen];
                float maxScore = Float.NEGATIVE_INFINITY;

                for (int source = 0; source < seqLen; source++) {
                    if (source > position) {
                        scores[source] = -1.0e9f;
                    } else {
                        float dot = 0.0f;
                        for (int dim = 0; dim < dHead; dim++) {
                            dot += qData[position][offset + dim] * kData[source][offset + dim];
                        }
                        scores[source] = dot * scale;
                    }

                    if (scores[source] > maxScore) {
                        maxScore = scores[source];
                    }
                }

                float expSum = 0.0f;
                for (int source = 0; source <= position; source++) {
                    scores[source] = (float) Math.exp(scores[source] - maxScore);
                    expSum += scores[source];
                }

                for (int source = 0; source <= position; source++) {
                    float weight = scores[source] / expSum;
                    for (int dim = 0; dim < dHead; dim++) {
                        context[position][offset + dim] += weight * vData[source][offset + dim];
                    }
                }
            }
        }

        return out.forward(new Tensor(context));
    }
}
