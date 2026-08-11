package com.blt.transformer;

import com.blt.nn.LayerNorm;
import com.blt.nn.Linear;
import com.blt.nn.Module;
import com.blt.tensor.Tensor;

import java.nio.charset.StandardCharsets;
import java.util.Random;

/**
 * The Byte Latent Transformer (GPT) Model.
 *
 * ASSIGNMENT:
 * Assemble the full architecture!
 * 1. Token Embeddings (Vocab size 256 for bytes).
 * 2. Positional Embeddings (Learnable).
 * 3. Stack of Transformer Blocks.
 * 4. Final LayerNorm.
 * 5. Language Model Head (Linear to Vocab size).
 */
public class GPT extends Module {

    private static final int DEFAULT_MAX_SEQUENCE_LENGTH = 256;

    private final int vocabSize;
    private final int dModel;
    private final int maxSequenceLength;
    private final Tensor tokenEmbeddings;
    private final Tensor positionalEmbeddings;
    private final Block[] blocks;
    private final LayerNorm finalLayerNorm;
    private final Linear head;

    public GPT(int vocabSize, int dModel, int numLayers, int numHeads) {
        this(vocabSize, dModel, numLayers, numHeads, DEFAULT_MAX_SEQUENCE_LENGTH, 0L);
    }

    public GPT(int vocabSize, int dModel, int numLayers, int numHeads, int maxSequenceLength) {
        this(vocabSize, dModel, numLayers, numHeads, maxSequenceLength, 0L);
    }

    public GPT(
            int vocabSize,
            int dModel,
            int numLayers,
            int numHeads,
            int maxSequenceLength,
            long seed) {
        if (vocabSize <= 0 || dModel <= 0 || numLayers < 0 || maxSequenceLength <= 0) {
            throw new IllegalArgumentException("Model dimensions must be positive.");
        }
        if (numHeads <= 0 || dModel % numHeads != 0) {
            throw new IllegalArgumentException("dModel must be divisible by a positive numHeads.");
        }

        this.vocabSize = vocabSize;
        this.dModel = dModel;
        this.maxSequenceLength = maxSequenceLength;
        this.tokenEmbeddings = new Tensor(vocabSize, dModel);
        this.positionalEmbeddings = new Tensor(maxSequenceLength, dModel);
        this.blocks = new Block[numLayers];
        this.finalLayerNorm = new LayerNorm(dModel);

        Random rng = new Random(seed);
        this.tokenEmbeddings.fillRandom(rng, 0.1f);
        this.positionalEmbeddings.fillRandom(rng, 0.1f);

        for (int i = 0; i < numLayers; i++) {
            this.blocks[i] = new Block(dModel, numHeads, rng);
        }
        this.head = new Linear(dModel, vocabSize, rng);
    }

    @Override
    public Tensor forward(Tensor idx) {
        int[] tokens = readSingleSequence(idx);
        if (tokens.length > maxSequenceLength) {
            throw new IllegalArgumentException(
                    "Sequence length " + tokens.length + " exceeds max length " + maxSequenceLength + ".");
        }

        float[][] tokenData = tokenEmbeddings.getData();
        float[][] positionData = positionalEmbeddings.getData();
        float[][] hidden = new float[tokens.length][dModel];

        for (int position = 0; position < tokens.length; position++) {
            int token = tokens[position];
            if (token < 0 || token >= vocabSize) {
                throw new IllegalArgumentException("Token " + token + " is outside vocab size " + vocabSize + ".");
            }

            for (int dim = 0; dim < dModel; dim++) {
                hidden[position][dim] = tokenData[token][dim] + positionData[position][dim];
            }
        }

        Tensor x = new Tensor(hidden);
        for (Block block : blocks) {
            x = block.forward(x);
        }

        return head.forward(finalLayerNorm.forward(x));
    }

    /**
     * Generate text!
     * 
     * @param startBytes   Initial prompt
     * @param maxNewTokens How many new bytes to generate
     */
    public void generate(byte[] startBytes, int maxNewTokens) {
        byte[] output = generateBytes(startBytes, maxNewTokens);
        System.out.println(new String(output, StandardCharsets.ISO_8859_1));
    }

    public byte[] generateBytes(byte[] startBytes, int maxNewTokens) {
        if (startBytes == null || startBytes.length == 0) {
            throw new IllegalArgumentException("startBytes must contain at least one byte.");
        }
        if (maxNewTokens < 0) {
            throw new IllegalArgumentException("maxNewTokens must be non-negative.");
        }

        int[] tokens = new int[startBytes.length + maxNewTokens];
        int length = startBytes.length;
        for (int i = 0; i < startBytes.length; i++) {
            tokens[i] = startBytes[i] & 0xff;
            if (tokens[i] >= vocabSize) {
                throw new IllegalArgumentException("Start byte " + tokens[i] + " is outside vocab size " + vocabSize + ".");
            }
        }

        for (int step = 0; step < maxNewTokens; step++) {
            int contextLength = Math.min(length, maxSequenceLength);
            int contextStart = length - contextLength;
            float[][] context = new float[1][contextLength];
            for (int i = 0; i < contextLength; i++) {
                context[0][i] = tokens[contextStart + i];
            }

            Tensor logits = forward(new Tensor(context));
            float[] nextTokenLogits = logits.getData()[logits.getRows() - 1];
            tokens[length] = argmax(nextTokenLogits);
            length++;
        }

        byte[] output = new byte[length];
        for (int i = 0; i < length; i++) {
            output[i] = (byte) (tokens[i] & 0xff);
        }
        return output;
    }

    private int[] readSingleSequence(Tensor idx) {
        if (idx.getRows() == 1) {
            int[] tokens = new int[idx.getCols()];
            for (int col = 0; col < idx.getCols(); col++) {
                tokens[col] = Math.round(idx.get(0, col));
            }
            return tokens;
        }

        if (idx.getCols() == 1) {
            int[] tokens = new int[idx.getRows()];
            for (int row = 0; row < idx.getRows(); row++) {
                tokens[row] = Math.round(idx.get(row, 0));
            }
            return tokens;
        }

        throw new UnsupportedOperationException(
                "GPT.forward currently supports one sequence as a 1xN or Nx1 Tensor.");
    }

    private int argmax(float[] values) {
        int bestIndex = 0;
        float bestValue = values[0];
        for (int i = 1; i < values.length; i++) {
            if (values[i] > bestValue) {
                bestValue = values[i];
                bestIndex = i;
            }
        }
        return bestIndex;
    }
}
