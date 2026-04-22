package com.blt.tensor;

import java.util.Random;

public class Tensor {

    private final float[][] data;
    private final int rows;
    private final int cols;

    public Tensor(float[][] data) {
        if (data == null || data.length == 0 || data[0].length == 0) {
            throw new IllegalArgumentException("Tensor data must be non-empty.");
        }
        this.rows = data.length;
        this.cols = data[0].length;
        this.data = new float[rows][cols];

        for (int i = 0; i < rows; i++) {
            if (data[i].length != cols) {
                throw new IllegalArgumentException("Tensor data must be rectangular.");
            }
            System.arraycopy(data[i], 0, this.data[i], 0, cols);
        }
    }

    public Tensor(int rows, int cols) {
        if (rows <= 0 || cols <= 0) {
            throw new IllegalArgumentException("Tensor dimensions must be positive.");
        }
        this.data = new float[rows][cols];
        this.rows = rows;
        this.cols = cols;
    }

    public float[][] getData() {
        return data;
    }

    public int getRows() {
        return rows;
    }

    public int getCols() {
        return cols;
    }

    public float get(int row, int col) {
        return data[row][col];
    }

    public void set(int row, int col, float value) {
        data[row][col] = value;
    }

    public Tensor copy() {
        return new Tensor(data);
    }

    public Tensor matmul(Tensor other) {
        if (this.cols != other.rows) {
            throw new IllegalArgumentException(
                    "Cannot multiply shapes " + this + " and " + other + ".");
        }

        float[][] res = new float[this.rows][other.cols];
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                for (int k = 0; k < this.cols; k++) {
                    res[i][j] += this.data[i][k] * other.data[k][j];
                }
            }
        }
        return new Tensor(res);
    }

    public Tensor add(Tensor other) {
        if (other.rows == 1 && other.cols == this.cols) {
            return addRowVector(other);
        }
        if (other.rows != this.rows || other.cols != this.cols) {
            throw new IllegalArgumentException("Cannot add shapes " + this + " and " + other + ".");
        }

        float[][] res = new float[this.rows][this.cols];
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                res[i][j] = this.data[i][j] + other.data[i][j];
            }
        }
        return new Tensor(res);
    }

    public Tensor subtract(Tensor other) {
        if (other.rows != this.rows || other.cols != this.cols) {
            throw new IllegalArgumentException("Cannot subtract shapes " + this + " and " + other + ".");
        }

        float[][] res = new float[this.rows][this.cols];
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                res[i][j] = this.data[i][j] - other.data[i][j];
            }
        }
        return new Tensor(res);
    }

    public Tensor multiply(float scalar) {
        float[][] res = new float[this.rows][this.cols];
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                res[i][j] = this.data[i][j] * scalar;
            }
        }
        return new Tensor(res);
    }

    public Tensor transpose() {
        float[][] res = new float[this.cols][this.rows];
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                res[j][i] = this.data[i][j];
            }
        }
        return new Tensor(res);
    }

    public Tensor softmax() {
        float[][] res = new float[rows][cols];

        for (int i = 0; i < rows; i++) {
            float max = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < cols; j++) {
                if (data[i][j] > max) {
                    max = data[i][j];
                }
            }

            float sumExp = 0.0f;
            for (int j = 0; j < cols; j++) {
                float val = (float) Math.exp(data[i][j] - max);
                res[i][j] = val;
                sumExp += val;
            }
            for (int j = 0; j < cols; j++) {
                res[i][j] /= sumExp;
            }
        }
        return new Tensor(res);
    }

    public void fillRandom() {
        fillRandom(0.1f);
    }

    public void fillRandom(float scale) {
        Random rng = new Random();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                this.data[i][j] = (rng.nextFloat() - 0.5f) * scale;
            }
        }
    }

    public void fill(float value) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                this.data[i][j] = value;
            }
        }
    }

    private Tensor addRowVector(Tensor rowVector) {
        float[][] res = new float[this.rows][this.cols];
        float[] bias = rowVector.data[0];
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                res[i][j] = this.data[i][j] + bias[j];
            }
        }
        return new Tensor(res);
    }

    @Override
    public String toString() {
        return "Tensor(" + rows + "x" + cols + ")";
    }
}
