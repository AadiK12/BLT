package com.blt.tensor;

import java.util.Random;

public class Tensor {

    private float[][] data;
    private int rows;
    private int cols;
    private float[] grad;

    public Tensor(float[][] data) {
        this.data = data;
        this.rows = data.length;
        this.cols = data[0].length;
    }

    public Tensor(int rows, int cols) {
        this.data = new float[rows][cols];
        this.rows = rows;
        this.cols = cols;
    }

    public float[][] getData() {
        return data;
    }

    public Tensor matmul(Tensor other) {
        float[][] res = new float[this.rows][other.cols];
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                for (int k = 0; k < other.rows; k++) {
                    res[i][j] += this.data[i][k] * other.data[k][j];
                }
            }
        }
        return new Tensor(res);
    }

    public Tensor add(Tensor other) {
        float[][] res = new float[this.rows][this.cols];
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                res[i][j] = this.data[i][j] + other.data[i][j];
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
            float sumExp = 0.0f;
            for (int j = 0; j < cols; j++) {
                float val = (float) Math.exp(data[i][j]);
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
        Random rng = new Random();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                this.data[i][j] = (rng.nextFloat() - 0.5f) * 0.1f;
            }
        }
    }

    // Helper for debugging
    public String toString() {
        return "Tensor(" + rows + "x" + cols + ")";
    }
}
