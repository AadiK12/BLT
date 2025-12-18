package com.blt.nn;

import com.blt.tensor.Tensor;

/**
 * Abstract base class for all Neural Network modules.
 *
 * ASSIGNMENT:
 * This class abstracts the common functionality of all layers.
 * - forward(): The main computation.
 * - parameters(): (Optional) Useful if you want to implement an optimizer.
 */
public abstract class Module {

    /**
     * The forward pass of the layer.
     * Takes an input Tensor and returns the output Tensor.
     */
    public abstract Tensor forward(Tensor input);

    /**
     * Optional: Implement a list of parameters for optimization.
     */
    // public List<Tensor> parameters() { ... }
}
