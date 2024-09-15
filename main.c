#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INPUT_SIZE 2
#define HIDDEN_SIZE 4
#define OUTPUT_SIZE 1
#define LEARNING_RATE 0.1

// Sigmoid activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of sigmoid
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// Initialize weights with small random values
void initialize_weights(double weights[], int size) {
    for (int i = 0; i < size; i++) {
        weights[i] = (double)rand() / RAND_MAX * 0.2 - 0.1;
    }
}

int main() {
    // Input data (XOR problem)
    double inputs[4][INPUT_SIZE] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    double expected_output[4][OUTPUT_SIZE] = {
        {0},
        {1},
        {1},
        {0}
    };

    // Weights
    double input_hidden_weights[INPUT_SIZE * HIDDEN_SIZE];
    double hidden_output_weights[HIDDEN_SIZE * OUTPUT_SIZE];

    // Initialize weights
    initialize_weights(input_hidden_weights, INPUT_SIZE * HIDDEN_SIZE);
    initialize_weights(hidden_output_weights, HIDDEN_SIZE * OUTPUT_SIZE);

    // Training
    for (int epoch = 0; epoch < 10000; epoch++) {
        for (int i = 0; i < 4; i++) {
            // Forward propagation
            double hidden_layer[HIDDEN_SIZE];
            double output_layer[OUTPUT_SIZE];
            double hidden_layer_input[HIDDEN_SIZE];

            // Calculate hidden layer input
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                hidden_layer_input[j] = 0;
                for (int k = 0; k < INPUT_SIZE; k++) {
                    hidden_layer_input[j] += inputs[i][k] * input_hidden_weights[j * INPUT_SIZE + k];
                }
                hidden_layer[j] = sigmoid(hidden_layer_input[j]);
            }

            // Calculate output layer input
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                output_layer[j] = 0;
                for (int k = 0; k < HIDDEN_SIZE; k++) {
                    output_layer[j] += hidden_layer[k] * hidden_output_weights[j * HIDDEN_SIZE + k];
                }
                output_layer[j] = sigmoid(output_layer[j]);
            }

            // Backpropagation
            double output_error[OUTPUT_SIZE];
            double hidden_error[HIDDEN_SIZE];
            double hidden_output_weights_update[HIDDEN_SIZE * OUTPUT_SIZE];
            double input_hidden_weights_update[INPUT_SIZE * HIDDEN_SIZE];

            // Calculate output layer error
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                output_error[j] = expected_output[i][j] - output_layer[j];
            }

            // Calculate hidden layer error
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                hidden_error[j] = 0;
                for (int k = 0; k < OUTPUT_SIZE; k++) {
                    hidden_error[j] += output_error[k] * hidden_output_weights[k * HIDDEN_SIZE + j];
                }
                hidden_error[j] *= sigmoid_derivative(hidden_layer[j]);
            }

            // Update hidden-output weights
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                for (int k = 0; k < HIDDEN_SIZE; k++) {
                    hidden_output_weights_update[j * HIDDEN_SIZE + k] = LEARNING_RATE * output_error[j] * hidden_layer[k];
                    hidden_output_weights[j * HIDDEN_SIZE + k] += hidden_output_weights_update[j * HIDDEN_SIZE + k];
                }
            }

            // Update input-hidden weights
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                for (int k = 0; k < INPUT_SIZE; k++) {
                    input_hidden_weights_update[j * INPUT_SIZE + k] = LEARNING_RATE * hidden_error[j] * inputs[i][k];
                    input_hidden_weights[j * INPUT_SIZE + k] += input_hidden_weights_update[j * INPUT_SIZE + k];
                }
            }
        }
    }

    // Test the network
    for (int i = 0; i < 4; i++) {
        double hidden_layer[HIDDEN_SIZE];
        double output_layer[OUTPUT_SIZE];
        double hidden_layer_input[HIDDEN_SIZE];

        // Forward propagation
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            hidden_layer_input[j] = 0;
            for (int k = 0; k < INPUT_SIZE; k++) {
                hidden_layer_input[j] += inputs[i][k] * input_hidden_weights[j * INPUT_SIZE + k];
            }
            hidden_layer[j] = sigmoid(hidden_layer_input[j]);
        }

        for (int j = 0; j < OUTPUT_SIZE; j++) {
            output_layer[j] = 0;
            for (int k = 0; k < HIDDEN_SIZE; k++) {
                output_layer[j] += hidden_layer[k] * hidden_output_weights[j * HIDDEN_SIZE + k];
            }
            output_layer[j] = sigmoid(output_layer[j]);
        }

        printf("Input: (%f, %f) => Output: %f\n", inputs[i][0], inputs[i][1], output_layer[0]);
    }

    return 0;
}
