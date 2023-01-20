package org.ea.mlp;

import java.util.Arrays;
import java.util.Map;
import java.util.Random;

public class MultiLayerPerceptron {
    private float[] inputNeurons;
    private float[] outputNeurons;
    private float[] hiddenNeurons;
    private float[] weights;

    //a buffer for the weights
    private float[] tempWeights;
    //used to keep the previous weights before modification, for momentum
    private float[] prWeights;

    private int inputN;
    private int outputN;
    private int hiddenN;
    private int hiddenL;

    public MultiLayerPerceptron(int input, int output, int hiddenN, int hiddenL) {
        inputN = input;
        outputN = output; //numbers 0-9
        this.hiddenL = hiddenL;
        this.hiddenN = hiddenN;

        int weightsSize = inputN*hiddenN+(hiddenN*hiddenN*(hiddenL-1))+hiddenN*outputN;

        //init weights
        weights = new float[weightsSize];

        //init neurons vector
        inputNeurons = new float[inputN];
        hiddenNeurons = new float[hiddenN*hiddenL];
        outputNeurons = new float[outputN];

        Random rn = new Random();
        rn.setSeed(System.currentTimeMillis());
        for(int i = 0; i < weightsSize; i++) {
            weights[i] = (rn.nextFloat() - 0.5f);
        }
    }

    //direct distribution
    public void calculateNetwork() {
        //first hidden layer
        for(int hidden = 0; hidden < hiddenN; hidden++) {
            hiddenNeurons[hidden] = 0.0f; // Layer one neuron.
            for(int input = 0 ; input < inputN; input++) {
                hiddenNeurons[hidden] += inputNeurons[input] * inputToHidden(input,hidden);
            }
            //activation function
            hiddenNeurons[hidden] = sigmoid(hiddenNeurons[hidden]);
        }

        //next hidden layers
        for(int i = 2; i <= hiddenL; i++) {
            //layer calculation
            for(int j = 0; j < hiddenN; j++) {
                hiddenNeurons[(i-1)*hiddenN + j] = 0.0f;
                for(int k = 0; k <hiddenN; k++) {
                    hiddenNeurons[(i-1)*hiddenN + j] += hiddenNeurons[(i-2)*hiddenN + k] * hiddenToHidden(i,k,j);
                }
                //activation function
                hiddenNeurons[(i-1)*hiddenN + j] = sigmoid(hiddenNeurons[(i-1)*hiddenN + j]);
            }
        }

        //output hidden
        for(int i = 0; i < outputN; i++) {
            outputNeurons[i] = 0.0f;
            for(int j = 0; j < hiddenN; j++) {
                outputNeurons[i] += hiddenNeurons[(hiddenL-1)*hiddenN + j] * hiddenToOutput(j,i);
            }
            //activation function
            outputNeurons[i] = sigmoid( outputNeurons[i] );
        }
    }

    //input
    public void populateInput(int[] data) {
        for(int i = 0; i < inputN; i++) {
            //if pixel is on
            //set the corresponding neuron
            if(data[i] <= 0) {
                inputNeurons[i] = 1.0f;
            } else {
                inputNeurons[i] = 0.0f;
            }
        }
    }

    //trains the network
    public boolean trainNetwork(float teachingStep, float lmse, float momentum, Map<String, Integer> imageMap, int maxEpochs, int step) {
        float mse = 999.0f;
        int epochs = 1;
        float error = 0.0f;
        //the delta of the output layer
        float[] odelta = new float[outputN];
        //the delta of each hidden layer
        float[] hdelta = new float[hiddenN*hiddenL];

        ImageReader imageReader = new ImageReader();
        tempWeights = Arrays.copyOf(weights, weights.length);

        //save previous weights before mod
        prWeights = Arrays.copyOf(weights, weights.length);

        int target = 0;
        while(Math.abs(mse-lmse) > 0.0001f) {
            //reset the mean square error
            mse = 0.0f;

            //for each file
            for(Map.Entry<String, Integer> entry : imageMap.entrySet()) {
                //input neurons
                populateInput(imageReader.readImage(entry.getKey()));
                target = entry.getValue();

                //calculate the network
                calculateNetwork();

                //error calculation
                for(int i = 0; i < outputN; i++) {
                    //delta of the output layer and error
                    if(i != target) {
                        odelta[i] = (0.0f - outputNeurons[i])*dersigmoid(outputNeurons[i]);
                        error += (0.0f - outputNeurons[i])*(0.0f-outputNeurons[i]);
                    } else {
                        odelta[i] = (1.0f - outputNeurons[i])*dersigmoid(outputNeurons[i]);
                        error += (1.0f - outputNeurons[i])*(1.0f-outputNeurons[i]);
                    }
                }

                //reverse propagation
                for(int i = 0; i < hiddenN; i++) {
                    //zero the values
                    hdelta[(hiddenL-1)*hiddenN+i] = 0.0f;
                    //calculation delta
                    for(int j = 0; j < outputN; j++) {
                        hdelta[(hiddenL-1)*hiddenN+i] += odelta[j] * hiddenToOutput(i,j);
                    }
                    //derivative sigmoid
                    hdelta[(hiddenL-1)*hiddenN+i] *= dersigmoid(hiddenAt(hiddenL,i));
                }

                //reverse propagation for next hidden layers
                for(int i = hiddenL-1; i > 0; i--) {
                    //calculation delta
                    for(int j = 0; j < hiddenN; j++) { //from
                        //zero the values
                        hdelta[(i-1)*hiddenN+j] = 0.0f;
                        for(int k = 0; k < hiddenN; k++) { //to
                            //calculation delta
                            hdelta[(i-1)*hiddenN+j] += hdelta[i*hiddenN+k] * hiddenToHidden(i+1,j,k);
                        }
                        //derivative sigmoid
                        hdelta[(i-1)*hiddenN+j] *= dersigmoid(hiddenAt(i,j));
                    }
                }

                //Weights modification
                //save weights
                tempWeights = Arrays.copyOf(weights, weights.length);

                //hidden to Input weights
                for(int i = 0; i < inputN; i ++) {
                    for(int j = 0; j < hiddenN; j ++) {
                        weights[inputN*j+i] +=
                                (momentum * (inputToHidden(i,j) - _prev_inputToHidden(i,j))) +
                                        (teachingStep * hdelta[j] * inputNeurons[i]);
                    }
                }

                //hidden to hidden weights, provided more than 1 layer exists
                for(int i = 2; i <= hiddenL; i++) {
                    for(int j = 0; j < hiddenN; j ++) { //from
                        for(int k =0; k < hiddenN; k ++) { //to
                            weights[inputN*hiddenN+((i-2)*hiddenN*hiddenN)+hiddenN*j+k] +=
                                    (momentum * (hiddenToHidden(i,j,k) - _prev_hiddenToHidden(i,j,k))) +
                                            (teachingStep * hdelta[(i-1)*hiddenN+k] * hiddenAt(i-1,j));
                        }
                    }
                }

                //last hidden layer to output weights
                for(int i = 0; i < outputN; i++) {
                    for(int j = 0; j < hiddenN; j ++) {
                        weights[(inputN*hiddenN + (hiddenL-1)*hiddenN*hiddenN + j*outputN+i)] +=
                                (momentum * (hiddenToOutput(j,i) - _prev_hiddenToOutput(j,i))) +
                                        (teachingStep * odelta[i] * hiddenAt(hiddenL,j));
                    }
                }

                prWeights = Arrays.copyOf(tempWeights, tempWeights.length);

                //add to the total mse for this epoch
                mse += error / (outputN+1f);
                //zero the values
                error = 0.0f;
            }

            if(epochs % step == 0) {
                System.out.println(epochs + " - " + mse);
            }
            if(epochs > maxEpochs) break;
            epochs++;
        }
        System.out.println(epochs + " - " + mse);
        return true;
    }

    //recalls the network for a given bitmap file
    public int recallNetwork(String filename, int[] data, int answer) {
        //fill input neurons
        populateInput(data);
        //calculate the network
        calculateNetwork();

        float winner = -1;
        int index = 0;

        //find the best fitting output
        for(int i = 0; i < outputN; i++) {
            if(outputNeurons[i] > winner) {
                winner = outputNeurons[i];
                index = i;
            }
        }
        System.out.println(filename + ", correct answer "+answer+", represents a " + index);
        //comparison of expectation and result
        return index == answer ? 1 : 0;
    }

    public int recallNetworkTest(int[] data) {
        //fill input neurons
        populateInput(data);
        //calculate the network
        calculateNetwork();

        float winner = -1;
        int index = 0;

        //find the best fitting output
        for(int i = 0; i < outputN; i++) {
            if(outputNeurons[i] > winner) {
                winner = outputNeurons[i];
                index = i;
            }
        }
        return index;
    }

    private float inputToHidden(int inp, int hid) {
        return weights[inputN*hid+inp];
    }
    private float hiddenToHidden(int toLayer, int fromHid, int toHid) {
        return weights[inputN*hiddenN+ ((toLayer-2)*hiddenN*hiddenN)+hiddenN*fromHid+toHid];
    }
    private float hiddenToOutput(int hid, int out)  {
        return weights[(inputN*hiddenN + (hiddenL-1)*hiddenN*hiddenN + hid*outputN+out)];
    }

    /*macros for the previous Weights*/
    private float _prev_inputToHidden(int inp, int hid) {
        return prWeights[inputN*hid+inp];
    }
    private float _prev_hiddenToHidden(int toLayer, int fromHid, int toHid) {
        return prWeights[inputN*hiddenN+ ((toLayer-2)*hiddenN*hiddenN)+hiddenN*fromHid+toHid];
    }
    private float _prev_hiddenToOutput(int hid, int out) {
        return prWeights[inputN*hiddenN + (hiddenL-1)*hiddenN*hiddenN + hid*outputN+out];
    }

    /*macro to locate the appropriate hidden neuron*/
    private float hiddenAt(int layer,int hid) {
        return hiddenNeurons[(layer-1)*hiddenN + hid];
    }

    /*function*/
    private float sigmoid(float value) {
        return (float)(1f/(1f+Math.exp(-value)));
    }
    private float dersigmoid(float value) {
        return (value*(1f-value));
    }

}
