# Laboratory
Before training an artificial neural network, we must first understand what an artificial neural network is. For this reason, I have implemented a modular multilayer perceptron model that will help us understand how a neural network works and how we can use it to diagnose breast cancer. For now, we are going to cover the following scripts:
- **Perceptron.py**
- **LayerPerceptron.py**
- **MultilayerPerceptron.py**

### Perceptron
**A perceptron is an artificial neuron.** They are defined by the presence of one or more input connections, an activation function and a single output. Each connection contains a weight which is learned during the training phase.<br>

Its main function is to replicate the behavior of a biological neuron. Broadly speaking, we can see that a neuron can be divided into four parts:
- **Dendrites:** where a neuron receives input from other cells.
- **Synapse:** points of contact between neurons where information is passed from one neuron to the next.
- **Soma:** where the signals from the dendrites are joined and passed on.
- **Axons:** where electrical impulses from the neuron travel away to be received by other neurons.

<div align="center">
<img width=700 alt="biological neuron" src="https://user-images.githubusercontent.com/74931024/177190004-06cefb93-511e-425d-b5a5-4d1b5afceaca.jpg">
</div>

Now that we understand how a biological neuron works, we can find its similarities to an artificial neuron:
- **Input ~ Dendrites**
- **Weights ~ Synapse**
- **Soma ~ Node**
- **Output ~ Axon**

<div align="center">
<img width=700 alt="artificial neuron" src="https://user-images.githubusercontent.com/74931024/177190161-47bec551-5910-40d0-9a39-6b94380da046.png">
</div>
