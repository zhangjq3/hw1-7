This assignment is due on **Wednesday, Sept 30 2020** at 11:59pm Beijing Time.

<details>
<summary>Handy Download Links</summary>

 <ul>
  <li><a href="{{ site.hw_1_jupyter }}">Jupyter starter code</a></li>
</ul>

</details>

- [Goals](#goals)
- [Setup](#setup)
  - [Local Development](#option-b-local-development)
- [Q0: Perceptron algorithm (15 points)](#q0-perceptron-learning-20-points)
- [Q1: k-Nearest Neighbor classifier (20 points)](#q1-k-nearest-neighbor-classifier-20-points)
- [Q2: Training a Support Vector Machine (20 points)](#q2-training-a-support-vector-machine-25-points)
- [Q3: Implement a Softmax classifier (20 points)](#q3-implement-a-softmax-classifier-20-points)
- [Q4: Two-Layer Neural Network (20 points)](#q4-two-layer-neural-network-25-points)
- [Q5: Higher Level Representations: Image Features (10 points)](#q5-higher-level-representations-image-features-10-points)
- [Submitting your work](#submitting-your-work)

### Goals

In this assignment you will practice putting together a simple image classification pipeline based on the k-Nearest Neighbor or the SVM/Softmax classifier. The goals of this assignment are as follows:

- Understand the basic **Image Classification pipeline** and the data-driven approach (train/predict stages)
- Understand the train/val/test **splits** and the use of validation data for **hyperparameter tuning**.
- Develop proficiency in writing efficient **vectorized** code with numpy
- Implement and apply a k-Nearest Neighbor (**kNN**) classifier
- Implement and apply a Multiclass Support Vector Machine (**SVM**) classifier
- Implement and apply a **Softmax** classifier
- Implement and apply a **Two layer neural network** classifier
- Understand the differences and tradeoffs between these classifiers
- Get a basic understanding of performance improvements from using **higher-level representations** as opposed to raw pixels, e.g. color histograms, Histogram of Gradient (HOG) features, etc.

### Setup

**Download.** Starter code containing jupyter notebooks will be released through git classroom ({{site.hw_1_jupyter}}).

**Install Packages**. Once you have the starter code, activate your environment (the one you installed in the [Software Setup]({{site.baseurl}}/setup-instructions/) page) and run `pip install -r requirements.txt`.

**Download CIFAR-10**. Next, you will need to download the CIFAR-10 dataset. Run the following from the `assignment1` directory:

```bash
cd cs231n/datasets
./get_datasets.sh
```
**Start Jupyter Server**. After you have the CIFAR-10 data, you should start the Jupyter server from the
`assignment1` directory by executing `jupyter notebook` in your terminal.

Complete each notebook, then once you are done, go to the [submission instructions](#submitting-your-work).

### Q0: Perceptron Learning Algorithm (15 points)

The notebook **perceptron.ipynb** will walk you through implementing the perceptron algorithm.

### Q1: k-Nearest Neighbor classifier (20 points)

The notebook **knn.ipynb** will walk you through implementing the kNN classifier.

### Q2: Training a Support Vector Machine (20 points)

The notebook **svm.ipynb** will walk you through implementing the SVM classifier.

### Q3: Implement a Softmax classifier (20 points)

The notebook **softmax.ipynb** will walk you through implementing the Softmax classifier.

### Q4: Two-Layer Neural Network (20 points)

The notebook **two\_layer\_net.ipynb** will walk you through the implementation of a two-layer neural network classifier.

### Q5: Higher Level Representations: Image Features (10 points)

The notebook **features.ipynb** will examine the improvements gained by using higher-level representations
as opposed to using raw pixel values.

### Submitting your work

**Important**. Please make sure that the submitted notebooks have been run and the cell outputs are visible.

Please follow the instructions on Piazza to submit your homework.
