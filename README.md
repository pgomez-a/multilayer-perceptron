# multilayer-perceptron

<img width="850" alt="multilayer-perceptron" src="https://user-images.githubusercontent.com/74931024/176994282-d2ff2103-87ac-4723-a90e-da0dd16ad4aa.png">

**If you want to learn more about IT topics, [I invite you to join my Patreon channel](https://www.patreon.com/pgomeza) and visit my website:** [**IA Notes**](https://ia-notes.com/)

With this project we are going to implement a **multilayer perceptron**, in order to predict whether a cancer is malignant or benign on a dataset of breast cancer diagnosis in the Wisconsin. The goal of this project is to give us a first approach to **artificial neural networks**, and to made us implement the algorithms at the heart of the training process.

<img align="right" alt="breast cancer" src="https://user-images.githubusercontent.com/74931024/177033104-c876fd69-af55-453d-9ea8-950c2317f29a.png">

### Introduction
**Breast cancer** is cancer that develops from **breast tissue**, most commonly in the cells that line the milk ducts of the breast. Breast cancer happens when abnormal cells in the breast begin to grow and divide in an uncontrolled way and eventually form a tumour. It mainly affects women, but men can get it too.<br>

Around **70%** of breast cancers have no special features when the cells are looked at under the microscope. They are called invasive breast cancer no special type (NST). **Mammography** has assisted in reducing breast cancer-related deaths by providing early detection when cancer is still treatable. However, it is less sensitive in women with extremely dense breast tissue than fatty breast tissue. Additionally, women with extremely dense breasts are three to six times more likely to develop breast cancer than women with almost entirely fatty breasts and two times more likely than the average woman.

With **Machine Learning**, medical professionals can quickly and accurately sort through breast MRIs in patients with dense breast tissue to eliminate those without cancer. **Supplemental screening in women with dense breast tissue increased the sensitivity of cancer detection.**

### Objective
To carry out this project, we are going to train an **[artificial neural network](https://github.com/pgomez-a/multilayer-perceptron/tree/main/lab)** that will be able **to detect breast cancer.** The features of the dataset describe the characteristics of a breast mass cell nucleus extracted with fine-needle aspiration. In this way, we are going to go through two major stages to complete our project:
- **[Data Analysis](https://github.com/pgomez-a/multilayer-perceptron/tree/main/data_analysis)**
  - Data Cleaning
  - Data Visualization
- **Model Implementation**
  - Model Training
  - Model Evaluation


### How to use it?
The **training program** will use **backpropagation** and **gradient descent** to learn on the training dataset and will save the model (network topology and weights) at the end of its execution:

    python train_network.py

<img width="1474" alt="learning_curve" src="https://user-images.githubusercontent.com/74931024/178154668-4e4b6dbb-160e-4bcc-a2e1-d0fc0b4c9900.png">

The **prediction program** will load the weights learned in the previous phase, perform a prediction on a given set (which will also be loaded), then evaluate it using the **[binary cross-entropy error function](https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_error_function_and_logistic_regression)**:

    python evaluate_network.py

<img width="870" alt="evaluate_network" src="https://user-images.githubusercontent.com/74931024/178147374-230efff9-5dfc-4c3a-b019-1f24b59b8d4a.png">
