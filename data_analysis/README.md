# Data Analysis
Before implementing our **Machine Learning** model, we must first understand the available data. This is a very important step that we have to take if we want to develop a model that we can fully understand. In fact, to complete this project, we are provided with a dataset with raw data. In the **[/dataset/](https://github.com/pgomez-a/multilayer-perceptron/tree/main/datasets)** folder we can find this dataset called **dataset_raw.csv**.<br>

### Data Cleaning
If we read **dataset_raw.csv**, we see that this file contains a lot of numbers that don't seem to make sense. So our first task is to give this raw data a format that helps us understand the available data. This process is called data cleaning. It consists of reading the **dataset_info.txt** file to understand what each row and each column represents.<br>

Once we understand this, we can format our **dataset_raw.csv to dataset_clean.csv**. In our case, this format has been given following the below information:
- **Tittle:** [Wisconsin Diagnostic Breast Cancer (WDBC)](https://ftp.cs.wisc.edu/math-prog/cpo-dataset/machine-learn/cancer/WDBC/)
- **Date:** November 1995
- **Relevant Information:** features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.
- **Number of instances:** 569
- **Number of attributes:** 32
  - ID number
  - Diagnosis (M = malignant, B = bening)
  - 30 real-valued input features
- **10 real-valued features:**
  - Radius - mean of distances from center to points on the perimeter.
  - Texture - standard deviation of gray-scale values.
  - Perimeter
  - Area
  - Smoothness - local variation in radius lengths.
  - Compactness - perimeter^2 / area - 1.0.
  - Concavity - severity of concave portions of the contour.
  - Concave points - number of concave portions of the contour.
  - Symmetry
  - Fractal dimension - "coastline aproximation" - 1
- **Measures:** the mean, standard error, and "worst" (mean of the three largest values) of these features were computed for each image, resulting in 30 features.
- **Class distribution:** 357 bening, 212 malignant.

Following this description, the **clean.py** script is responsible for creating the **dataset_clean.py** file, which is stored in **[/datasets/](https://github.com/pgomez-a/multilayer-perceptron/tree/main/datasets)**. This csv file is a dataset that identifies each column with a meaningful name and removes the ID number feature, as we are onnly interested in analyzing the features of cell nuclei. This dataset will be the one we will use from now on to complete our project.

    python clean.py
    
<div align="center">
<a href="https://ftp.cs.wisc.edu/math-prog/cpo-dataset/machine-learn/cancer/cancer_images/">
<img width=330 alt="benign" src="https://user-images.githubusercontent.com/74931024/177000188-35ab107b-684a-4482-838d-b951c407a009.gif">
</a>
<a href="https://ftp.cs.wisc.edu/math-prog/cpo-dataset/machine-learn/cancer/cancer_images/">
<img width=330 alt="benign" src="https://user-images.githubusercontent.com/74931024/177000273-dc005458-aa74-4ba3-88d1-37154a08ae7c.gif">
</a>
<a href="https://ftp.cs.wisc.edu/math-prog/cpo-dataset/machine-learn/cancer/cancer_images/">
<img width=330 alt="malignant" src="https://user-images.githubusercontent.com/74931024/177000299-480df067-4238-4971-bb30-4ccb939e105f.gif">
</a>
</div>

### Data Visualization
For a better understanding of the data, another important step to take is to visualize the data so that we can get an idea of the distribution of the data and the relationship between the features. To do this, we have four different scripts at our disposal:

    python histogram_mean.py
    
This program does not take any parameters. Computes a histogram for each of the features related to the measure "mean":

    python histogram_std.py
    
This program does not take any parameters. Computes a histogram for each of the features related to the measure "std":

    python histogram_worst.py
    
This program does not take any parameters. Computes a histogram for each of the features related to the measure "worst":
