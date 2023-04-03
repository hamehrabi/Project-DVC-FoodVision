# Project: An End to end FoodVision101 using CNN with VGG16&EfficientnetB1 + DVC
## ✨ Project information:
An end-to-end CNN Image Classification Model was developed using Transfer Learning to identify food items in images. The popular EfficientnetB1 model, which had been pretrained on the large Food101 dataset, was employed and retrained for the project's purposes. Remarkably, the [DeepFood](https://arxiv.org/abs/1606.05675) Paper's model, which had an accuracy of 77.4% and was also trained on Food101, was outperformed by the Model developed in this project. The project uses DVC (data version control) for managing data. It is built on a microservices architecture and is an end-to-end project. The dataset can be downloaded from this [link](https://drive.google.com/file/d/1-KYL8N8oQ8HkaqSp4pYlKlsU1urQb8wt/view?usp=share_link).

> **Dataset :** `Food101`

> **Model :** `EfficientNetB1 & VGG16`
The project's model will be built using all of the data from the Food101 dataset, comprising 75,750 training images and 25,250 testing images.

Two methods to significantly improve the speed of the model training:
*Prefetching
*Mixed precision training

### **Checking the GPU**

For this Project we will working with **Mixed Precision**. And mixed precision works best with a with a GPU with compatibility capacity **7.0+**.

At the time of writing, colab offers the following GPU's :
* Nvidia K80
* **Nvidia T4**
* Nvidia P100

Colab allocates a random GPU everytime we factory reset runtime. So you can reset the runtime till you get a **Tesla T4 GPU** as T4 GPU has a rating 7.5.

> In case using local hardware, use a GPU with rating 7.0+ for better results.

## 📚 **Libraries used** :
* Tensorflow
* tfds
* Keras
* pandas
* numpy
* seaborn
* os
* DVC
* 
## **Preprocessing the Data**

Since we've downloaded the data from TensorFlow Datasets, there are a couple of preprocessing steps we have to take before it's ready to model. 

More specifically, our data is currently:

* In `uint8` data type
* Comprised of all differnet sized tensors (different sized images)
* Not scaled (the pixel values are between 0 & 255)

Whereas, models like data to be:

* In `float32` data type
* Have all of the same size tensors (batches require all tensors have the same shape, e.g. `(224, 224, 3)`)
* Scaled (values between 0 & 1), also called normalized

To take care of these, we'll create a `preprocess_img()` function which:

* Resizes an input image tensor to a specified size using [`tf.image.resize()`](https://www.tensorflow.org/api_docs/python/tf/image/resize)
* Converts an input image tensor's current datatype to `tf.float32` using [`tf.cast()`](https://www.tensorflow.org/api_docs/python/tf/cast)

## **Building the Model : EfficientNetB1**

Implemented Mixed Precision training and Prefetching to decrease the time taken for the model to train.

### **Getting the Callbacks ready**
As we are dealing with a complex Neural Network (EfficientNetB0) its a good practice to have few call backs set up. Few callbacks I will be using throughtout this Notebook are :
 * **TensorBoard Callback :** TensorBoard provides the visualization and tooling needed for machine learning experimentation

 * **EarlyStoppingCallback :** Used to stop training when a monitored metric has stopped improving.
 
 * **ReduceLROnPlateau :** Reduce learning rate when a metric has stopped improving.
 
### Evaluating the results
 
### Loss vs Epochs
 
![image](https://user-images.githubusercontent.com/61462986/202082223-83c3a8f2-26c9-455e-97d5-ee833a4b10cc.png)

### Accuracy vs Epochs

![image](https://user-images.githubusercontent.com/61462986/202082253-0d28ea8e-72af-4182-bf79-33b4119f27ef.png)



## 🚀 Project structure (MLOps-DVC):
<img src="https://user-images.githubusercontent.com/62473531/229567697-a85f4b6e-80e6-48ef-8220-c4ef320ef395.png" alt="workflow" width="70%">
<img src="https://user-images.githubusercontent.com/62473531/229579230-f671e5ee-2206-4816-af0a-40226b89d36b.JPG" alt="workflow" width="70%">

### 🔥 Technologies Used:
``` 
1. Python 
2. shell scripting 
3. aws cloud Provider 
4. DVC
```

## 👷 Initial Setup: 
```commandline
conda create --prefix ./env python=3.9
conda activate ./env 
pip install -r requirements.txt
dvc init
```

## Conclusion<a id='conclusion-'></a>
This project is production ready to be used for the similar use cases and it will provide the automated and orchesrated production ready pipelines(Training & Serving)
#### **Thanks for taking a look at this project. If you find it valuable, kindly rate it by clicking the star icon. Your support is highly appreciated! 😊🙏⭐**<br><br>

#### **📃 License**
MIT license ©
My Website **[``website``](https://)** <br>
Let's connect on **[``LinkedIn``](https://www.linkedin.com/in/hamed-mehrabi/)** <br>
