## 📘 Introduction

This study presents the training of one of the largest convolutional neural networks (CNNs) at the time, applied to subsets of the ImageNet dataset used in the ILSVRC-2010 and ILSVRC-2012 competitions. The model was set to classify 1.2 million high-resolution images into 1,000 distinct categories.

A central focus of the work was the idea that improving object recognition in machine learning requires:  
1️⃣ Access to larger datasets  
2️⃣ Training more powerful models  
3️⃣ Computational Limitations  
4️⃣ Preventing overfitting  

---

### 🔹 1. Access to larger datasets

The first challenge addressed is the high variability of objects in real-world settings. Effective recognition in such environments necessitates much larger training sets. Only recently has it become feasible to collect labeled datasets with millions of images, such as ImageNet.

The full ImageNet dataset contains over 15 million labeled high-resolution images across more than 22,000 categories. A subset of this dataset was used in the present study.

---

### 🔹 2. Training more powerful models

The second challenge involves the need for high-capacity models to learn from large and complex datasets. CNNs are well-suited for this because:
- ✅ Their capacity can be adjusted by modifying depth and width
- ✅ They make strong and often accurate assumptions about the spatial structure of images

Advantages:
- 📉 CNNs have significantly fewer parameters and connections than fully connected networks, making them easier to train

Limitations:
- 💸 Applying CNNs to large-scale, high-resolution images remains computationally expensive

---

### 🔹 3. Computational Limitations: Multi-GPU Training

The computational expense of CNNs leads to the third challenge: the need for multi-GPU training, as managing the demands of such a large network requires distributing the workload across multiple GPUs.

---

### 🔹 4. Preventing overfitting 

The fourth challenge is overfitting, which became a significant concern due to the network’s size, even with 1.2 million labeled training examples.  
To address this, regularization techniques such as Dropout and Data augmentation were used to improve the model’s generalization ability.

---

### 🧱 Final Architecture

The final architecture consists of five convolutional layers followed by three fully connected layers. Notably:
- Each convolutional layer accounts for less than 1% of total parameters
- Removing any one of them led to degraded performance, highlighting the importance of depth

---

### ⚙️ How can the network be improved?

Ultimately, the model’s size was limited by:
- 🧠 Available GPU memory
- ⏱️ Acceptable training time

The authors suggest that further improvements could be achieved with faster GPUs and larger datasets in the future.

---

### 🎯 Study Goal

The primary aim of Krizhevsky et al. (2012) was to demonstrate that deep CNNs, when trained on large-scale datasets with the help of GPU acceleration and Regularization strategies can dramatically outperform traditional image classification methods.
