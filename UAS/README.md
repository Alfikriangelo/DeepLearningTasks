Berikut adalah analisis dari masing-masing file Jupyter Notebook

### **CHAPTER_1_The_Machine_Learning_Landscape.ipynb**

Notebook ini memberikan pengenalan lanskap Machine Learning. Ini dimulai dengan memuat dataset `oecd_bli_2015.csv` dan `gdp_per_capita.csv` untuk memprediksi kepuasan hidup berdasarkan PDB per kapita. Sebuah model regresi linear sederhana dari Scikit-Learn digunakan untuk membuat prediksi ini.

**Topik utama yang dibahas:**

* **Apa itu Machine Learning:** Notebook ini kemungkinan besar mendefinisikan Machine Learning dan memberikan contoh aplikasinya.
* **Jenis-jenis Sistem Machine Learning:** Menjelaskan berbagai kategori sistem Machine Learning, termasuk:
    * Supervised/Unsupervised Learning
    * Batch dan Online Learning
    * Instance-Based dan Model-Based Learning
* **Tantangan Utama dalam Machine Learning:** Membahas tantangan umum seperti "Kualitas Data yang Buruk", "Data Pelatihan yang Tidak Relevan", "Overfitting", dan "Underfitting".
* **Testing dan Validasi:** Menjelaskan pentingnya pengujian dan validasi model Machine Learning.

### **CHAPTER_2_End_to_End_Machine_Learning_Project.ipynb**

Notebook ini menyajikan panduan langkah demi langkah untuk proyek Machine Learning end-to-end menggunakan dataset perumahan California.

**Langkah-langkah proyek yang dibahas:**

* **Melihat Gambaran Besar:** Mendefinisikan masalah dan tujuan proyek.
* **Mendapatkan Data:** Menunjukkan cara mengunduh dan memuat data.
* **Menjelajahi dan Memvisualisasikan Data:** Menganalisis dan memvisualisasikan data untuk mendapatkan wawasan.
* **Mempersiapkan Data:** Mencakup pembersihan data, penanganan atribut teks dan kategorikal, serta transformasi kustom.
* **Memilih dan Melatih Model:** Memilih dan melatih model regresi linear dan model-model lain.
* **Fine-Tuning Model:** Menjelaskan teknik fine-tuning seperti Grid Search, Randomized Search, dan Ensemble Methods.
* **Menganalisis Model Terbaik dan Kesalahannya:** Mengevaluasi model terbaik dan menganalisis kesalahannya.
* **Mengevaluasi Sistem pada Test Set:** Mengukur kinerja akhir model pada test set.
* **Menjalankan, Memantau, dan Memelihara Sistem:** Membahas langkah-langkah setelah model dilatih.

### **CHAPTER_3_Classification.ipynb**

Notebook ini berfokus pada berbagai aspek klasifikasi dalam Machine Learning, menggunakan dataset MNIST sebagai contoh utama.

**Konsep-konsep penting yang dicakup:**

* **Dataset MNIST:** Pengenalan dataset MNIST yang berisi gambar-gambar angka tulisan tangan.
* **Melatih Klasifikasi Biner:** Menunjukkan cara melatih *binary classifier* untuk mendeteksi satu digit, misalnya angka 5.
* **Ukuran Kinerja:** Membahas berbagai metrik untuk mengevaluasi kinerja model klasifikasi, termasuk:
    * Akurasi menggunakan *cross-validation*.
    * *Confusion Matrix*.
    * *Precision* dan *Recall*.
    * Kurva ROC.
* **Klasifikasi Multikelas:** Menjelaskan cara kerja klasifikasi untuk lebih dari dua kelas.
* **Analisis Kesalahan:** Cara menganalisis kesalahan yang dibuat oleh model.
* **Klasifikasi Multilabel dan Multioutput:** Pengenalan singkat tentang jenis klasifikasi ini.

### **CHAPTER_4_Training_Models.ipynb**

Notebook ini membahas berbagai metode untuk melatih model Machine Learning, dengan fokus pada regresi.

**Metode-metode yang dibahas:**

* **Regresi Linear:**
    * Menggunakan *Normal Equation* untuk solusi bentuk tertutup.
    * Menggunakan *Gradient Descent* (Batch, Stochastic, dan Mini-batch).
* **Regresi Polinomial:** Menunjukkan cara menggunakan regresi polinomial untuk data non-linear.
* **Kurva Belajar:** Cara menggunakan kurva belajar untuk mendiagnosis *overfitting* dan *underfitting*.
* **Regularisasi Model Linear:**
    * Regresi Ridge
    * Regresi Lasso
    * Elastic Net
* **Early Stopping:** Teknik regularisasi dengan menghentikan pelatihan lebih awal.
* **Regresi Logistik:**
    * Estimasi probabilitas.
    * Batas keputusan.
    * Regresi Softmax untuk klasifikasi multikelas.

### **CHAPTER_5_Support_Vector_Machines.ipynb**

Notebook ini memberikan pengenalan tentang *Support Vector Machines* (SVM), sebuah model Machine Learning yang kuat.

**Topik-topik yang dibahas:**

* **Klasifikasi Linear SVM:**
    * *Soft Margin Classification*.
* **Klasifikasi Non-linear SVM:**
    * Menggunakan fitur polinomial.
    * Menggunakan *kernel trick* (Polinomial dan Gaussian RBF).
* **Regresi SVM.**
* **Cara Kerja SVM (Under the Hood).**

### **CHAPTER_6_Decision_Trees.ipynb**

Notebook ini menjelaskan cara kerja, penggunaan, dan manfaat dari *Decision Trees*.

**Poin-poin utama:**

* **Pelatihan dan Visualisasi:** Menunjukkan cara melatih dan memvisualisasikan *Decision Tree*.
* **Membuat Prediksi:** Menjelaskan bagaimana *Decision Tree* membuat prediksi.
* **Estimasi Probabilitas Kelas.**
* **Algoritma Pelatihan CART.**
* **Kompleksitas Komputasi.**
* **Gini Impurity vs. Entropy.**
* **Regularisasi Hyperparameters.**
* **Regresi.**
* **Ketidakstabilan.**

### **CHAPTER_7_Ensemble_Learning_and_Random_Forests.ipynb**

Notebook ini berfokus pada teknik *Ensemble Learning* dan *Random Forests*.

**Teknik-teknik yang dibahas:**

* **Voting Classifiers:** Menggabungkan prediksi dari beberapa *classifier*.
* **Bagging dan Pasting:** Menggunakan beberapa *classifier* yang dilatih pada subset data yang berbeda.
* **Random Forests:** Sebuah *ensemble* dari *Decision Trees*.
* **Boosting:**
    * AdaBoost
    * Gradient Boosting
* **Stacking.**

### **CHAPTER_8_Dimensionality_Reduction.ipynb**

Notebook ini membahas masalah "kutukan dimensi" dan berbagai teknik untuk menguranginya.

**Teknik-teknik yang dibahas:**

* **Pendekatan Utama untuk Pengurangan Dimensi:**
    * Proyeksi
    * Manifold Learning
* **PCA (Principal Component Analysis):**
    * Menjaga varians.
    * Komponen utama.
    * Memproyeksikan ke *d* dimensi.
    * Menggunakan Scikit-Learn.
    * Rasio varians yang dijelaskan.
    * Memilih jumlah dimensi yang tepat.
    * PCA untuk kompresi.
    * Randomized PCA.
    * Incremental PCA.
* **Kernel PCA.**
* **LLE (Locally Linear Embedding).**
* **Teknik Pengurangan Dimensi Lainnya.**

### **CHAPTER_9_Unsupervised_Learning_Techniques.ipynb**

Notebook ini membahas teknik-teknik *unsupervised learning*, terutama *clustering*.

**Algoritma yang dibahas:**

* **Clustering:**
    * **K-Means:**
        * Metode inisialisasi *centroid*.
        * Menemukan jumlah *cluster* yang optimal.
        * Batasan K-Means.
        * Menggunakan *clustering* untuk pra-pemrosesan.
    * **DBSCAN.**
    * **Algoritma Clustering Lainnya.**
* **Gaussian Mixtures.**

### **CHAPTER_10_Introduction_to_Artificial_Neural_Networks_with_Keras (1).ipynb**

Notebook ini memberikan pengenalan tentang Jaringan Saraf Tiruan (ANN) menggunakan Keras, dengan fokus pada pembuatan *image classifier*.

**Topik-topik utama:**

* **Membangun Image Classifier menggunakan Sequential API:**
    * Membuat model menggunakan Sequential API.
    * Meng-compile model.
    * Melatih dan mengevaluasi model.
    * Menggunakan model untuk membuat prediksi.
* **Membangun MLP Regresi menggunakan Sequential API.**
* **Membangun Model Kompleks menggunakan Functional API.**
* **Membangun Model Dinamis menggunakan Subclassing API.**
* **Menyimpan dan Mengembalikan Model.**
* **Menggunakan Callback.**
* **Menggunakan TensorBoard untuk Visualisasi.**
* **Fine-Tuning Hyperparameters Neural Network.**

### **CHAPTER_11_Training_Deep_Neural_Networks.ipynb**

Notebook ini membahas masalah-masalah yang sering muncul saat melatih *deep neural networks* (DNN) dan cara mengatasinya.

**Topik utama yang dibahas:**

* **Vanishing/Exploding Gradients Problem:** Menjelaskan masalah gradien yang hilang atau meledak dan solusinya, seperti inisialisasi Xavier dan He.
* **Fungsi Aktivasi Nonsaturating:** Memperkenalkan berbagai fungsi aktivasi seperti ReLU, Leaky ReLU, ELU, dan SELU.
* **Batch Normalization:** Menjelaskan cara kerja dan implementasi *Batch Normalization* di Keras.
* **Gradient Clipping:** Teknik untuk mengatasi gradien yang meledak dengan memotongnya.
* **Transfer Learning:** Menjelaskan konsep menggunakan kembali lapisan dari model yang sudah ada untuk tugas baru.
* **Faster Optimizers:** Membahas berbagai *optimizer* yang lebih cepat dari *Gradient Descent* biasa, seperti Momentum, Nesterov Accelerated Gradient, AdaGrad, RMSProp, dan Adam.
* **Learning Rate Scheduling:** Menjelaskan berbagai strategi untuk menyesuaikan *learning rate* selama pelatihan.
* **Menghindari Overfitting Melalui Regularisasi:** Membahas berbagai teknik regularisasi seperti ℓ1 dan ℓ2, *dropout*, dan *max-norm*.

### **CHAPTER_12_Custom_Models_and_Training_with_TensorFlow.ipynb**

Notebook ini memberikan pengenalan yang lebih dalam tentang TensorFlow, menunjukkan cara menggunakannya seperti NumPy dan cara membuat model serta *training loop* kustom.

**Topik utama yang dibahas:**

* **Menggunakan TensorFlow seperti NumPy:** Menunjukkan cara membuat dan memanipulasi tensor, serta interoperabilitasnya dengan NumPy.
* **Model dan Lapisan Kustom:** Menjelaskan cara membuat lapisan, model, dan fungsi kerugian kustom di TensorFlow.
* ***Training Loop* Kustom:** Menunjukkan cara membuat *training loop* kustom untuk kontrol yang lebih besar atas proses pelatihan.
* **Fungsi dan Grafik TensorFlow:** Menjelaskan cara kerja `tf.function` untuk mempercepat kode Python.

### **CHAPTER_13_Loading_and_Preprocessing_Data_with_TensorFlow.ipynb**

Notebook ini berfokus pada cara efisien untuk memuat dan memproses data menggunakan TensorFlow.

**Topik utama yang dibahas:**

* **Data API:** Pengenalan `tf.data` API untuk membuat *input pipeline* yang efisien.
* **TFRecord Format:** Menjelaskan format TFRecord untuk menyimpan dan membaca data dalam jumlah besar.
* **Preprocessing Input Features:** Menunjukkan cara melakukan pra-pemrosesan fitur menggunakan Keras.
* **TF Transform:** Pengenalan singkat tentang pustaka `tf.Transform` untuk pra-pemrosesan data.
* **TensorFlow Datasets (TFDS) Project:** Memperkenalkan proyek TFDS untuk mengunduh dataset standar dengan mudah.

### **CHAPTER_14_Deep_Computer_Vision_Using_Convolutional_Neural_Networks.ipynb**

Notebook ini membahas *Convolutional Neural Networks* (CNN) untuk tugas-tugas *computer vision*.

**Topik utama yang dibahas:**

* **Lapisan Konvolusional:** Menjelaskan cara kerja filter, *stride*, dan *padding*.
* **Lapisan Pooling:** Membahas *max pooling* dan *average pooling*.
* **Arsitektur CNN:** Membahas berbagai arsitektur CNN klasik seperti LeNet-5, AlexNet, GoogLeNet, VGGNet, ResNet, Xception, dan DenseNet.
* **Klasifikasi dan Lokalisasi:** Menjelaskan cara melakukan klasifikasi dan lokalisasi objek dalam gambar.
* **Deteksi dan Segmentasi Objek:** Pengenalan singkat tentang deteksi objek dan segmentasi.

### **CHAPTER_15_Processing_Sequences_Using_RNNs_and_CNNs.ipynb**

Notebook ini berfokus pada pemrosesan sekuens menggunakan *Recurrent Neural Networks* (RNN) dan CNN.

**Topik utama yang dibahas:**

* **RNN Dasar:** Menjelaskan cara kerja RNN dan cara mengimplementasikannya di Keras.
* **Melatih RNN:** Membahas tantangan dalam melatih RNN, seperti *vanishing/exploding gradients*.
* **RNN Lanjutan:** Memperkenalkan sel LSTM dan GRU untuk mengatasi masalah gradien.
* **Natural Language Processing (NLP):** Menjelaskan cara menggunakan RNN untuk tugas-tugas NLP.

### **CHAPTER_16_Natural_Language_Processing_with_RNNs_and_Attention.ipynb**

Notebook ini melanjutkan pembahasan tentang NLP, dengan fokus pada mekanisme atensi dan model-model canggih.

**Topik utama yang dibahas:**

* **Char-RNN:** Membangkitkan teks Shakespeare menggunakan RNN tingkat karakter.
* **Analisis Sentimen:** Melakukan analisis sentimen pada ulasan film IMDb.
* **Encoder-Decoder Network untuk Terjemahan Neural Machine (NMT):** Menjelaskan arsitektur *encoder-decoder* untuk tugas terjemahan.
* **Mekanisme Atensi:** Memperkenalkan mekanisme atensi untuk meningkatkan kinerja model *encoder-decoder*.
* **Model Transformer:** Pengenalan arsitektur Transformer yang populer.
* **Riwayat NLP:** Memberikan gambaran singkat tentang perkembangan NLP.

### **CHAPTER_17_Representation_Learning_and_Generative_Learning_Using_Autoencoders_and_GANs.ipynb**

Notebook ini membahas *representation learning* dan *generative learning* menggunakan *autoencoder* dan *Generative Adversarial Networks* (GAN).

**Topik utama yang dibahas:**

* **Autoencoder:**
    * PCA dengan *undercomplete linear autoencoder*.
    * *Stacked Autoencoder*.
    * *Denoising Autoencoder*.
    * *Sparse Autoencoder*.
    * *Variational Autoencoder* (VAE).
* **Generative Adversarial Networks (GAN):**
    * Menjelaskan cara kerja GAN.
    * *Deep Convolutional GAN* (DCGAN).
    * *Progressive Growing of GANs* (PGAN).
    * *StyleGAN*.

### **CHAPTER_18_Reinforcement_Learning.ipynb**

Notebook ini memberikan pengenalan tentang *Reinforcement Learning* (RL).

**Topik utama yang dibahas:**

* **Pengenalan OpenAI Gym:** Memperkenalkan OpenAI Gym sebagai lingkungan untuk melatih agen RL.
* **Strategi Neural Network:** Menjelaskan cara menggunakan *neural network* sebagai kebijakan (*policy*).
* **Algoritma RL:**
    * *Policy Gradients*.
    * *Markov Decision Processes* (MDP).
    * *Q-Learning* dan *Deep Q-Network* (DQN).
* **TF-Agents Library:** Pengenalan singkat tentang pustaka TF-Agents untuk RL.

### **CHAPTER_19_Training_and_Deploying_TensorFlow_Models_at_Scale.ipynb**

Notebook ini membahas cara melatih dan menerapkan model TensorFlow dalam skala besar.

**Topik utama yang dibahas:**

* **Menyajikan Model TensorFlow:**
    * Menggunakan TensorFlow Serving.
    * Menerapkan model di Google AI Platform.
* **Menerapkan Model di Perangkat Seluler atau Tertanam.**
* **Menggunakan GPU untuk Mempercepat Komputasi:** Menjelaskan cara menggunakan GPU tunggal, beberapa GPU pada satu mesin, dan beberapa GPU di beberapa mesin.
* **Paralelisme Data dan Model.**
* **Melatih Model di AI Platform.**
