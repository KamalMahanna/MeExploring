# AI Researcher Roadmap: From Fundamentals to SOTA

This roadmap is designed for individuals aiming to reach a researcher-level understanding of Artificial Intelligence. It covers foundational knowledge, core machine learning principles, deep learning specializations, and the skills required to contribute to the field.

---

## Phase 1: Foundational Knowledge

### 📚 Mathematics

- [ ] **Linear Algebra**
  - [ ] Scalars, Vectors, Matrices, and Tensors
  - [ ] Matrix Operations (multiplication, inversion, determinants)
  - [ ] Systems of Linear Equations
  - [ ] Eigenvalues & Eigenvectors
  - [ ] Singular Value Decomposition (SVD)
  - [ ] Principal Component Analysis (PCA)
- [ ] **Calculus**
  - [ ] Limits, Derivatives, and Integrals
  - [ ] Partial Derivatives
  - [ ] The Chain Rule
  - [ ] Gradient, Hessian, Jacobian
  - [ ] Optimization with Gradient Descent (and its variants: SGD, Adam, etc.)
- [ ] **Probability & Statistics**
  - [ ] Basic Probability Theory (axioms, conditional probability, Bayes' theorem)
  - [ ] Random Variables (discrete and continuous)
  - [ ] Common Distributions (Gaussian, Bernoulli, Binomial, Poisson)
  - [ ] Expectation, Variance, Covariance
  - [ ] Maximum Likelihood Estimation (MLE)
  - [ ] Bayesian Inference & Maximum A Posteriori (MAP)
  - [ ] Hypothesis Testing & A/B Testing
- [ ] **Information Theory**
  - [ ] Entropy & Cross-Entropy
  - [ ] Kullback-Leibler (KL) Divergence
  - [ ] Mutual Information

### 💻 Computer Science & Programming

- [ ] **Programming Proficiency**
  - [ ] Python (mastery, including OOP and functional programming)
  - [ ] NumPy (for numerical operations)
  - [ ] Pandas (for data manipulation)
  - [ ] Matplotlib & Seaborn (for data visualization)
- [ ] **Data Structures & Algorithms**
  - [ ] Big O Notation (time and space complexity analysis)
  - [ ] Core Data Structures (Arrays, Linked Lists, Stacks, Queues, Trees, Graphs, Hash Tables)
  - [ ] Core Algorithms (Searching, Sorting, Graph Traversal)
- [ ] **Software Engineering Practices**
  - [ ] Git & Version Control
  - [ ] Virtual Environments (conda, venv)
  - [ ] Basics of Shell Scripting
  - [ ] Docker (for reproducible research)

---

## Phase 2: Core Machine Learning

### 🧠 Traditional Models

- [ ] **Supervised Learning**
  - [ ] Linear & Polynomial Regression
  - [ ] Logistic Regression
  - [ ] k-Nearest Neighbors (k-NN)
  - [ ] Support Vector Machines (SVMs) & Kernel Methods
  - [ ] Decision Trees & Random Forests
  - [ ] Gradient Boosting Machines (XGBoost, LightGBM, CatBoost)
- [ ] **Unsupervised Learning**
  - [ ] K-Means Clustering
  - [ ] Hierarchical Clustering
  - [ ] DBSCAN
  - [ ] Dimensionality Reduction (PCA, t-SNE)
- [ ] **Core Concepts & Theory**
  - [ ] The Bias-Variance Tradeoff
  - [ ] Overfitting vs. Underfitting
  - [ ] Regularization (L1, L2, Dropout)
  - [ ] Cross-Validation Techniques
  - [ ] Evaluation Metrics (Accuracy, Precision, Recall, F1-Score, ROC/AUC, MSE)
  - [ ] Statistical Learning Theory (VC Dimension, PAC Learning) - *Theoretical*

---

## Phase 3: Deep Learning

### 🌐 Neural Networks Fundamentals

- [ ] **Building Blocks**
  - [ ] The Perceptron
  - [ ] Activation Functions (Sigmoid, Tanh, ReLU, Leaky ReLU, etc.)
  - [ ] Feedforward Neural Networks (Multi-Layer Perceptrons)
  - [ ] Backpropagation Algorithm (understanding the math)
  - [ ] Loss Functions (MSE, Cross-Entropy)
  - [ ] Optimization Algorithms (SGD, Momentum, RMSprop, Adam)
- [ ] **Deep Learning Frameworks**
  - [ ] PyTorch (recommended for research)
  - [ ] TensorFlow / Keras

### 👁️ Computer Vision (CV)

- [ ] **Core Architectures**
  - [ ] Convolutional Neural Networks (CNNs)
  - [ ] Convolution & Pooling Layers
  - [ ] Classic Architectures (LeNet, AlexNet, VGG, GoogLeNet)
  - [ ] Modern Architectures (ResNet, Inception, DenseNet)
- [ ] **Advanced CV Topics**
  - [ ] Transfer Learning & Fine-tuning
  - [ ] Object Detection (R-CNN, Fast R-CNN, Faster R-CNN, YOLO, SSD)
  - [ ] Semantic Segmentation (FCN, U-Net)
  - [ ] Instance Segmentation (Mask R-CNN)
  - [ ] Generative Models (see Phase 4)

### 🗣️ Natural Language Processing (NLP)

- [ ] **Classical NLP**
  - [ ] Text Preprocessing (Tokenization, Stemming, Lemmatization)
  - [ ] TF-IDF
  - [ ] Bag-of-Words (BoW)
- [ ] **Deep Learning for NLP**
  - [ ] Word Embeddings (Word2Vec, GloVe, FastText)
  - [ ] Recurrent Neural Networks (RNNs)
  - [ ] Vanishing/Exploding Gradient Problem
  - [ ] Long Short-Term Memory (LSTM)
  - [ ] Gated Recurrent Unit (GRU)
  - [ ] Encoder-Decoder Architectures & Seq2Seq Models

### 🎲 Reinforcement Learning (RL)

- [ ] **Fundamentals**
  - [ ] Markov Decision Processes (MDPs)
  - [ ] Bellman Equations
  - [ ] Value Iteration & Policy Iteration
- [ ] **Core Algorithms**
  - [ ] Q-Learning
  - [ ] Deep Q-Networks (DQN)
  - [ ] Policy Gradients (REINFORCE)
  - [ ] Actor-Critic Methods (A2C, A3C)

---

## Phase 4: State-of-the-Art & Advanced Topics

### 🤖 Transformer Architecture & LLMs

- [ ] **The Transformer**
  - [ ] Self-Attention Mechanism (understanding the `Q, K, V` formulation)
  - [ ] Multi-Head Attention
  - [ ] Positional Encodings
  - [ ] The "Attention Is All You Need" Paper (Vaswani et al., 2017)
- [ ] **Foundational LLMs**
  - [ ] BERT (and its variants: RoBERTa, ALBERT)
  - [ ] GPT (Generative Pre-trained Transformer) family (GPT-2, GPT-3, GPT-4)
  - [ ] T5 (Text-to-Text Transfer Transformer)
- [ ] **Advanced LLM Concepts**
  - [ ] Prompt Engineering & In-Context Learning
  - [ ] Fine-tuning vs. Instruction Tuning vs. RLHF
  - [ ] Parameter-Efficient Fine-Tuning (PEFT): LoRA, QLoRA
  - [ ] Quantization and Model Compression
  - [ ] Open-Source LLMs (Llama, Mistral, Falcon)

### ✨ Generative AI

- [ ] **Generative Adversarial Networks (GANs)**
  - [ ] Original GAN Paper (Goodfellow et al., 2014)
  - [ ] DCGAN, WGAN, StyleGAN
- [ ] **Variational Autoencoders (VAEs)**
  - [ ] The Reparameterization Trick
- [ ] **Diffusion Models**
  - [ ] Denoising Diffusion Probabilistic Models (DDPM)
  - [ ] Latent Diffusion Models (e.g., Stable Diffusion)
  - [ ] Classifier-Free Guidance

### 🔬 Other SOTA Frontiers

- [ ] **Multimodal Models** (e.g., CLIP, Flamingo, GPT-4V)
- [ ] **Graph Neural Networks (GNNs)**
- [ ] **Self-Supervised Learning** (e.g., SimCLR, MoCo)
- [ ] **AI Alignment & Safety**
- [ ] **Efficient AI** (making models smaller, faster, and cheaper to run)

---

## Phase 5: Becoming a Researcher

- [ ] **Reading & Understanding Papers**
  - [ ] Learn to dissect a paper: Abstract, Introduction, Methods, Results, Conclusion
  - [ ] Follow top conferences (NeurIPS, ICML, ICLR, CVPR, ICCV, ACL, EMNLP)
  - [ ] Use tools like arXiv, Papers with Code, and Google Scholar
- [ ] **Implementation & Experimentation**
  - [ ] Re-implement important papers from scratch
  - [ ] Develop a rigorous experimentation workflow (tracking, logging with tools like Weights & Biases or MLflow)
  - [ ] Learn to design and run ablation studies
- [ ] **Communication & Contribution**
  - [ ] Write clear, concise technical reports or blog posts
  - [ ] Contribute to open-source projects
  - [ ] Formulate a research question and hypothesis
  - [ ] (Goal) Publish your first paper

---

## Expanded Algorithm Appendix

This section provides a more exhaustive list of algorithms, including specialized, historical, and less common (but still relevant) techniques. It's designed to be a reference for deeper exploration once you are comfortable with the core roadmap.

### 🧠 Expanded Supervised Learning

- [ ] **Regression**
  - [ ] Ridge Regression
  - [ ] Lasso Regression
  - [ ] ElasticNet Regression
  - [ ] Bayesian Linear Regression
  - [ ] Support Vector Regression (SVR)
  - [ ] Isotonic Regression
- [ ] **Classification**
  - [ ] Naive Bayes (Gaussian, Multinomial, Bernoulli)
  - [ ] Linear Discriminant Analysis (LDA)
  - [ ] Quadratic Discriminant Analysis (QDA)
  - [ ] AdaBoost
- [ ] **Ensemble Methods**
  - [ ] Bagging (Bootstrap Aggregating)
  - [ ] Extremely Randomized Trees (ExtraTrees)
  - [ ] Stacking & Blending

### 🧠 Expanded Unsupervised Learning

- [ ] **Clustering**
  - [ ] Gaussian Mixture Models (GMM)
  - [ ] OPTICS
  - [ ] BIRCH
  - [ ] Affinity Propagation
  - [ ] Spectral Clustering
- [ ] **Dimensionality Reduction / Manifold Learning**
  - [ ] Independent Component Analysis (ICA)
  - [ ] Factor Analysis
  - [ ] Multidimensional Scaling (MDS)
  - [ ] Locally Linear Embedding (LLE)
  - [ ] ISOMAP
- [ ] **Association Rule Learning**
  - [ ] Apriori Algorithm
  - [ ] Eclat Algorithm

### 🎲 Expanded Reinforcement Learning

- [ ] **Value-Based**
  - [ ] Double DQN
  - [ ] Dueling DQN
- [ ] **Policy-Based**
  - [ ] Trust Region Policy Optimization (TRPO)
  - [ ] Proximal Policy Optimization (PPO)
  - [ ] Asynchronous Advantage Actor-Critic (A3C)
- [ ] **Model-Based**
  - [ ] Dyna-Q
- [ ] **Advanced Topics**
  - [ ] Soft Actor-Critic (SAC)
  - [ ] Deep Deterministic Policy Gradient (DDPG)
  - [ ] Multi-Agent RL (MARL)
  - [ ] Hierarchical RL

### 🧬 Evolutionary & Genetic Algorithms

- [ ] **Core Concepts**
  - [ ] Genetic Algorithm (Selection, Crossover, Mutation)
  - [ ] Evolution Strategies (ES)
  - [ ] Genetic Programming
- [ ] **Related Algorithms**
  - [ ] Particle Swarm Optimization (PSO)
  - [ ] Ant Colony Optimization (ACO)
  - [ ] Differential Evolution

### 🏛️ Symbolic AI & Classical Methods

- [ ] **Search Algorithms**
  - [ ] Breadth-First Search (BFS)
  - [ ] Depth-First Search (DFS)
  - [ ] A* Search Algorithm
  - [ ] Minimax Algorithm (for games)
  - [ ] Alpha-Beta Pruning
- [ ] **Logic & Reasoning**
  - [ ] Propositional Logic
  - [ ] First-Order Logic
  - [ ] Expert Systems
  - [ ] Constraint Satisfaction Problems (CSP)

### 🔬 Other Niche & Specialized Models

- [ ] **Bayesian Methods**
  - [ ] Bayesian Networks
  - [ ] Hidden Markov Models (HMM)
  - [ ] Kalman Filters
- [ ] **Kernel Methods**
  - [ ] Kernel PCA
  - [ ] Kernel Density Estimation
- [ ] **Specialized Neural Networks**
  - [ ] Radial Basis Function Networks (RBFN)
  - [ ] Kohonen Self-Organizing Maps (SOM)
  - [ ] Siamese Networks
  - [ ] Attention-based RNNs/LSTMs

