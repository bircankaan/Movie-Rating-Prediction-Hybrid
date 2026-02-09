# Hybrid Movie Recommender System (🥈 2nd Place Winner)

This high-performance recommendation engine was developed for the **CMP2003 Data Structures and Algorithms** course. It achieves state-of-the-art accuracy by fusing **Neighborhood-Based Collaborative Filtering (NBCF)** with **Matrix Factorization** techniques, optimized for massive datasets.

## 📊 Performance Benchmarks
The system was engineered for extreme efficiency, meeting the dual competition goals of accuracy and speed:
* **Accuracy (RMSE):** 0.9055
* **Execution Time:** 0.0970 seconds (for million-scale interaction data)
* **Ranking:** 🏆 Awarded **2nd Place** in the overall team competition.

## 🧠 Core Methodology
The system implements a hybrid prediction model, merging two distinct algorithmic approaches to minimize the Error Metric:
$$RMSE = \sqrt{\frac{\sum_{i=1}^{n}(y_{i}-\hat{y}_{i})^{2}}{n}}$$



### 1. Item-Based Collaborative Filtering (IBCF)
* **Similarity Metric:** Custom implementation of **Cosine Similarity** optimized for sparse vectors.
* **Sparse Management:** Instead of dense matrices, the system uses `std::vector<std::pair<int, float>>` to handle only non-zero interactions, significantly reducing memory overhead.
* **Efficiency:** While UBCF was initially tested, IBCF was selected for its superior speed and lower RMSE in large-scale testing.

### 2. Matrix Factorization (Latent Factor Model)
* **Stochastic Gradient Descent (SGD):** The model represents users and items in a 10-dimensional latent factor space, optimized over 70 iterations.
* **Dynamic Learning Rates:** Implements a multi-stage learning rate ($lr_1=0.008f$ to $lr_2=0.004f$) to allow for broad exploration followed by fine-tuned convergence.
* **Regularization:** Integrated L2 regularization ($reg=0.075f$) to prevent overfitting and ensure model generalization.



## 🛠️ Engineering Highlights & Optimizations
* **Standard Library Only:** Built from scratch using only the **C++ Standard Library**; no external frameworks (like Eigen or Surprise) were used.
* **Fast I/O & Parsing:** Utilized `std::ios::sync_with_stdio(false)` and `std::strtol/std::strtof` for high-speed parsing of interaction logs.
* **Hybrid Weighting:** Final predictions use an optimized weighted ensemble: **46% IBCF + 54% Matrix Factorization**, found through numerical trial and error.

## 📂 Project Structure
* **`src/`**: Contains the optimized `main.cpp` and class definitions.
* **`docs/`**: Detailed technical report (PDF) and complexity analysis.
* **`data/`**: 
    * `training_data.csv`: Historical user-movie ratings used for model training.
    * `test_data.csv`: Hidden ratings used to evaluate prediction accuracy.

## ⚙️ Usage
To compile and run the system:

```bash
# Compile with O3 optimization for maximum execution speed
g++ -O3 src/main.cpp -o recommender

# Run the system using the provided datasets
./recommender < data/training_data.csv data/test_data.csv > results.txt
