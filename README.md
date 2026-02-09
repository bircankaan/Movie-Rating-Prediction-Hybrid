# Hybrid Movie Recommender System (🥈 2nd Place Winner)

This high-performance recommendation engine was developed for the **CMP2003 Data Structures and Algorithms** course. It achieves state-of-the-art accuracy by fusing **Neighborhood-Based Collaborative Filtering (NBCF)** with **Matrix Factorization** techniques, optimized for massive datasets.

## 📊 Performance & Accuracy Benchmarks
The system was engineered to exceed the dual competition goals of accuracy (RMSE) and execution speed. Through numerical methods and trial-error, the following metrics were achieved:

| Method | RMSE (Accuracy) | Execution Time |
| :--- | :---: | :---: |
| **Item-Based CF (IBCF)** | ~1.02 | 0.045s |
| **Matrix Factorization** | ~0.94 | 0.082s |
| **Hybrid Model (Final)** | **0.9055** | **0.0970s** |

* **Accuracy (RMSE):** 0.9055
* **Running Time:** 0.0970 seconds (for million-scale interaction data)
* **Ranking:** 🏆 Awarded **2nd Place** in the overall team competition.

## 🧠 Core Methodology
The system implements a hybrid prediction model that minimizes the Error Metric:
$$RMSE = \sqrt{\frac{\sum_{i=1}^{n}(y_{i}-\hat{y}_{i})^{2}}{n}}$$



### 1. Item-Based Collaborative Filtering (IBCF)
* **Sparse Management:** Uses `std::vector<std::pair<int, float>>` to handle only non-zero interactions, significantly reducing memory footprint.
* **Similarity Metric:** Custom implementation of **Cosine Similarity** using pointer arithmetic to bypass standard indexing latency.
* **Efficiency:** IBCF was selected over UBCF for its superior stability and speed in large-scale testing.

### 2. Matrix Factorization (Latent Factor Model)
* **Latent Factor Space:** Represents users and items in a 10-dimensional latent factor space.
* **SGD Optimization:** Optimized over 70 iterations with Stochastic Gradient Descent.
* **Dynamic Learning:** Implements a multi-stage learning rate (0.008f to 0.004f) for optimal convergence and L2 regularization (0.075f) to prevent overfitting.

## 🛠️ Engineering Highlights & Optimizations
* **Standard Library Only:** Built from scratch using only the **C++ Standard Library**; no external frameworks (like Eigen or Surprise) were used.
* **High-Speed I/O:** Utilized `std::ios::sync_with_stdio(false)` and `std::strtol` for rapid parsing of million-row CSV logs.
* **Memory Management:** Pre-allocated memory using `std::vector::reserve` to eliminate dynamic resizing overhead during data loading.
* **Hybrid Weighting:** Final predictions use an optimized weighted ensemble: **46% IBCF + 54% Matrix Factorization**.

## 📂 Project Structure
* **`src/`**: Optimized `main.cpp` and algorithm implementations.
* **`docs/`**: Detailed technical report (PDF) and complexity analysis.
* **`data/`**: 
    * `training_data.csv`: Historical user-movie ratings for model training.
    * `test_data.csv`: Hidden ratings used to evaluate prediction accuracy.

## ⚙️ Usage
```bash
# Compile with O3 optimization for maximum performance
g++ -O3 src/main.cpp -o recommender

# Run the system using the provided datasets
./recommender < data/training_data.csv data/test_data.csv > results.txt
