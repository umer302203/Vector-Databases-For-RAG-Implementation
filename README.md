# 🔍 Similarity Search by Hand & Grocery Vector Search

> **From mathematical foundations to practical vector search — a comprehensive exploration of similarity metrics and their real-world applications.**

[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.44.0-orange?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4.24-yellow?style=for-the-badge)](https://www.trychroma.com/)
[![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-all--MiniLM--L6--v2-green?style=for-the-badge)](https://www.sbert.net/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

---

## 📌 Overview

This project is a **two-part deep dive** into the mathematical and practical foundations of similarity search:

### Part 1: Similarity Search by Hand (Jupyter Notebook)
An educational, interactive lab that explores the mathematical underpinnings of vector similarity through manual implementation and verification. This is where you learn **how** similarity metrics work — by building them from scratch.

### Part 2: Grocery Vector Similarity Search (Gradio App)
A practical, real-world web application that applies the same mathematical concepts to build a semantic search engine for grocery items using **ChromaDB** vector database and **Sentence Transformers**.

---

## 🎯 Why This Project?

Understanding **how** similarity search works under the hood is just as important as knowing **how to use** it. This project bridges that gap:

- **Mathematical Foundation:** Manual implementations of L2 distance, dot product, and cosine similarity help build an intuitive understanding.
- **Practical Application:** The Grocery Search app demonstrates how these concepts power real-world vector search systems.
- **End-to-End Learning:** From raw math to production-ready web app — all in one repository.

---

## 🧠 Part 1: Similarity Search by Hand (Notebook)

### What You'll Learn

| Concept | Description |
|---------|-------------|
| **L2 (Euclidean) Distance** | The straight-line distance between two vectors in n-dimensional space |
| **Dot Product Similarity** | Measures the magnitude and direction alignment between vectors |
| **Dot Product Distance** | Negative dot product for distance interpretation |
| **Cosine Similarity** | Measures the angle between vectors — independent of magnitude |
| **Cosine Distance** | `1 - cosine_similarity` for distance interpretation |
| **Vector Normalization** | Scaling vectors to unit length (L2 norm = 1) |

### Example: Semantic Disambiguation

The notebook uses a clever dataset of four sentences that all start with the word **"Bugs"** but refer to different concepts:

```python
documents = [
    'Bugs introduced by the intern had to be squashed by the lead developer.',  # Software bug
    'Bugs found by the quality assurance engineer were difficult to debug.',   # Software bug
    'Bugs are common throughout the warm summer months, according to the entomologist.',  # Insect
    'Bugs, in particular spiders, are extensively studied by arachnologists.'  # Insect
]
```

The challenge: Can vector similarity distinguish between **software bugs** and **insects** based on context alone?

**Results:** Both L2 Distance and Cosine Similarity successfully grouped the first two sentences (software bugs) and the last two sentences (insects) — demonstrating the power of semantic embeddings!

### Key Code Snippets

**Manual Euclidean Distance:**
```python
def euclidean_distance_fn(vector1, vector2):
    squared_sum = sum((x - y) ** 2 for x, y in zip(vector1, vector2))
    return math.sqrt(squared_sum)
```

**Manual Cosine Similarity (via Normalization):**
```python
# Normalize vectors (L2 norm = 1)
l2_norms = np.sqrt(np.sum(embeddings**2, axis=1))
normalized_embeddings = embeddings / l2_norms.reshape(-1, 1)

# Cosine similarity = dot product of normalized vectors
cosine_similarity = normalized_embeddings @ normalized_embeddings.T
```

**Query Search:**
```python
# Find the most similar document to a user query
query_embedding = model.encode(["Who is responsible for a coding project?"])
normalized_query = normalize(query_embedding)
similarities = normalized_embeddings @ normalized_query.T
best_match = documents[similarities.argmax()]
```

---

## 🛒 Part 2: Grocery Vector Similarity Search (Gradio App)

### Overview

A production-ready web application that performs semantic search on grocery items using **ChromaDB** and **Sentence Transformers**.

### Dataset

14 grocery items with diverse categories:
- **Fruits:** fresh red apples, organic bananas, ripe mangoes, golden apple, red fruit
- **Bakery:** whole wheat bread
- **Dairy:** farm-fresh eggs, natural yogurt
- **Meat:** grass-fed beef, free-range chicken
- **Seafood:** fresh salmon fillet
- **Pantry:** aromatic coffee beans, pure honey, frozen vegetables

### How It Works

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Grocery Similarity Search Pipeline                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   1. Data Ingestion                                                     │
│      Text items → Embedding using Sentence Transformers                │
│                                                                         │
│   2. Vector Storage                                                     │
│      Embeddings stored in ChromaDB with cosine similarity space        │
│                                                                         │
│   3. Query Processing                                                   │
│      User query → Embedded → Cosine similarity search                  │
│                                                                         │
│   4. Results Display                                                    │
│      Top-K matching items with cosine distance scores                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Example Searches

| Query | Top Results | Why It Works |
|-------|-------------|--------------|
| **"sweet red fruit"** | fresh red apples, golden apple, ripe mangoes | "red" + "sweet" matches fruit descriptions |
| **"breakfast item"** | whole wheat bread, farm-fresh eggs, natural yogurt | All common breakfast foods |
| **"meat products"** | grass-fed beef, free-range chicken | Meat category items |
| **"healthy snack"** | pure honey, natural yogurt, fresh salmon fillet | Semantic association with "healthy" |

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| **Embedding Model** | Sentence Transformers (`all-MiniLM-L6-v2`) |
| **Vector Database** | ChromaDB (Cosine Similarity) |
| **Frontend** | Gradio (Web UI) |
| **Math Operations** | NumPy, SciPy, PyTorch |
| **Language** | Python 3.11+ |

---

## 📁 Project Structure

```
similarity-search-project/
│
├── 📓 Similarity Search by Hand.ipynb    # Educational notebook
│   ├── Manual L2 distance implementation
│   ├── Manual dot product implementation
│   ├── Manual cosine similarity via normalization
│   └── Query-based similarity search
│
├── 🛒 similarity_search.py               # Gradio web application
│   ├── ChromaDB collection setup
│   ├── Grocery dataset (14 items)
│   ├── Semantic search function
│   └── Gradio UI
│
├── 📦 requirements.txt                   # Python dependencies
└── 📖 README.md                          # This file
```

---

## 🚀 Installation

### Prerequisites

- Python 3.11 or higher

### Step 1: Clone the Repository

```bash
git clone https://github.com/umer302203/similarity-search.git
cd similarity-search
```

### Step 2: Create Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Notebook (Part 1)

```bash
jupyter notebook "Similarity Search by Hand.ipynb"
```

### Step 5: Run the Web App (Part 2)

```bash
python similarity_search.py
```

Open your browser at `http://127.0.0.1:7860`

---

## 📊 Comparison: Metrics at a Glance

| Metric | Formula | Range | Best For |
|--------|---------|-------|----------|
| **L2 Distance** | `√Σ(ai - bi)²` | `[0, ∞)` | Low values = more similar |
| **Dot Product** | `Σ(ai * bi)` | `(-∞, ∞)` | High values = more similar |
| **Dot Product Distance** | `-Σ(ai * bi)` | `(-∞, ∞)` | Low values = more similar |
| **Cosine Similarity** | `(a·b) / (||a||·||b||)` | `[-1, 1]` | High values = more similar |
| **Cosine Distance** | `1 - cossim(a,b)` | `[0, 2]` | Low values = more similar |

---

## 🎓 Learning Outcomes

After completing this project, you will be able to:

- ✅ **Implement similarity metrics** from scratch using Python
- ✅ **Understand the mathematical differences** between L2 distance, dot product, and cosine similarity
- ✅ **Normalize vectors** to unit length for cosine similarity calculations
- ✅ **Build a vector search application** using ChromaDB and Gradio
- ✅ **Perform semantic search** with real-world data

---

## 🎬 Demo

🔗 **[Watch the Demo on LinkedIn](https://www.linkedin.com/posts/rana-umer-05a9a9359_rag-vectordatabases-chromadb-ugcPost-7489676548418158592-0K3M/)**

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing`)
5. Open a Pull Request

---

## 📄 License

This project is distributed under the **MIT License** — free to use, modify, and distribute.

---

## 🙏 Acknowledgments

- **IBM Skills Network** for the educational lab structure
- **Sentence Transformers** for the `all-MiniLM-L6-v2` embedding model
- **ChromaDB** for high-performance vector storage
- **Gradio** for the modern, interactive web interface

---

## 📬 Connect with Me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rana-umer-05a9a9359/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/umer302203)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-FF9D00?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/Umer78786)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=120&section=footer&text=Similarity%20Search&fontSize=24&fontColor=white&fontAlignY=65" />
</p>

> Built with ❤️ by [Rana Umer](https://www.linkedin.com/in/rana-umer-05a9a9359/) 🚀
