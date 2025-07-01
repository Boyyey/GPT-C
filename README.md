# 🧠 GPT-C: A Pure C Transformer Language Model

> **A tiny GPT-like neural network, written from scratch in C.**  
> No frameworks. No Python. Just raw arrays, math, and memory.  
> Train it on your own writing. Chat with your own C-powered muse.

---

## 🚀 What is this?

**GPT-C** is a minimal, hackable implementation of a transformer-based language model (like GPT) in pure C.  
It includes everything: tokenizer, embeddings, multi-head self-attention, layer norm, MLP, softmax, Adam optimizer, batching, and a training/chat loop.

- **No dependencies** except a C compiler.
- **No external libraries** for math, neural nets, or data.
- **No magic.** Every tensor, every gradient, every update is explicit.

---

## 🏗️ How does it work?

### **Architecture**
- **Character-level tokenizer:** Each ASCII character is a token.
- **Embeddings:** Each token is mapped to a learned vector.
- **Stacked Transformer Blocks:** Each block has:
  - Multi-head self-attention (with Q, K, V projections)
  - Layer normalization (with learnable scale/shift)
  - Feed-forward MLP (with GELU activation)
  - Residual connections
- **Output projection:** Maps the final hidden state to logits over the vocabulary.
- **Softmax + Sampling:** Turns logits into probabilities, samples next token.
- **Adam optimizer:** For fast, stable training.

### **Training**
- **Next-token prediction:** For each position in your text, predict the next character.
- **Batching:** Trains on multiple samples at once for speed and stability.
- **Backpropagation:** Full gradients for all parameters, including attention, MLP, and embeddings.

### **Chat Loop**
- After training, type a prompt and the model generates a response, character by character.

---

## 🛠️ How do I use it?

### 1. **Clone and Build**
```sh
git clone https://github.com/yourname/gpt-c.git
cd gpt-c
make
```

### 2. **Prepare Training Data**
Edit `data/train.txt` and put in your own writing, stories, or any English text.

### 3. **Run and Train**
```sh
./gpt_c
```
- The model will train on your data, then enter chat mode.

### 4. **Chat!**
Type a prompt and see what your C-powered GPT says.

---

## 🧬 Example Output

```
You: what is the meaning of life?
Bot: eanig of life is to be and to see the world as it is.
```

*(With a small model and dataset, expect poetic, weird, or remix-y outputs!)*

---

## 🧑‍💻 Code Structure

```
src/
  main.c           # Entry point, training/chat loop
  tensor.c/h       # Tensor struct and helpers
  tokenizer.c/h    # Char-level tokenizer
  model.c/h        # Transformer, attention, MLP, layer norm, optimizer
  utils.c/h        # Matrix math, softmax, random
data/
  train.txt        # Your training data
Makefile           # Build script
```

---

## 🧠 How does the math work?

### **Transformer Block**
- **Attention:**  
  \( \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \)
- **LayerNorm:**  
  \( \text{LayerNorm}(x) = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta \)
- **MLP:**  
  \( \text{MLP}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2 \)
- **Residuals:**  
  Add input to output at each sublayer.

### **Backpropagation**
- All gradients are computed by hand, including for attention, MLP, and layer norm.
- Adam optimizer maintains first and second moments for each parameter.

---

## 🏎️ Performance and Limitations

- **Tiny model:** Default is 8-dim, 2 layers, 128-char vocab. You can increase these for more power (at the cost of speed/memory).
- **Char-level:** Learns character patterns, not words.
- **No CUDA/AVX:** All math is plain C for maximum portability and hackability.
- **Educational:** This is for learning, hacking, and fun—not for SOTA results!

---

## 🧩 Want to hack it?

- **Change the model size:** Edit `dim`, `NUM_LAYERS`, `MLP_HIDDEN` in `model.h`.
- **Train on your own data:** Put anything in `data/train.txt`.
- **Add features:** Try temperature sampling, longer context, or word-level tokenization.
- **Visualize:** Print weights, gradients, or activations for insight.

---

## 🦾 Why do this in C?

- **Learn how transformers really work.**
- **No black boxes.** Every operation is visible and hackable.
- **Maximum control.** You can optimize, debug, or extend any part.
- **Fun!** It's like building a neural net with nothing but a soldering iron and a dream.

---

## 📝 Credits & Inspiration

- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [minGPT](https://github.com/karpathy/minGPT)
- [tinygrad](https://github.com/geohot/tinygrad)
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

---

## 🧨 Disclaimer

This is a toy/educational project.  
It will not replace ChatGPT, but it will teach you more about neural nets than any black box ever could.

---

**Star this repo if you love C, AI, or just want to see how deep the rabbit hole goes!**

---

Let me know if you want to add badges, diagrams, or more technical deep-dives! 