#include "model.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "utils.h"
#include "tokenizer.h"

float* embedding_matrix = NULL;
float* proj_W = NULL;
float* proj_b = NULL;
float* grad_proj_W = NULL;
float* grad_proj_b = NULL;
float* grad_embedding_matrix = NULL;
float* m_embedding_matrix = NULL;
float* v_embedding_matrix = NULL;
float* m_proj_W = NULL;
float* v_proj_W = NULL;
float* m_proj_b = NULL;
float* v_proj_b = NULL;

void init_embedding(int vocab_size, int dim) {
    if (embedding_matrix) free(embedding_matrix);
    if (grad_embedding_matrix) free(grad_embedding_matrix);
    if (m_embedding_matrix) free(m_embedding_matrix);
    if (v_embedding_matrix) free(v_embedding_matrix);
    embedding_matrix = (float*)malloc(vocab_size * dim * sizeof(float));
    grad_embedding_matrix = (float*)calloc(vocab_size * dim, sizeof(float));
    m_embedding_matrix = (float*)calloc(vocab_size * dim, sizeof(float));
    v_embedding_matrix = (float*)calloc(vocab_size * dim, sizeof(float));
    for (int i = 0; i < vocab_size * dim; ++i) embedding_matrix[i] = randf() * 2.0f - 1.0f;
}

void embed_tokens(const int* tokens, int len, int dim, Tensor* out) {
    // out: [len, dim]
    for (int i = 0; i < len; ++i) {
        int idx = tokens[i];
        for (int d = 0; d < dim; ++d) {
            out->data[i*dim + d] = embedding_matrix[idx*dim + d];
        }
    }
}

void positional_encoding(Tensor* t) {
    // Add simple sinusoidal positional encoding
    int seq_len = t->shape[0];
    int dim = t->shape[1];
    for (int pos = 0; pos < seq_len; ++pos) {
        for (int i = 0; i < dim; ++i) {
            float angle = pos / powf(10000, 2.0f * (i/2) / dim);
            t->data[pos*dim + i] += (i % 2 == 0) ? sinf(angle) : cosf(angle);
        }
    }
}

void layer_norm(Tensor* t, float* gamma, float* beta) {
    // Simple layer norm over last dimension
    int batch = t->shape[0];
    int dim = t->shape[1];
    for (int b = 0; b < batch; ++b) {
        float mean = 0.0f, var = 0.0f;
        for (int i = 0; i < dim; ++i) mean += t->data[b*dim + i];
        mean /= dim;
        for (int i = 0; i < dim; ++i) var += (t->data[b*dim + i] - mean) * (t->data[b*dim + i] - mean);
        var /= dim;
        float std = sqrtf(var + 1e-5f);
        for (int i = 0; i < dim; ++i) {
            t->data[b*dim + i] = gamma[i] * ((t->data[b*dim + i] - mean) / std) + beta[i];
        }
    }
}

void attention(Tensor* Q, Tensor* K, Tensor* V, Tensor* output) {
    // Self-attention: output = softmax(QK^T/sqrt(d_k))V
    // For simplicity, assume Q, K, V: [seq, dim], output: [seq, dim]
    int seq = Q->shape[0];
    int dim = Q->shape[1];
    float scale = 1.0f / sqrtf((float)dim);
    float* scores = (float*)malloc(seq * sizeof(float));
    // Compute QK^T
    for (int i = 0; i < seq; ++i) {
        for (int j = 0; j < seq; ++j) {
            float dot = 0.0f;
            for (int d = 0; d < dim; ++d) dot += Q->data[i*dim + d] * K->data[j*dim + d];
            scores[j] = dot * scale;
        }
    }
    // Softmax over rows
    for (int i = 0; i < seq; ++i) {
        float max = scores[i];
        for (int j = 1; j < seq; ++j) if (scores[i] > max) max = scores[i];
        float sum = 0.0f;
        for (int j = 0; j < seq; ++j) {
            scores[j] = expf(scores[j] - max);
            sum += scores[j];
        }
        for (int j = 0; j < seq; ++j) scores[j] /= sum;
    }
    // Weighted sum: scores x V
    for (int i = 0; i < seq; ++i) {
        for (int d = 0; d < dim; ++d) {
            float val = 0.0f;
            for (int j = 0; j < seq; ++j) val += scores[j] * V->data[j*dim + d];
            output->data[i*dim + d] = val;
        }
    }
    free(scores);
}

void init_transformer_block(TransformerBlock* block, int dim) {
    block->dim = dim;
    int d = dim, h = MLP_HIDDEN;
    block->W_q = (float*)malloc(d * d * sizeof(float));
    block->W_k = (float*)malloc(d * d * sizeof(float));
    block->W_v = (float*)malloc(d * d * sizeof(float));
    block->W_o = (float*)malloc(d * d * sizeof(float));
    block->W1 = (float*)malloc(d * h * sizeof(float));
    block->b1 = (float*)calloc(h, sizeof(float));
    block->W2 = (float*)malloc(h * d * sizeof(float));
    block->b2 = (float*)calloc(d, sizeof(float));
    block->ln1_gamma = (float*)malloc(d * sizeof(float));
    block->ln1_beta = (float*)calloc(d, sizeof(float));
    block->ln2_gamma = (float*)malloc(d * sizeof(float));
    block->ln2_beta = (float*)calloc(d, sizeof(float));
    for (int i = 0; i < d * d; ++i) block->W_q[i] = randf() * 0.2f - 0.1f;
    for (int i = 0; i < d * d; ++i) block->W_k[i] = randf() * 0.2f - 0.1f;
    for (int i = 0; i < d * d; ++i) block->W_v[i] = randf() * 0.2f - 0.1f;
    for (int i = 0; i < d * d; ++i) block->W_o[i] = randf() * 0.2f - 0.1f;
    for (int i = 0; i < d * h; ++i) block->W1[i] = randf() * 0.2f - 0.1f;
    for (int i = 0; i < h * d; ++i) block->W2[i] = randf() * 0.2f - 0.1f;
    for (int i = 0; i < d; ++i) block->ln1_gamma[i] = 1.0f;
    for (int i = 0; i < d; ++i) block->ln2_gamma[i] = 1.0f;
}

void free_transformer_block(TransformerBlock* block) {
    free(block->W_q); free(block->W_k); free(block->W_v); free(block->W_o);
    free(block->W1); free(block->b1); free(block->W2); free(block->b2);
    free(block->ln1_gamma); free(block->ln1_beta); free(block->ln2_gamma); free(block->ln2_beta);
}

// Helper: GELU activation
float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
}

void transformer_block_forward(TransformerBlock* block, Tensor* input, Tensor* output) {
    int seq = input->shape[0];
    int dim = block->dim;
    int heads = NUM_HEADS;
    int head_dim = dim / heads;
    // Allocate temp tensors
    float* Q = (float*)malloc(seq * dim * sizeof(float));
    float* K = (float*)malloc(seq * dim * sizeof(float));
    float* V = (float*)malloc(seq * dim * sizeof(float));
    // Linear projections
    matmul(input->data, block->W_q, Q, seq, dim, dim);
    matmul(input->data, block->W_k, K, seq, dim, dim);
    matmul(input->data, block->W_v, V, seq, dim, dim);
    // Multi-head attention
    float* attn_out = (float*)calloc(seq * dim, sizeof(float));
    for (int h = 0; h < heads; ++h) {
        // Slice Q, K, V for this head
        float* Qh = Q + h * head_dim;
        float* Kh = K + h * head_dim;
        float* Vh = V + h * head_dim;
        float* out_h = attn_out + h * head_dim;
        // Compute attention for this head
        // For each token in sequence
        for (int i = 0; i < seq; ++i) {
            float* scores = (float*)malloc(seq * sizeof(float));
            for (int j = 0; j < seq; ++j) {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d)
                    dot += Qh[i*dim + d] * Kh[j*dim + d];
                scores[j] = dot / sqrtf((float)head_dim);
            }
            softmax(scores, seq);
            // Weighted sum of V
            for (int d = 0; d < head_dim; ++d) {
                float val = 0.0f;
                for (int j = 0; j < seq; ++j)
                    val += scores[j] * Vh[j*dim + d];
                out_h[i*dim + d] = val;
            }
            free(scores);
        }
    }
    // Project back
    float* attn_proj = (float*)malloc(seq * dim * sizeof(float));
    matmul(attn_out, block->W_o, attn_proj, seq, dim, dim);
    // Residual + LayerNorm 1
    for (int i = 0; i < seq * dim; ++i) attn_proj[i] += input->data[i];
    Tensor temp1 = {attn_proj, input->shape, 2, seq*dim};
    layer_norm(&temp1, block->ln1_gamma, block->ln1_beta);
    // MLP
    float* mlp1 = (float*)malloc(seq * MLP_HIDDEN * sizeof(float));
    matmul(temp1.data, block->W1, mlp1, seq, MLP_HIDDEN, dim);
    for (int i = 0; i < seq * MLP_HIDDEN; ++i) mlp1[i] = gelu(mlp1[i] + block->b1[i % MLP_HIDDEN]);
    float* mlp2 = (float*)malloc(seq * dim * sizeof(float));
    matmul(mlp1, block->W2, mlp2, seq, dim, MLP_HIDDEN);
    for (int i = 0; i < seq * dim; ++i) mlp2[i] += block->b2[i % dim];
    // Residual + LayerNorm 2
    for (int i = 0; i < seq * dim; ++i) mlp2[i] += temp1.data[i];
    Tensor temp2 = {mlp2, input->shape, 2, seq*dim};
    layer_norm(&temp2, block->ln2_gamma, block->ln2_beta);
    // Output
    memcpy(output->data, temp2.data, seq * dim * sizeof(float));
    // Free
    free(Q); free(K); free(V); free(attn_out); free(attn_proj); free(mlp1); free(mlp2);
}

void transformer_block(Tensor* input, Tensor* output) {
    // Stub: Copy input to output
    memcpy(output->data, input->data, input->size * sizeof(float));
    // In a real model, add attention, MLP, layer norm, etc.
}

void init_projection(int dim, int vocab_size) {
    if (proj_W) free(proj_W);
    if (proj_b) free(proj_b);
    if (grad_proj_W) free(grad_proj_W);
    if (grad_proj_b) free(grad_proj_b);
    if (m_proj_W) free(m_proj_W);
    if (v_proj_W) free(v_proj_W);
    if (m_proj_b) free(m_proj_b);
    if (v_proj_b) free(v_proj_b);
    proj_W = (float*)malloc(dim * vocab_size * sizeof(float));
    proj_b = (float*)calloc(vocab_size, sizeof(float));
    grad_proj_W = (float*)calloc(dim * vocab_size, sizeof(float));
    grad_proj_b = (float*)calloc(vocab_size, sizeof(float));
    m_proj_W = (float*)calloc(dim * vocab_size, sizeof(float));
    v_proj_W = (float*)calloc(dim * vocab_size, sizeof(float));
    m_proj_b = (float*)calloc(vocab_size, sizeof(float));
    v_proj_b = (float*)calloc(vocab_size, sizeof(float));
    for (int i = 0; i < dim * vocab_size; ++i) proj_W[i] = randf() * 0.2f - 0.1f;
}

void free_projection() {
    if (proj_W) free(proj_W);
    if (proj_b) free(proj_b);
    if (grad_proj_W) free(grad_proj_W);
    if (grad_proj_b) free(grad_proj_b);
    proj_W = NULL;
    proj_b = NULL;
    grad_proj_W = NULL;
    grad_proj_b = NULL;
}

void sgd_update_projection(int dim, int vocab_size, float lr) {
    for (int d = 0; d < dim; ++d) {
        for (int v = 0; v < vocab_size; ++v) {
            proj_W[d * vocab_size + v] -= lr * grad_proj_W[d * vocab_size + v];
            grad_proj_W[d * vocab_size + v] = 0.0f;
        }
    }
    for (int v = 0; v < vocab_size; ++v) {
        proj_b[v] -= lr * grad_proj_b[v];
        grad_proj_b[v] = 0.0f;
    }
}

void project_logits(const float* input, int dim, float* logits, int vocab_size) {
    // input: [dim], logits: [vocab_size]
    for (int v = 0; v < vocab_size; ++v) {
        float sum = 0.0f;
        for (int d = 0; d < dim; ++d) sum += input[d] * proj_W[d * vocab_size + v];
        logits[v] = sum + proj_b[v];
    }
}

int sample_logits(const float* logits, int vocab_size) {
    float* probs = (float*)malloc(vocab_size * sizeof(float));
    memcpy(probs, logits, vocab_size * sizeof(float));
    softmax(probs, vocab_size);
    float r = randf();
    float cum = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        cum += probs[i];
        if (r < cum) {
            free(probs);
            return i;
        }
    }
    free(probs);
    return vocab_size - 1;
}

void init_model(Model* model, int num_layers, int dim) {
    model->num_layers = num_layers;
    model->dim = dim;
    model->blocks = (TransformerBlock*)malloc(num_layers * sizeof(TransformerBlock));
    for (int i = 0; i < num_layers; ++i) {
        init_transformer_block(&model->blocks[i], dim);
    }
}

void free_model(Model* model) {
    for (int i = 0; i < model->num_layers; ++i) {
        free_transformer_block(&model->blocks[i]);
    }
    free(model->blocks);
}

void model_forward(Model* model, Tensor* input, Tensor* output) {
    int seq = input->shape[0];
    int dim = input->shape[1];
    Tensor* in = tensor_create(input->shape, 2);
    Tensor* out = tensor_create(input->shape, 2);
    memcpy(in->data, input->data, seq * dim * sizeof(float));
    for (int l = 0; l < model->num_layers; ++l) {
        transformer_block_forward(&model->blocks[l], in, out);
        // Swap in/out for next layer
        float* tmp = in->data;
        in->data = out->data;
        out->data = tmp;
    }
    memcpy(output->data, in->data, seq * dim * sizeof(float));
    tensor_free(in);
    tensor_free(out);
}

void chat_loop(Model* model, int dim, int max_gen) {
    char input_text[256];
    int max_len = 64;
    printf("\nType something! (type 'exit' to quit)\n");
    while (1) {
        printf("You: ");
        fflush(stdout);
        if (!fgets(input_text, sizeof(input_text), stdin)) break;
        if (strncmp(input_text, "exit", 4) == 0) break;
        int tok_len;
        int* tokens = tokenize(input_text, &tok_len);
        if (tok_len > max_len) tok_len = max_len;
        int* gen_tokens = (int*)malloc((tok_len + max_gen) * sizeof(int));
        memcpy(gen_tokens, tokens, tok_len * sizeof(int));
        int cur_len = tok_len;
        for (int step = 0; step < max_gen; ++step) {
            int shape[2] = {cur_len, dim};
            Tensor* input = tensor_create(shape, 2);
            embed_tokens(gen_tokens, cur_len, dim, input);
            positional_encoding(input);
            Tensor* output = tensor_create(shape, 2);
            model_forward(model, input, output);
            float* logits = (float*)malloc(VOCAB_SIZE * sizeof(float));
            project_logits(output->data, dim, logits, VOCAB_SIZE);
            int next_token = sample_logits(logits, VOCAB_SIZE);
            gen_tokens[cur_len] = next_token;
            cur_len++;
            tensor_free(input);
            tensor_free(output);
            free(logits);
        }
        char* resp = detokenize(gen_tokens + tok_len, max_gen);
        printf("Bot: %s\n", resp);
        free(tokens);
        free(gen_tokens);
        free(resp);
    }
}

void save_model(const char* path, Model* model, int dim, int vocab_size) {
    FILE* f = fopen(path, "wb");
    if (!f) { printf("Failed to open %s for writing\n", path); return; }
    // Embedding
    fwrite(embedding_matrix, sizeof(float), vocab_size * dim, f);
    // Transformer blocks
    for (int l = 0; l < model->num_layers; ++l) {
        TransformerBlock* b = &model->blocks[l];
        fwrite(b->W_q, sizeof(float), dim*dim, f);
        fwrite(b->W_k, sizeof(float), dim*dim, f);
        fwrite(b->W_v, sizeof(float), dim*dim, f);
        fwrite(b->W_o, sizeof(float), dim*dim, f);
        fwrite(b->W1, sizeof(float), dim*MLP_HIDDEN, f);
        fwrite(b->b1, sizeof(float), MLP_HIDDEN, f);
        fwrite(b->W2, sizeof(float), MLP_HIDDEN*dim, f);
        fwrite(b->b2, sizeof(float), dim, f);
        fwrite(b->ln1_gamma, sizeof(float), dim, f);
        fwrite(b->ln1_beta, sizeof(float), dim, f);
        fwrite(b->ln2_gamma, sizeof(float), dim, f);
        fwrite(b->ln2_beta, sizeof(float), dim, f);
    }
    // Projection
    fwrite(proj_W, sizeof(float), dim*vocab_size, f);
    fwrite(proj_b, sizeof(float), vocab_size, f);
    fclose(f);
    printf("Model saved to %s\n", path);
}

void load_model(const char* path, Model* model, int dim, int vocab_size) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("Failed to open %s for reading\n", path); return; }
    fread(embedding_matrix, sizeof(float), vocab_size * dim, f);
    for (int l = 0; l < model->num_layers; ++l) {
        TransformerBlock* b = &model->blocks[l];
        fread(b->W_q, sizeof(float), dim*dim, f);
        fread(b->W_k, sizeof(float), dim*dim, f);
        fread(b->W_v, sizeof(float), dim*dim, f);
        fread(b->W_o, sizeof(float), dim*dim, f);
        fread(b->W1, sizeof(float), dim*MLP_HIDDEN, f);
        fread(b->b1, sizeof(float), MLP_HIDDEN, f);
        fread(b->W2, sizeof(float), MLP_HIDDEN*dim, f);
        fread(b->b2, sizeof(float), dim, f);
        fread(b->ln1_gamma, sizeof(float), dim, f);
        fread(b->ln1_beta, sizeof(float), dim, f);
        fread(b->ln2_gamma, sizeof(float), dim, f);
        fread(b->ln2_beta, sizeof(float), dim, f);
    }
    fread(proj_W, sizeof(float), dim*vocab_size, f);
    fread(proj_b, sizeof(float), vocab_size, f);
    fclose(f);
    printf("Model loaded from %s\n", path);
}

// Updated train: backprop for output projection, SGD, file input
typedef struct { char* data; size_t size; } FileData;
static FileData read_file(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) { printf("Failed to open %s\n", filename); exit(1); }
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* data = (char*)malloc(size + 1);
    fread(data, 1, size, f);
    data[size] = '\0';
    fclose(f);
    FileData fd = { data, size };
    return fd;
}

void train(Model* model, int dim, int vocab_size, const char* filename, int epochs, float lr, int use_adam) {
    FileData fd = read_file(filename);
    printf("[DEBUG] Training file size: %d\n", (int)fd.size);
    printf("[DEBUG] dim=%d, vocab_size=%d\n", dim, vocab_size);
    if (!fd.data) {
        printf("[ERROR] File data is NULL\n");
        exit(1);
    }
    int len = fd.size;
    if (len < 2) {
        printf("[ERROR] Training data too small (len=%d)\n", len);
        exit(1);
    }
    int* tokens = (int*)malloc(len * sizeof(int));
    if (!tokens) { printf("Failed to allocate tokens\n"); exit(1); }
    printf("[DEBUG] Allocated tokens array at %p for len=%d\n", (void*)tokens, len);
    for (int i = 0; i < len; ++i) {
        tokens[i] = (unsigned char)fd.data[i];
        if (i % 1000 == 0) printf("[DEBUG] tokens[%d] = %d\n", i, tokens[i]);
    }
    printf("[DEBUG] Finished filling tokens array\n");
    printf("[DEBUG] grad_proj_W at %p, grad_proj_b at %p, grad_embedding_matrix at %p\n", (void*)grad_proj_W, (void*)grad_proj_b, (void*)grad_embedding_matrix);
    if (grad_proj_W) printf("[DEBUG] grad_proj_W[0]=%f\n", grad_proj_W[0]);
    if (grad_proj_b) printf("[DEBUG] grad_proj_b[0]=%f\n", grad_proj_b[0]);
    if (grad_embedding_matrix) printf("[DEBUG] grad_embedding_matrix[0]=%f\n", grad_embedding_matrix[0]);
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    int t = 1;
    printf("[DEBUG] Starting epoch loop\n");
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        int count = 0;
        printf("[DEBUG] Epoch %d\n", epoch+1);
        for (int i = 0; i < len - 1; i += BATCH_SIZE) {
            int batch_end = (i + BATCH_SIZE < len - 1) ? BATCH_SIZE : (len - 1 - i);
            printf("[DEBUG] Zeroing gradients for batch starting at i=%d, batch_end=%d, t=%d\n", i, batch_end, t);
            for (int j = 0; j < dim * vocab_size; ++j) grad_proj_W[j] = 0.0f;
            for (int j = 0; j < vocab_size; ++j) grad_proj_b[j] = 0.0f;
            for (int j = 0; j < vocab_size * dim; ++j) grad_embedding_matrix[j] = 0.0f;
            if (grad_proj_W) printf("[DEBUG] (batch) grad_proj_W[0]=%f\n", grad_proj_W[0]);
            if (grad_proj_b) printf("[DEBUG] (batch) grad_proj_b[0]=%f\n", grad_proj_b[0]);
            if (grad_embedding_matrix) printf("[DEBUG] (batch) grad_embedding_matrix[0]=%f\n", grad_embedding_matrix[0]);
            float batch_loss = 0.0f;
            printf("[DEBUG] Entering batch loop, batch_end=%d\n", batch_end);
            for (int b = 0; b < batch_end; ++b) {
                int idx = i + b;
                int shape[2] = {1, dim};
                printf("[DEBUG] Allocating input tensor for idx=%d\n", idx);
                Tensor* input = tensor_create(shape, 2);
                if (!input) { printf("Failed to allocate input tensor\n"); exit(1); }
                printf("[DEBUG] Embedding tokens for idx=%d\n", idx);
                embed_tokens(&tokens[idx], 1, dim, input);
                positional_encoding(input);
                printf("[DEBUG] Allocating output tensor for idx=%d\n", idx);
                Tensor* output = tensor_create(shape, 2);
                if (!output) { printf("Failed to allocate output tensor\n"); exit(1); }
                printf("[DEBUG] Forward pass for idx=%d\n", idx);
                model_forward(model, input, output);
                printf("[DEBUG] Allocating logits for idx=%d\n", idx);
                float* logits = (float*)malloc(vocab_size * sizeof(float));
                if (!logits) { printf("Failed to allocate logits\n"); exit(1); }
                printf("[DEBUG] Projecting logits for idx=%d\n", idx);
                project_logits(output->data, dim, logits, vocab_size);
                printf("[DEBUG] Softmax for idx=%d\n", idx);
                softmax(logits, vocab_size);
                printf("[DEBUG] Calculating loss for idx=%d\n", idx);
                if (idx+1 >= len) {
                    printf("[ERROR] idx+1 (%d) >= len (%d)\n", idx+1, len);
                    exit(1);
                }
                float loss = -logf(logits[tokens[idx+1]] + 1e-8f);
                batch_loss += loss;
                // Backprop for output projection
                for (int v = 0; v < vocab_size; ++v) {
                    float grad = logits[v];
                    if (v == tokens[idx+1]) grad -= 1.0f;
                    for (int d = 0; d < dim; ++d) {
                        grad_proj_W[d * vocab_size + v] += grad * input->data[d];
                        grad_embedding_matrix[tokens[idx]*dim + d] += grad * proj_W[d * vocab_size + v];
                    }
                    grad_proj_b[v] += grad;
                }
                printf("[DEBUG] Freeing tensors for idx=%d\n", idx);
                tensor_free(input);
                tensor_free(output);
                free(logits);
            }
            printf("[DEBUG] Finished batch loop for i=%d, batch_end=%d\n", i, batch_end);
            printf("[DEBUG] Before optimizer update, t=%d\n", t);
            if (use_adam) {
                if (!m_proj_W || !v_proj_W || !m_proj_b || !v_proj_b) {
                    printf("[ERROR] Adam optimizer buffers are NULL!\n");
                    exit(1);
                }
                printf("[DEBUG] Calling adam_update_projection\n");
                adam_update_projection(dim, vocab_size, lr, beta1, beta2, eps, t);
                printf("[DEBUG] Calling adam_update_embedding\n");
                adam_update_embedding(vocab_size, dim, lr, beta1, beta2, eps, t);
            } else {
                printf("[DEBUG] Calling sgd_update_projection\n");
                sgd_update_projection(dim, vocab_size, lr);
                // (SGD for embedding not implemented)
            }
            printf("[DEBUG] After optimizer update, t=%d\n", t);
            t++;
            total_loss += batch_loss / batch_end;
            count++;
            printf("[DEBUG] End of batch, next i=%d\n", i+BATCH_SIZE);
        }
        printf("Epoch %d, avg loss: %.4f\n", epoch+1, total_loss/count);
    }
    free(tokens);
    free(fd.data);
    printf("Training done. (Only output projection/embedding are trained)\n");
}

// === Backpropagation for Transformer Block ===

// Helper: derivative of GELU (approximate)
static float d_gelu(float x) {
    float tanh_out = tanhf(0.79788456f * (x + 0.044715f * x * x * x));
    float left = 0.5f * tanh_out + 0.5f;
    float right = 0.5f * x * (1 - tanh_out * tanh_out) * 0.79788456f * (1 + 3 * 0.044715f * x * x);
    return left + right;
}

void mlp_backward(TransformerBlock* block, Tensor* input, Tensor* d_output, Tensor* d_input) {
    // input: [1, dim], d_output: [1, dim], d_input: [1, dim]
    int dim = block->dim;
    int h = MLP_HIDDEN;
    float* W1 = block->W1;
    float* W2 = block->W2;
    float* b1 = block->b1;
    float* grad_W1 = block->grad_W1;
    float* grad_W2 = block->grad_W2;
    float* grad_b1 = block->grad_b1;
    float* grad_b2 = block->grad_b2;
    // Forward pass (needed for backward)
    float* z1 = (float*)malloc(h * sizeof(float));
    for (int i = 0; i < h; ++i) {
        z1[i] = b1[i];
        for (int j = 0; j < dim; ++j) z1[i] += input->data[j] * W1[j*h + i];
    }
    float* a1 = (float*)malloc(h * sizeof(float));
    for (int i = 0; i < h; ++i) a1[i] = gelu(z1[i]);
    // Backward pass
    float* dz1 = (float*)calloc(h, sizeof(float));
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < h; ++j) {
            grad_W2[j*dim + i] += a1[j] * d_output->data[i];
            dz1[j] += W2[j*dim + i] * d_output->data[i];
        }
        grad_b2[i] += d_output->data[i];
    }
    for (int j = 0; j < h; ++j) dz1[j] *= d_gelu(z1[j]);
    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < dim; ++i) {
            grad_W1[i*h + j] += input->data[i] * dz1[j];
            d_input->data[i] += W1[i*h + j] * dz1[j];
        }
        grad_b1[j] += dz1[j];
    }
    free(z1);
    free(a1);
    free(dz1);
}

void layer_norm_backward(TransformerBlock* block, Tensor* input, float* gamma, float* beta, Tensor* d_output, Tensor* d_input, float* grad_gamma, float* grad_beta) {
    // input: [1, dim], d_output: [1, dim], d_input: [1, dim]
    int dim = block->dim;
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < dim; ++i) mean += input->data[i];
    mean /= dim;
    for (int i = 0; i < dim; ++i) var += (input->data[i] - mean) * (input->data[i] - mean);
    var /= dim;
    float std = sqrtf(var + 1e-5f);
    for (int i = 0; i < dim; ++i) {
        float norm = (input->data[i] - mean) / std;
        grad_gamma[i] += d_output->data[i] * norm;
        grad_beta[i] += d_output->data[i];
    }
    for (int i = 0; i < dim; ++i) {
        float norm = (input->data[i] - mean) / std;
        float dnorm = d_output->data[i] * gamma[i];
        float dvar = -0.5f * dnorm * norm / (var + 1e-5f);
        float dmean = -dnorm / std;
        d_input->data[i] = dnorm / std + dvar * 2.0f * (input->data[i] - mean) / dim + dmean / dim;
    }
}

void attention_backward(TransformerBlock* block, Tensor* input, Tensor* d_output, Tensor* d_input) {
    // Full multi-head attention backward pass
    int seq = input->shape[0];
    int dim = block->dim;
    int heads = NUM_HEADS;
    int head_dim = dim / heads;
    float scale = 1.0f / sqrtf((float)head_dim);

    // Allocate temp buffers
    float* Q = (float*)malloc(seq * dim * sizeof(float));
    float* K = (float*)malloc(seq * dim * sizeof(float));
    float* V = (float*)malloc(seq * dim * sizeof(float));
    matmul(input->data, block->W_q, Q, seq, dim, dim);
    matmul(input->data, block->W_k, K, seq, dim, dim);
    matmul(input->data, block->W_v, V, seq, dim, dim);

    float* attn_out = (float*)calloc(seq * dim, sizeof(float));
    float* scores = (float*)malloc(seq * seq * heads * sizeof(float));
    float* softmax_scores = (float*)malloc(seq * seq * heads * sizeof(float));

    // Forward pass: compute attention scores and softmax (needed for backward)
    for (int h = 0; h < heads; ++h) {
        for (int i = 0; i < seq; ++i) {
            for (int j = 0; j < seq; ++j) {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d)
                    dot += Q[i*dim + h*head_dim + d] * K[j*dim + h*head_dim + d];
                scores[h*seq*seq + i*seq + j] = dot * scale;
            }
            // Softmax
            float max = scores[h*seq*seq + i*seq];
            for (int j = 1; j < seq; ++j)
                if (scores[h*seq*seq + i*seq + j] > max) max = scores[h*seq*seq + i*seq + j];
            float sum = 0.0f;
            for (int j = 0; j < seq; ++j) {
                softmax_scores[h*seq*seq + i*seq + j] = expf(scores[h*seq*seq + i*seq + j] - max);
                sum += softmax_scores[h*seq*seq + i*seq + j];
            }
            for (int j = 0; j < seq; ++j)
                softmax_scores[h*seq*seq + i*seq + j] /= sum;
        }
    }

    // Backward pass
    float* dQ = (float*)calloc(seq * dim, sizeof(float));
    float* dK = (float*)calloc(seq * dim, sizeof(float));
    float* dV = (float*)calloc(seq * dim, sizeof(float));
    float* d_scores = (float*)calloc(seq * seq * heads, sizeof(float));
    float* d_softmax = (float*)calloc(seq * seq * heads, sizeof(float));

    // 1. Backprop through output projection W_o
    float* d_attn_concat = (float*)calloc(seq * dim, sizeof(float));
    matmul(d_output->data, block->W_o, d_attn_concat, seq, dim, dim);
    // Accumulate grad_W_o
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            float sum = 0.0f;
            for (int s = 0; s < seq; ++s) sum += attn_out[s*dim + i] * d_output->data[s*dim + j];
            block->grad_W_o[i*dim + j] += sum;
        }
    }

    // 2. Backprop through attention weighted sum and softmax
    for (int h = 0; h < heads; ++h) {
        for (int i = 0; i < seq; ++i) {
            for (int d = 0; d < head_dim; ++d) {
                // dV: weighted sum of softmax_scores
                float grad = 0.0f;
                for (int j = 0; j < seq; ++j) {
                    grad += softmax_scores[h*seq*seq + i*seq + j] * d_attn_concat[i*dim + h*head_dim + d];
                }
                dV[i*dim + h*head_dim + d] += grad;
            }
        }
        // d_softmax: propagate to softmax inputs
        for (int i = 0; i < seq; ++i) {
            for (int j = 0; j < seq; ++j) {
                float grad = 0.0f;
                for (int d = 0; d < head_dim; ++d)
                    grad += V[j*dim + h*head_dim + d] * d_attn_concat[i*dim + h*head_dim + d];
                d_softmax[h*seq*seq + i*seq + j] = grad;
            }
        }
        // d_scores: softmax backward (Jacobian)
        for (int i = 0; i < seq; ++i) {
            for (int j = 0; j < seq; ++j) {
                float s = softmax_scores[h*seq*seq + i*seq + j];
                float sum = 0.0f;
                for (int k = 0; k < seq; ++k)
                    sum += d_softmax[h*seq*seq + i*seq + k] * ((j == k) ? s * (1 - s) : -s * softmax_scores[h*seq*seq + i*seq + k]);
                d_scores[h*seq*seq + i*seq + j] = sum;
            }
        }
        // dQ, dK: propagate through QK^T
        for (int i = 0; i < seq; ++i) {
            for (int d = 0; d < head_dim; ++d) {
                float gradQ = 0.0f;
                for (int j = 0; j < seq; ++j)
                    gradQ += d_scores[h*seq*seq + i*seq + j] * K[j*dim + h*head_dim + d] * scale;
                dQ[i*dim + h*head_dim + d] += gradQ;
            }
        }
        for (int j = 0; j < seq; ++j) {
            for (int d = 0; d < head_dim; ++d) {
                float gradK = 0.0f;
                for (int i = 0; i < seq; ++i)
                    gradK += d_scores[h*seq*seq + i*seq + j] * Q[i*dim + h*head_dim + d] * scale;
                dK[j*dim + h*head_dim + d] += gradK;
            }
        }
    }

    // 3. Backprop through Q, K, V projections
    // Accumulate grad_W_q, grad_W_k, grad_W_v
    for (int i = 0; i < seq; ++i) {
        for (int d = 0; d < dim; ++d) {
            for (int k = 0; k < dim; ++k) {
                block->grad_W_q[d*dim + k] += input->data[i*dim + d] * dQ[i*dim + k];
                block->grad_W_k[d*dim + k] += input->data[i*dim + d] * dK[i*dim + k];
                block->grad_W_v[d*dim + k] += input->data[i*dim + d] * dV[i*dim + k];
            }
        }
    }
    // Propagate gradients to input
    for (int i = 0; i < seq; ++i) {
        for (int d = 0; d < dim; ++d) {
            float grad = 0.0f;
            for (int k = 0; k < dim; ++k) {
                grad += dQ[i*dim + k] * block->W_q[d*dim + k];
                grad += dK[i*dim + k] * block->W_k[d*dim + k];
                grad += dV[i*dim + k] * block->W_v[d*dim + k];
            }
            d_input->data[i*dim + d] += grad;
        }
    }
    free(Q); free(K); free(V); free(attn_out); free(scores); free(softmax_scores);
    free(dQ); free(dK); free(dV); free(d_scores); free(d_softmax); free(d_attn_concat);
}

void transformer_block_backward(TransformerBlock* block, Tensor* input, Tensor* output, Tensor* d_output, Tensor* d_input) {
    // Backprop through layer norm, MLP, attention, and residuals in reverse order
    // This is a simplified version for demonstration
    Tensor* d_mlp = tensor_create(input->shape, 2);
    Tensor* d_ln2 = tensor_create(input->shape, 2);
    Tensor* d_attn = tensor_create(input->shape, 2);
    Tensor* d_ln1 = tensor_create(input->shape, 2);
    // Backprop through MLP
    mlp_backward(block, output, d_output, d_mlp);
    // Backprop through layer norm 2
    layer_norm_backward(block, output, block->ln2_gamma, block->ln2_beta, d_mlp, d_ln2, block->grad_ln2_gamma, block->grad_ln2_beta);
    // Backprop through attention
    attention_backward(block, input, d_ln2, d_attn);
    // Backprop through layer norm 1
    layer_norm_backward(block, input, block->ln1_gamma, block->ln1_beta, d_attn, d_ln1, block->grad_ln1_gamma, block->grad_ln1_beta);
    // Residual: add gradients from both branches
    for (int i = 0; i < input->size; ++i) {
        d_input->data[i] = d_ln1->data[i];
    }
    tensor_free(d_mlp);
    tensor_free(d_ln2);
    tensor_free(d_attn);
    tensor_free(d_ln1);
}

void free_embedding(int vocab_size, int dim) {
    if (embedding_matrix) free(embedding_matrix);
    if (grad_embedding_matrix) free(grad_embedding_matrix);
    if (m_embedding_matrix) free(m_embedding_matrix);
    if (v_embedding_matrix) free(v_embedding_matrix);
    embedding_matrix = NULL;
    grad_embedding_matrix = NULL;
    m_embedding_matrix = NULL;
    v_embedding_matrix = NULL;
}

void adam_update_projection(int dim, int vocab_size, float lr, float beta1, float beta2, float eps, int t) {
    for (int i = 0; i < dim * vocab_size; ++i) {
        m_proj_W[i] = beta1 * m_proj_W[i] + (1.0f - beta1) * grad_proj_W[i];
        v_proj_W[i] = beta2 * v_proj_W[i] + (1.0f - beta2) * grad_proj_W[i] * grad_proj_W[i];
        float m_hat = m_proj_W[i] / (1.0f - powf(beta1, t));
        float v_hat = v_proj_W[i] / (1.0f - powf(beta2, t));
        proj_W[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
        grad_proj_W[i] = 0.0f;
    }
    for (int i = 0; i < vocab_size; ++i) {
        m_proj_b[i] = beta1 * m_proj_b[i] + (1.0f - beta1) * grad_proj_b[i];
        v_proj_b[i] = beta2 * v_proj_b[i] + (1.0f - beta2) * grad_proj_b[i] * grad_proj_b[i];
        float m_hat = m_proj_b[i] / (1.0f - powf(beta1, t));
        float v_hat = v_proj_b[i] / (1.0f - powf(beta2, t));
        proj_b[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
        grad_proj_b[i] = 0.0f;
    }
}

void adam_update_embedding(int vocab_size, int dim, float lr, float beta1, float beta2, float eps, int t) {
    for (int i = 0; i < vocab_size * dim; ++i) {
        m_embedding_matrix[i] = beta1 * m_embedding_matrix[i] + (1.0f - beta1) * grad_embedding_matrix[i];
        v_embedding_matrix[i] = beta2 * v_embedding_matrix[i] + (1.0f - beta2) * grad_embedding_matrix[i] * grad_embedding_matrix[i];
        float m_hat = m_embedding_matrix[i] / (1.0f - powf(beta1, t));
        float v_hat = v_embedding_matrix[i] / (1.0f - powf(beta2, t));
        embedding_matrix[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
        grad_embedding_matrix[i] = 0.0f;
    }
} 