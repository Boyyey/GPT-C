#ifndef MODEL_H
#define MODEL_H

#include "tensor.h"

// Embedding
extern float* embedding_matrix;
extern float* grad_embedding_matrix;
extern float* m_embedding_matrix;
extern float* v_embedding_matrix;
void init_embedding(int vocab_size, int dim);
void free_embedding(int vocab_size, int dim);
void embed_tokens(const int* tokens, int len, int dim, Tensor* out);
void adam_update_embedding(int vocab_size, int dim, float lr, float beta1, float beta2, float eps, int t);

// Transformer block
#define NUM_HEADS 2
#define MLP_HIDDEN 16
#define NUM_LAYERS 2
#define BATCH_SIZE 8

typedef struct {
    // Attention weights
    float* W_q; float* grad_W_q; float* m_W_q; float* v_W_q;
    float* W_k; float* grad_W_k; float* m_W_k; float* v_W_k;
    float* W_v; float* grad_W_v; float* m_W_v; float* v_W_v;
    float* W_o; float* grad_W_o; float* m_W_o; float* v_W_o;
    // MLP weights
    float* W1; float* grad_W1; float* m_W1; float* v_W1;
    float* b1; float* grad_b1; float* m_b1; float* v_b1;
    float* W2; float* grad_W2; float* m_W2; float* v_W2;
    float* b2; float* grad_b2; float* m_b2; float* v_b2;
    // Layer norm params
    float* ln1_gamma; float* grad_ln1_gamma; float* m_ln1_gamma; float* v_ln1_gamma;
    float* ln1_beta;  float* grad_ln1_beta;  float* m_ln1_beta;  float* v_ln1_beta;
    float* ln2_gamma; float* grad_ln2_gamma; float* m_ln2_gamma; float* v_ln2_gamma;
    float* ln2_beta;  float* grad_ln2_beta;  float* m_ln2_beta;  float* v_ln2_beta;
    int dim;
} TransformerBlock;

// Model with multiple layers
typedef struct {
    TransformerBlock* blocks;
    int num_layers;
    int dim;
} Model;

void init_transformer_block(TransformerBlock* block, int dim);
void free_transformer_block(TransformerBlock* block);
void transformer_block_forward(TransformerBlock* block, Tensor* input, Tensor* output);

void init_model(Model* model, int num_layers, int dim);
void free_model(Model* model);
void model_forward(Model* model, Tensor* input, Tensor* output);

// Output projection
extern float* proj_W; // [dim, vocab_size]
extern float* proj_b; // [vocab_size]
extern float* grad_proj_W; // [dim, vocab_size]
extern float* grad_proj_b; // [vocab_size]
extern float* m_proj_W; extern float* v_proj_W;
extern float* m_proj_b; extern float* v_proj_b;
void init_projection(int dim, int vocab_size);
void free_projection();
void project_logits(const float* input, int dim, float* logits, int vocab_size);
int sample_logits(const float* logits, int vocab_size);
void sgd_update_projection(int dim, int vocab_size, float lr);
void adam_update_projection(int dim, int vocab_size, float lr, float beta1, float beta2, float eps, int t);

// Save/load
void save_model(const char* path, Model* model, int dim, int vocab_size);
void load_model(const char* path, Model* model, int dim, int vocab_size);

// Training
void train(Model* model, int dim, int vocab_size, const char* filename, int epochs, float lr, int use_adam);

// Chat loop
void chat_loop(Model* model, int dim, int max_gen);

void positional_encoding(Tensor* t);
void layer_norm(Tensor* t, float* gamma, float* beta);
void attention(Tensor* Q, Tensor* K, Tensor* V, Tensor* output);

// === Backpropagation for Transformer Block ===
void transformer_block_backward(TransformerBlock* block, Tensor* input, Tensor* output, Tensor* d_output, Tensor* d_input);
void mlp_backward(TransformerBlock* block, Tensor* input, Tensor* d_output, Tensor* d_input);
void layer_norm_backward(TransformerBlock* block, Tensor* input, float* gamma, float* beta, Tensor* d_output, Tensor* d_input, float* grad_gamma, float* grad_beta);
void attention_backward(TransformerBlock* block, Tensor* input, Tensor* d_output, Tensor* d_input);

float gelu(float x);

#endif 