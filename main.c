#include "tensor.h"
#include "tokenizer.h"
#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    // Example text
    const char* text = "hello world!";
    build_vocab();
    int tok_len;
    int* tokens = tokenize(text, &tok_len);
    printf("Tokenized: ");
    for (int i = 0; i < tok_len; ++i) printf("%d ", tokens[i]);
    printf("\n");

    // Embedding
    int dim = 8;
    init_embedding(VOCAB_SIZE, dim);
    int shape[2] = {tok_len, dim};
    Tensor* input = tensor_create(shape, 2);
    embed_tokens(tokens, tok_len, dim, input);
    positional_encoding(input);

    // Model with multiple transformer layers
    Model model;
    init_model(&model, NUM_LAYERS, dim);
    Tensor* output = tensor_create(shape, 2);
    model_forward(&model, input, output);

    printf("\nOutput tensor:\n");
    tensor_print(output);

    // Projection layer
    init_projection(dim, VOCAB_SIZE);

    // Training from file
    printf("\nTraining from data/train.txt (create this file with your text) ...\n");
    train(&model, dim, VOCAB_SIZE, "data/train.txt", 2, 0.01f, 1); // use_adam=1
    save_model("model.bin", &model, dim, VOCAB_SIZE);
    load_model("model.bin", &model, dim, VOCAB_SIZE);

    chat_loop(&model, dim, 32);
    free_projection();
    free_embedding(VOCAB_SIZE, dim);

    tensor_free(input);
    tensor_free(output);
    free_model(&model);
    free(tokens);
    return 0;
} 