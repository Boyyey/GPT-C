#include "tokenizer.h"
#include <stdlib.h>
#include <string.h>

void build_vocab() {
    // No-op for char-level ASCII
}

int* tokenize(const char* text, int* out_len) {
    int len = strlen(text);
    int* tokens = (int*)malloc(len * sizeof(int));
    for (int i = 0; i < len; ++i) {
        tokens[i] = (unsigned char)text[i];
    }
    *out_len = len;
    return tokens;
}

char* detokenize(const int* tokens, int len) {
    char* text = (char*)malloc((len + 1) * sizeof(char));
    for (int i = 0; i < len; ++i) {
        text[i] = (char)tokens[i];
    }
    text[len] = '\0';
    return text;
} 