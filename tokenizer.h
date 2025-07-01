#ifndef TOKENIZER_H
#define TOKENIZER_H

#define VOCAB_SIZE 128 // ASCII

void build_vocab();
int* tokenize(const char* text, int* out_len);
char* detokenize(const int* tokens, int len);

#endif 