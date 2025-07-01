CC = gcc
CFLAGS = -O2 -Wall -I./src
SRC = src/main.c src/tensor.c src/tokenizer.c src/utils.c src/model.c
OUT = gpt_c

all: $(OUT)

$(OUT): $(SRC)
	$(CC) $(CFLAGS) -o $(OUT) $(SRC) -lm

clean:
	rm -f $(OUT) 