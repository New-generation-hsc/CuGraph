CC := nvcc
FLAGS := -O3 -std=c++11

SOURCE := ../../../buffer.cu ../../../graphio.cpp ../../../query.cu ../../../utils.cpp ../skelecton.cu main.cu compute.cu

TARGET := bfs

all : bfs

bfs :
	$(CC) $(FLAGS) $(SOURCE) -o $(TARGET)

clean :
	rm $(TARGET) out.txt