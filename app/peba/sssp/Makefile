CC := nvcc
FLAGS := -O3 -std=c++11

SOURCE := ../../../buffer.cu ../../../graphio.cpp ../../../query.cu ../../../utils.cpp main.cu compute.cu

TARGET := sssp

all : sssp

sssp :
	$(CC) $(FLAGS) $(SOURCE) -o $(TARGET)

clean :
	rm $(TARGET) out.txt