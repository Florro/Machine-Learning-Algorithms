CXX = g++
CXXFLAGS = -Wall -O3 -std=c++11 -march=native -fopenmp

# comment the following flags if you do not want to use OpenMP
#DFLAG += -DUSEOMP
#CXXFLAGS += -fopenmp

all: run-ftrl

run-ftrl: train.cpp data.o ftrl.o
	$(CXX) $(CXXFLAGS) -o $@ $^

data.o: data.cpp data.hpp
	$(CXX) $(CXXFLAGS) $(DFLAG) -c -o $@ $<

ftrl.o: ftrl.cpp ftrl.hpp
	$(CXX) $(CXXFLAGS) $(DFLAG) -c -o $@ $<
clean:
	rm -f run-ftrl data.o ftrl.o train.o
