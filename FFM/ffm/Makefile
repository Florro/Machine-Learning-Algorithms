CXX = g++
CXXFLAGS = -Wall -O3 -std=c++11 -march=native

# comment the following flags if you do not want to use OpenMP
#DFLAG += -DUSEOMP
#CXXFLAGS += -fopenmp

all: run-ffm

run-ffm: main.cpp fmdata.o ffm.o
	$(CXX) $(CXXFLAGS) -o $@ $^

fmdata.o: fmdata.cpp fmdata.hpp
	$(CXX) $(CXXFLAGS) $(DFLAG) -c -o $@ $<

ffm.o: ffm.cpp ffm.hpp
	$(CXX) $(CXXFLAGS) $(DFLAG) -c -o $@ $<
clean:
	rm -f run-ffm fmdata.o ffm.o main.o
