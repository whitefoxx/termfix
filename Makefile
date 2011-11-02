CXX = icpc
CXXFLAGS = -fast -w

all:	main classifier commen
	$(CXX) $(CXXFLAGS) tmp/main.o tmp/classifier.o tmp/commen.o -o bin/terminator
main:
	$(CXX) $(CXXFLAGS) -c src/main.cpp -o tmp/main.o
classifier:
	$(CXX) $(CXXFLAGS) -c src/classifier.cpp -o tmp/classifier.o
commen:
	$(CXX) $(CXXFLAGS) -c src/commen.cpp -o tmp/commen.o
clean:
	rm -rf tmp/* bin/*
