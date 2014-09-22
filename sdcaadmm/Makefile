CXX = mpic++
CC  = mpicc
CFLAGS = -Wall -Wconversion -O3 -fPIC -D_FILE_OFFSET_BITS=64
LIBS = blas/blas.a zlib/libz.a
SHVER = 1
#LIBS = -lblas

all: parallel-train blocksplit  parallel-predict

lib: linear.o tron.o blas/blas.a
	$(CXX) -shared -dynamiclib linear.o tron.o $(LIBS) -o liblinear.so.$(SHVER)

parallel-train: tron.o linear.o parallel-train.cpp block.o $(LIBS) block.h
	$(CXX) $(CFLAGS) -o parallel-train parallel-train.cpp block.o tron.o linear.o $(LIBS)

blocksplit: blocksplit.cpp $(LIBS)
	$(CXX) $(CFLAGS) -o blocksplit blocksplit.cpp $(LIBS)

block.o: block.cpp block.h
	$(CXX) $(CFLAGS) -c -o block.o block.cpp

train: tron.o linear.o train.c $(LIBS)
	$(CXX) $(CFLAGS) -o train train.c tron.o linear.o $(LIBS)

parallel-predict: tron.o linear.o block.o parallel-predict.cpp $(LIBS)
	$(CXX) $(CFLAGS) -o parallel-predict parallel-predict.cpp tron.o linear.o block.o $(LIBS)

tron.o: tron.cpp tron.h
	$(CXX) $(CFLAGS) -c -o tron.o tron.cpp

linear.o: linear.cpp linear.h
	$(CXX) $(CFLAGS) -c -o linear.o linear.cpp

blas/blas.a:
	cd blas; make OPTFLAGS='$(CFLAGS)' CC='$(CC)';

zlib/libz.a:
	cd zlib; make libz.a;

clean:
	cd blas;	make clean
	cd zlib; make clean
	rm -f *~ tron.o linear.o block.o train  liblinear.so.$(SHVER) parallel-train blocksplit parallel-predict
