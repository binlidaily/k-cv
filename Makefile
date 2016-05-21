CXX ?= g++
CFLAGS = -Wall -Wconversion -O3 -fPIC
SHVER = 2
OS = $(shell uname)

all: svm-train svm-predict svm-scale

lib: svm.o
	if [ "$(OS)" = "Darwin" ]; then \
		SHARED_LIB_FLAG="-dynamiclib -Wl,-install_name,libsvm.so.$(SHVER)"; \
	else \
		SHARED_LIB_FLAG="-shared -Wl,-soname,libsvm.so.$(SHVER)"; \
	fi; \
	$(CXX) $${SHARED_LIB_FLAG} svm.o rpi.o -o libsvm.so.$(SHVER)

svm-predict: svm-predict.c svm.o
	$(CXX) $(CFLAGS) svm-predict.c svm.o rpi.o -o svm-predict -lm -O2  
svm-train: svm-train.c svm.o rpi.o
	$(CXX) $(CFLAGS) svm-train.c svm.o rpi.o -o svm-train -lm -O2  
svm-scale: svm-scale.c
	$(CXX) $(CFLAGS) svm-scale.c -o svm-scale
rpi.o: rpi.cpp rpi.h svm.h svm.o
	$(CXX) $(CFLAGS) -g -c rpi.cpp -O2   
svm.o: svm.cpp svm.h rpi.h
	$(CXX) $(CFLAGS) -g -c svm.cpp -O2  
clean:
	rm -f *~ svm.o svm-train svm-predict svm-scale libsvm.so.$(SHVER)
