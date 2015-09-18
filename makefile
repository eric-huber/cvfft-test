CC=g++
#CFLAGS+=-g
CXXFLAGS += `pkg-config --cflags opencv`
CXXFLAGS += -std=c++11
LDFLAGS  += `pkg-config --libs opencv`
LDFLAGS  += -lboost_program_options

PROG = cvfft-test
OBJS = main.o

.PHONY: all clean
$(PROG): $(OBJS)
	$(CC) -o $(PROG) $(OBJS) $(LDFLAGS)

%.o: %.cc
	$(CC) -c $(CXXFLAGS) $<

all: $(PROG)

clean:
	rm -f $(OBJS) $(PROG)