# Makefile for Crossover Operations
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -g

# Source files
SOURCES = crossover_base.cc \
          one_point_crossover.cc \
          two_point_crossover.cc \
          uniform_crossover.cc \
          intermediate_recombination.cc \
          blend_crossover.cc \
          simulated_binary_crossover.cc \
          order_crossover.cc \
          partially_mapped_crossover.cc \
          cycle_crossover.cc \
          crossover_all.cc

# Object files
OBJECTS = $(SOURCES:.cc=.o)

# Header files
HEADERS = crossover_base.h \
          one_point_crossover.h \
          two_point_crossover.h \
          uniform_crossover.h \
          intermediate_recombination.h \
          blend_crossover.h \
          simulated_binary_crossover.h \
          order_crossover.h \
          partially_mapped_crossover.h \
          cycle_crossover.h \
          crossover_all.h

# Library name
LIBRARY = libcrossover.a

# Default target
all: $(LIBRARY)

# Create static library
$(LIBRARY): $(OBJECTS)
	ar rcs $@ $^
	@echo "Crossover library created successfully!"

# Compile source files
%.o: %.cc $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build files
clean:
	rm -f $(OBJECTS) $(LIBRARY)
	@echo "Build files cleaned!"

# Install headers and library (optional)
install: $(LIBRARY)
	@echo "Installing crossover library..."
	mkdir -p ../include/crossover
	cp $(HEADERS) ../include/crossover/
	cp $(LIBRARY) ../lib/
	@echo "Installation complete!"

# Test target (compile test program)
test: test_crossover.cc $(LIBRARY)
	$(CXX) $(CXXFLAGS) -o test_crossover test_crossover.cc -L. -lcrossover

# Show library contents
info:
	@echo "Library contents:"
	ar -t $(LIBRARY)

.PHONY: all clean install test info
