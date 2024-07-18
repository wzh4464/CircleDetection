# Makefile for CircleDet project on macOS

# Compiler and Linker
CC = clang++
CFLAGS = -std=c++17 -Wall -Wextra -I/opt/homebrew/include -I/opt/homebrew/include/opencv4
LDFLAGS = -L/opt/homebrew/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc

# Build configurations
DEBUG_CFLAGS = -g -O0
RELEASE_CFLAGS = -O3

# Source and Header files
SRCS = ED.cpp EDColor.cpp EDPF.cpp test.cpp
HEADERS = DetInflexPt.h ED.h EDColor.h EDLib.h EDPF.h myTools.h NFA.h RDP.h Reject_sharp_turn.h

# Object files
OBJS = $(SRCS:.cpp=.o)

# Output executable
TARGET = CircleDet

# Default target
all: debug

# Debug build
debug: CFLAGS += $(DEBUG_CFLAGS)
debug: $(TARGET)

# Release build
release: CFLAGS += $(RELEASE_CFLAGS)
release: $(TARGET)

# Build target
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Compile source files
%.o: %.cpp $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build files
clean:
	rm -f $(OBJS) $(TARGET)

# PHONY targets
.PHONY: all debug release clean
