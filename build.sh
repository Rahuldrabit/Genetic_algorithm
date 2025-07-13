#!/bin/bash

# Genetic Algorithm Framework Build Script
# This script provides a simple way to build the project using CMake

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if CMake is available
check_cmake() {
    if ! command -v cmake &> /dev/null; then
        print_error "CMake is not installed. Please install CMake first."
        echo "Ubuntu/Debian: sudo apt install cmake"
        echo "macOS: brew install cmake"
        echo "Windows: Download from https://cmake.org/download/"
        exit 1
    fi
    print_success "CMake found: $(cmake --version | head -n1)"
}

# Check if C++ compiler is available
check_compiler() {
    if command -v g++ &> /dev/null; then
        print_success "GCC found: $(g++ --version | head -n1)"
    elif command -v clang++ &> /dev/null; then
        print_success "Clang found: $(clang++ --version | head -n1)"
    else
        print_error "No C++ compiler found. Please install a C++17 compatible compiler."
        echo "Ubuntu/Debian: sudo apt install build-essential"
        echo "macOS: xcode-select --install"
        exit 1
    fi
}

# Parse command line arguments
BUILD_TYPE="Release"
CLEAN_BUILD=false
RUN_TESTS=false
INSTALL=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --run)
            RUN_TESTS=true
            shift
            ;;
        --install)
            INSTALL=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            echo "Genetic Algorithm Framework Build Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --debug     Build in debug mode"
            echo "  --clean     Clean build directory before building"
            echo "  --run       Run tests after building"
            echo "  --install   Install to system after building"
            echo "  --verbose   Verbose output"
            echo "  --help, -h  Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Build in release mode"
            echo "  $0 --debug --run      # Build in debug mode and run tests"
            echo "  $0 --clean --verbose  # Clean build with verbose output"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Main build function
main() {
    print_status "Starting build process..."
    
    # Check prerequisites
    check_cmake
    check_compiler
    
    # Create build directory
    if [ "$CLEAN_BUILD" = true ]; then
        print_status "Cleaning build directory..."
        rm -rf build
    fi
    
    mkdir -p build
    cd build
    
    # Configure with CMake
    print_status "Configuring with CMake (Build type: $BUILD_TYPE)..."
    CMAKE_ARGS="-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
    if [ "$VERBOSE" = true ]; then
        CMAKE_ARGS="$CMAKE_ARGS --debug-output"
    fi
    
    cmake .. $CMAKE_ARGS
    
    # Build the project
    print_status "Building project..."
    BUILD_ARGS=""
    if [ "$VERBOSE" = true ]; then
        BUILD_ARGS="--verbose"
    fi
    
    cmake --build . $BUILD_ARGS
    
    print_success "Build completed successfully!"
    
    # Run tests if requested
    if [ "$RUN_TESTS" = true ]; then
        print_status "Running tests..."
        if [ -f "bin/simple_ga_test" ]; then
            ./bin/simple_ga_test
        else
            print_error "Executable not found: bin/simple_ga_test"
            exit 1
        fi
    fi
    
    # Install if requested
    if [ "$INSTALL" = true ]; then
        print_status "Installing to system..."
        sudo cmake --build . --target install
        print_success "Installation completed!"
    fi
    
    print_success "All tasks completed successfully!"
    print_status "Executable location: build/bin/simple_ga_test"
}

# Run main function
main "$@" 