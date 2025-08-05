# PieBot Rust Implementation

A high-performance chess engine based on AlphaZero, ported from Python to Rust for significant speed improvements.

## Features

- Full AlphaZero neural network architecture
- Monte Carlo Tree Search (MCTS) with parallel rollouts
- Thread-safe implementation using Rust's ownership system
- PyTorch model compatibility via tch-rs
- Cross-platform support (CPU, CUDA, Apple Silicon)

## Building

```bash
cd piebot
cargo build --release
```

## Running

### Play against the engine

```bash
cargo run --bin piebot -- --model weights/AlphaZeroNet_20x256_rust.pt
```

### Command-line options

- `--model PATH`: Path to the PyTorch model file (.pt)
- `--mode MODE`: Game mode - 'h' for human, 's' for self-play, 'p' for profile
- `--color COLOR`: Your color when playing - 'w' for white, 'b' for black
- `--rollouts N`: Number of MCTS rollouts per move (default: 30)
- `--threads N`: Number of threads per rollout (default: 60)
- `--verbose`: Print search statistics
- `--fen FEN`: Starting position in FEN notation

## Testing

### Run unit tests

```bash
cargo test
```

### Compare with Python engine

```bash
# First, copy model weights from Python project
cp -r ../PieBot_v1/weights .

# Run comparison test
cargo test --test compare_engines test_single_position --release -- --nocapture
```

### Run full engine comparison

```bash
cargo test --test compare_engines test_engine_consistency --release -- --ignored --nocapture
```

## Performance

The Rust implementation provides significant performance improvements:
- ~5-10x faster MCTS rollouts
- Better memory efficiency
- True parallel search with thread safety
- Zero-copy tensor operations where possible

## Architecture

- `src/encoder.rs`: Board encoding/decoding for neural network
- `src/network.rs`: AlphaZero neural network implementation
- `src/mcts.rs`: Monte Carlo Tree Search algorithm
- `src/device_utils.rs`: Device detection and optimization
- `src/chess_openings.rs`: Standard chess openings for testing
- `src/main.rs`: Main chess playing interface

## Dependencies

- `chess`: Chess move generation and board representation
- `tch`: PyTorch C++ API bindings for Rust
- `ndarray`: N-dimensional array operations
- `rayon`: Data parallelism
- `clap`: Command-line argument parsing