use cozy_chess::{Board, Move};
use std::path::Path;

#[derive(Debug)]
pub struct NnueMeta {
    pub version: u32,
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
}

pub struct Nnue {
    pub meta: NnueMeta,
    // Placeholder weights; real implementation will store quantized layers
    pub weights: Vec<i8>,
}

impl Nnue {
    pub fn load<P: AsRef<Path>>(_path: P) -> anyhow::Result<Self> {
        // Stub: until we finalize the format, return a trivial network
        Ok(Self { meta: NnueMeta { version: 1, input_dim: 0, hidden_dim: 0, output_dim: 1 }, weights: vec![] })
    }

    pub fn evaluate(&self, _board: &Board) -> i32 {
        // Stub: return 0 cp until NNUE integrated
        0
    }

    pub fn refresh_accumulator(&mut self, _board: &Board) {
        // Stub: full recompute of accumulator
    }

    pub fn update_on_move(&mut self, _mv: Move) {
        // Stub: incremental update of accumulator
    }
}

