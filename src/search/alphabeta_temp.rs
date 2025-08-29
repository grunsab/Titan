// Temporary search implementation for A/B comparisons.
// By default this re-exports the baseline alphabeta searcher so the
// project always compiles. To try a modified search, copy the contents of
// `alphabeta.rs` to this file and iterate here. The compare runner can then
// pit `alphabeta` vs `alphabeta_temp` head-to-head.

pub use crate::search::alphabeta::*;

