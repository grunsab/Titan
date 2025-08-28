use clap::Parser;
use cozy_chess::Board;
use piebot::search::alphabeta::{Searcher, SearchParams};
use std::time::{Duration, Instant};

#[derive(Parser, Debug)]
#[command(name = "piebot-bench", version, about = "Benchmark PieBot search NPS with PST/NNUE")]
struct Args {
    /// FEN string or 'startpos'
    #[arg(long, default_value = "startpos")]
    fen: String,

    /// Threads
    #[arg(long, default_value_t = 1)]
    threads: usize,

    /// Movetime in milliseconds (ignored if depth is set)
    #[arg(long, default_value_t = 1000)]
    movetime: u64,

    /// Fixed search depth (overrides movetime when > 0)
    #[arg(long, default_value_t = 0)]
    depth: u32,

    /// Use NNUE (requires NNUE file)
    #[arg(long, default_value_t = false)]
    use_nnue: bool,

    /// Dense NNUE file (PIENNUE1)
    #[arg(long)]
    nnue_file: Option<String>,

    /// Quantized NNUE file (PIENNQ01)
    #[arg(long)]
    nnue_quant_file: Option<String>,

    /// Eval blend percent (0..100). 0=PST, 100=NNUE
    #[arg(long, default_value_t = 100)]
    blend: u8,

    /// Transposition table size in MB (approximate)
    #[arg(long, default_value_t = 64)]
    hash_mb: usize,
}

fn main() {
    env_logger::init();
    let args = Args::parse();
    let board = if args.fen == "startpos" { Board::default() } else { Board::from_fen(&args.fen, false).expect("valid FEN") };

    let mut s = Searcher::default();
    s.set_tt_capacity_mb(args.hash_mb);
    let mut p = SearchParams::default();
    p.use_tt = true; p.order_captures = true; p.use_history = true; p.threads = args.threads.max(1);
    if args.depth > 0 { p.depth = args.depth; } else { p.movetime = Some(Duration::from_millis(args.movetime)); }

    if args.use_nnue {
        s.set_use_nnue(true);
        s.set_eval_blend_percent(args.blend);
        if let Some(q) = args.nnue_quant_file.as_deref() {
            match piebot::eval::nnue::loader::QuantNnue::load_quantized(q) {
                Ok(model) => s.set_nnue_quant_model(model),
                Err(e) => eprintln!("failed to load quant NNUE: {}", e),
            }
        } else if let Some(d) = args.nnue_file.as_deref() {
            match piebot::eval::nnue::Nnue::load(d) {
                Ok(nn) => s.set_nnue_network(Some(nn)),
                Err(e) => eprintln!("failed to load dense NNUE: {}", e),
            }
        }
    }

    let t0 = Instant::now();
    // Ensure Rayon uses requested threads
    let res = if args.threads > 1 {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(args.threads).build().unwrap();
        pool.install(|| s.search_with_params(&board, p))
    } else {
        s.search_with_params(&board, p)
    };
    let dt = t0.elapsed();
    let nps = if dt.as_secs_f64() > 0.0 { res.nodes as f64 / dt.as_secs_f64() } else { 0.0 };
    println!("bestmove={} score_cp={} nodes={} elapsed={:.3}s nps={:.1}", res.bestmove.unwrap_or_else(|| "(none)".to_string()), res.score_cp, res.nodes, dt.as_secs_f64(), nps);
}
