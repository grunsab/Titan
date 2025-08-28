#[cfg(feature = "board-pleco")]
use clap::Parser;
#[cfg(feature = "board-pleco")]
use std::time::Instant;

#[cfg(feature = "board-pleco")]
#[derive(Parser, Debug)]
#[command(name = "piebot-bench-pleco", version, about = "Benchmark with Pleco board (make/unmake)")]
struct Args {
    #[arg(long, default_value = "startpos")] fen: String,
    #[arg(long, default_value_t = 4)] threads: usize,
    #[arg(long, default_value_t = 2000)] movetime: u64,
    #[arg(long, default_value_t = 6)] depth: u32,
}

#[cfg(feature = "board-pleco")]
fn main_inner() {
    let args = Args::parse();
    let mut board = if args.fen == "startpos" { pleco::Board::start_pos() } else { pleco::Board::from_fen(&args.fen).expect("valid fen") };
    let pool = rayon::ThreadPoolBuilder::new().num_threads(args.threads).build().unwrap();
    let t0 = Instant::now();
    let (bm, sc, nodes) = pool.install(|| {
        let mut s = piebot::search::alphabeta_pleco::PlecoSearcher::default();
        s.set_threads(args.threads);
        s.search_movetime(&mut board, args.movetime, args.depth)
    });
    let dt = t0.elapsed();
    println!("bestmove={:?} score_cp={} nodes={} elapsed={:.3}s nps={:.1}", bm, sc, nodes, dt.as_secs_f64(), nodes as f64 / dt.as_secs_f64());
}

#[cfg(not(feature = "board-pleco"))]
fn main() {
    eprintln!("bench_pleco requires --features board-pleco");
}

#[cfg(feature = "board-pleco")]
fn main() { main_inner(); }
