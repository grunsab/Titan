use cozy_chess::{Board, Move};
use crate::search::eval::{eval_cp, MATE_SCORE, DRAW_SCORE};
use std::time::{Duration, Instant};
use crate::search::zobrist;
use crate::search::tt::{Tt, Entry, Bound};
use std::sync::Arc;
use std::collections::HashMap;
use rayon::prelude::*;
use std::sync::atomic::{AtomicI32, Ordering};

#[derive(Default, Debug, Clone, Copy)]
pub struct SearchParams {
    pub depth: u32,
    pub use_tt: bool,
    pub max_nodes: Option<u64>,
    pub movetime: Option<Duration>,
    pub order_captures: bool,
    pub use_history: bool,
    pub threads: usize,
    pub use_aspiration: bool,
    pub aspiration_window_cp: i32,
    pub use_lmr: bool,
    pub use_killers: bool,
    pub use_nullmove: bool,
}

#[derive(Default, Debug, Clone)]
pub struct SearchResult {
    pub bestmove: Option<String>,
    pub score_cp: i32,
    pub nodes: u64,
}

pub struct Searcher {
    tt: Arc<Tt>,
    pub(crate) nodes: u64,
    node_limit: u64,
    deadline: Option<Instant>,
    history: HashMap<String, i32>,
    order_captures: bool,
    use_history: bool,
    threads: usize,
    abort: Option<Arc<std::sync::atomic::AtomicBool>>,
    killers: Vec<[Option<Move>; 2]>,
    cont_history: HashMap<(String, String), i32>,
    use_aspiration: bool,
    use_lmr: bool,
    use_killers: bool,
    use_nullmove: bool,
}

impl Default for Searcher {
    fn default() -> Self {
        let mut t = Tt::new();
        t.set_capacity_entries(4096);
        Self {
            tt: Arc::new(t),
            nodes: 0,
            node_limit: u64::MAX,
            deadline: None,
            history: HashMap::new(),
            order_captures: false,
            use_history: false,
            threads: 1,
            abort: None,
            killers: Vec::new(),
            cont_history: HashMap::new(),
            use_aspiration: false,
            use_lmr: false,
            use_killers: false,
            use_nullmove: false,
        }
    }
}

impl Searcher {
    pub fn see_gain_cp(&mut self, board: &Board, uci: &str) -> Option<i32> {
        // Locate a matching legal move by UCI string
        let mut chosen: Option<Move> = None;
        board.generate_moves(|ml| {
            for m in ml { if format!("{}", m) == uci { chosen = Some(m); break; } }
            chosen.is_some()
        });
        chosen.and_then(|m| crate::search::see::see_gain_cp(board, m))
    }

    pub fn qsearch_eval_cp(&mut self, board: &Board) -> i32 {
        self.qsearch(board, -MATE_SCORE, MATE_SCORE)
    }

    fn qsearch(&mut self, board: &Board, mut alpha: i32, beta: i32) -> i32 {
        // Stand pat
        let stand = eval_cp(board);
        if stand >= beta { return beta; }
        if stand > alpha { alpha = stand; }

        // Captures only
        let opp = if board.side_to_move() == cozy_chess::Color::White { cozy_chess::Color::Black } else { cozy_chess::Color::White };
        let opp_bb = board.colors(opp);
        let mut caps: Vec<Move> = Vec::new();
        board.generate_moves(|ml| {
            for m in ml {
                let mstr = format!("{}", m);
                let to = &mstr[2..4];
                let mut occ = false; for sq in opp_bb { if format!("{}", sq) == to { occ = true; break; } }
                if occ { caps.push(m); }
            }
            false
        });
        // Order captures by SEE descending
        caps.sort_by_key(|&m| -crate::search::see::see_gain_cp(board, m).unwrap_or(0));
        for m in caps {
            // SEE pruning: skip clearly losing captures
            if let Some(see) = crate::search::see::see_gain_cp(board, m) { if stand + see + 50 < alpha { continue; } }
            let mut child = board.clone();
            child.play(m);
            let score = -self.qsearch(&child, -beta, -alpha);
            if score >= beta { return beta; }
            if score > alpha { alpha = score; }
        }
        alpha
    }

    pub fn search_depth(&mut self, board: &Board, depth: u32) -> SearchResult {
        let mut alpha = -MATE_SCORE;
        let beta = MATE_SCORE;
        let mut bestmove: Option<Move> = None;
        let mut best_score = -MATE_SCORE;

        // Root-split parallel search if threads > 1 and depth > 1
        if self.threads > 1 && depth > 1 {
            return self.search_depth_parallel(board, depth);
        }

        let mut any = false;
        let orig_alpha = alpha;
        board.generate_moves(|moves| {
            for m in moves {
                any = true;
                let mut child = board.clone();
                child.play(m);
                let score = -self.alphabeta(&child, depth.saturating_sub(1), -beta, -alpha, 1);
                if score > best_score { best_score = score; bestmove = Some(m); }
                if score > alpha { alpha = score; }
            }
            false
        });
        if !any {
            return SearchResult { bestmove: None, score_cp: self.eval_terminal(board, 0), nodes: self.nodes };
        }

        // Store root in TT as exact when using full window
        let root_bound = if best_score <= orig_alpha { Bound::Upper } else if best_score >= beta { Bound::Lower } else { Bound::Exact };
        self.tt_put(board, depth, best_score, bestmove, root_bound);

        let bestmove_uci = bestmove.map(|m| format!("{}", m));
        SearchResult { bestmove: bestmove_uci, score_cp: best_score, nodes: self.nodes }
    }

    fn search_depth_parallel(&mut self, board: &Board, depth: u32) -> SearchResult {
        let mut moves: Vec<Move> = Vec::with_capacity(64);
        board.generate_moves(|ml| { for m in ml { moves.push(m); } false });
        if moves.is_empty() { return SearchResult { bestmove: None, score_cp: self.eval_terminal(board, 0), nodes: self.nodes }; }

        // Optional: TT move first (ordering only)
        if let Some(en) = self.tt_get(board) {
            if let Some(ttm) = en.best {
                if let Some(pos) = moves.iter().position(|&mv| mv == ttm) {
                    let mv = moves.remove(pos);
                    moves.insert(0, mv);
                }
            }
        }

        // Evaluate each root move independently with full window in parallel
        let deadline = self.deadline;
        let order_captures = self.order_captures;
        let use_history = self.use_history;
        let shared_tt = self.tt.clone();
        let results: Vec<(Move, i32, u64)> = moves.par_iter().map(|&m| {
            let mut child = board.clone();
            child.play(m);
            let mut w = Searcher::default();
            w.node_limit = u64::MAX; // rely on shared deadline for stopping
            w.deadline = deadline;
            w.order_captures = order_captures;
            w.use_history = use_history;
            w.tt = shared_tt.clone();
            let score = -w.alphabeta(&child, depth - 1, -MATE_SCORE, MATE_SCORE, 1);
            (m, score, w.nodes)
        }).collect();

        // Reduce to best
        let mut best: Option<(Move, i32)> = None;
        let mut total_nodes = 0u64;
        for (m, s, n) in results {
            total_nodes += n;
            if best.map_or(true, |(_, bs)| s > bs) { best = Some((m, s)); }
        }
        self.nodes = total_nodes;
        if let Some((bm, sc)) = best {
            // Store TT root as exact
            self.tt_put(board, depth, sc, Some(bm), Bound::Exact);
            return SearchResult { bestmove: Some(format!("{}", bm)), score_cp: sc, nodes: self.nodes };
        }
        SearchResult { bestmove: None, score_cp: self.eval_terminal(board, 0), nodes: self.nodes }
    }

    fn alphabeta(&mut self, board: &Board, depth: u32, mut alpha: i32, beta: i32, ply: i32) -> i32 {
        if let Some(ref flag) = self.abort { if flag.load(Ordering::Relaxed) { return eval_cp(board); } }
        self.nodes += 1;
        if self.nodes >= self.node_limit { return eval_cp(board); }
        if let Some(dl) = self.deadline { if Instant::now() >= dl { return eval_cp(board); } }
        if depth == 0 { return self.qsearch(board, alpha, beta); }
        // Null-move pruning (guarded)
        if self.use_nullmove && depth >= 3 {
            // avoid in check
            if (board.checkers()).is_empty() {
                // null move
                let mut nb = board.clone();
                // cozy_chess null move; if unavailable, skip
                let null_ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    nb.null_move();
                })).is_ok();
                if null_ok {
                    let r = 2 + (depth / 4) as u32;
                    let score = -self.alphabeta(&nb, depth - 1 - r, -beta, -beta + 1, ply + 1);
                    if score >= beta { return score; }
                }
            }
        }

        // TT probe (exact-only)
        if let Some(en) = self.tt_get(board) {
            if en.depth >= depth {
                match en.bound {
                    Bound::Exact => return en.score,
                    Bound::Lower => if en.score >= beta { return en.score; },
                    Bound::Upper => if en.score <= alpha { return en.score; },
                }
            }
        }

        // Build movelist and order
        let mut moves: Vec<Move> = Vec::with_capacity(64);
        board.generate_moves(|ml| { for m in ml { moves.push(m); } false });
        if moves.is_empty() { return self.eval_terminal(board, ply); }
        // TT move first
        if let Some(en) = self.tt_get(board) {
            if let Some(ttm) = en.best {
                if let Some(pos) = moves.iter().position(|&mv| mv == ttm) {
                    let mv = moves.remove(pos);
                    moves.insert(0, mv);
                }
            }
        }
        // Captures-first, killers, and history ordering
        if self.order_captures || self.use_history || self.use_killers {
            let opp = if board.side_to_move() == cozy_chess::Color::White { cozy_chess::Color::Black } else { cozy_chess::Color::White };
            let opp_bb = board.colors(opp);
            moves.sort_by_key(|&m| {
                let m_str = format!("{}", m);
                let to_str = &m_str[2..4];
                let mut occ = false; for sq in opp_bb { if format!("{}", sq) == to_str { occ = true; break; } }
                let is_cap = if self.order_captures { if occ { 1 } else { 0 } } else { 0 };
                let hist = if self.use_history { *self.history.get(&m_str).unwrap_or(&0) } else { 0 };
                let parent_key = self.current_parent_key(ply);
                let ch = if self.use_history { *self.cont_history.get(&(parent_key.clone(), m_str.clone())).unwrap_or(&0) } else { 0 };
                let kb = if self.use_killers { self.killer_bonus(ply, m) } else { 0 };
                -(is_cap * 10 + kb + hist + ch)
            });
        }

        // In-tree split (jamboree-lite): PV seed + parallel tail with shared alpha
        if self.threads > 1 && depth >= 3 && moves.len() >= 12 {
            let shared_tt = self.tt.clone();
            let deadline = self.deadline;
            let order_captures = self.order_captures;
            let use_history = self.use_history;

            // PV seed: evaluate first move serially to get a strong alpha
            let mut iter = moves.into_iter();
            let first = iter.next().unwrap();
            let mut child = board.clone();
            child.play(first);
            let mut seed = Searcher::default();
            seed.node_limit = u64::MAX;
            seed.deadline = deadline;
            seed.order_captures = order_captures;
            seed.use_history = use_history;
            seed.tt = shared_tt.clone();
            let mut best = -seed.alphabeta(&child, depth - 1, -MATE_SCORE, MATE_SCORE, ply + 1);
            let mut best_move_local: Option<Move> = Some(first);
            self.nodes += seed.nodes;
            let alpha_shared = AtomicI32::new(best);
            let abort_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));

            // Parallel tail search with shared alpha
            let tails: Vec<Move> = iter.collect();
            let results: Vec<(Move, i32, u64)> = tails.par_iter().map(|&m| {
                let mut c = board.clone();
                c.play(m);
                let mut w = Searcher::default();
                w.node_limit = u64::MAX;
                w.deadline = deadline;
                w.order_captures = order_captures;
                w.use_history = use_history;
                w.tt = shared_tt.clone();
                w.abort = Some(abort_flag.clone());
                // Read current alpha
                let a = alpha_shared.load(Ordering::Relaxed);
                if abort_flag.load(Ordering::Relaxed) { return (m, -MATE_SCORE, 0); }
                let score = -w.alphabeta(&c, depth - 1, -MATE_SCORE, -a, ply + 1);
                // Update shared alpha if improved
                let mut cur = a;
                while score > cur {
                    match alpha_shared.compare_exchange(cur, score, Ordering::Relaxed, Ordering::Relaxed) {
                        Ok(_) => break,
                        Err(observed) => { if observed >= score { break; } cur = observed; }
                    }
                }
                if score >= beta { abort_flag.store(true, Ordering::Relaxed); }
                (m, score, w.nodes)
            }).collect();

            for (m, s, n) in results {
                self.nodes += n;
                if s > best { best = s; best_move_local = Some(m); }
            }
            // Store as exact at this node
            self.tt_put(board, depth, best, best_move_local, Bound::Exact);
            if let Some(mv) = best_move_local {
                let key = format!("{}", mv);
                *self.history.entry(key).or_insert(0) += (depth as i32) * (depth as i32);
            }
            return best;
        }

        let mut best = -MATE_SCORE;
        let mut best_move_local: Option<Move> = None;
        let orig_alpha = alpha;
        for (idx, m) in moves.into_iter().enumerate() {
            let mut child = board.clone();
            child.play(m);
            let mut score;
            if self.use_lmr && depth >= 3 {
                // Simple LMR: reduce late quiet moves
                let is_cap = self.is_capture(&board, m);
                if !is_cap && idx >= 3 {
                    let r = 1; // basic reduction
                    let red = -self.alphabeta(&child, depth - 1 - r, -alpha - 1, -alpha, ply + 1);
                    if red > alpha { score = -self.alphabeta(&child, depth - 1, -beta, -alpha, ply + 1); } else { score = red; }
                } else {
                    score = -self.alphabeta(&child, depth - 1, -beta, -alpha, ply + 1);
                }
            } else {
                score = -self.alphabeta(&child, depth - 1, -beta, -alpha, ply + 1);
            }
            if score > best { best = score; best_move_local = Some(m); }
            if best > alpha { alpha = best; }
            if alpha >= beta { break; }
            if let Some(dl) = self.deadline { if Instant::now() >= dl { break; } }
            // Record continuation history for improved ordering
            if self.use_history {
                let parent_key = self.current_parent_key(ply);
                let move_key = format!("{}", m);
                *self.cont_history.entry((parent_key, move_key)).or_insert(0) += 1;
            }
        }
        // Store exact score and best move
        let bound = if best <= orig_alpha { Bound::Upper } else if best >= beta { Bound::Lower } else { Bound::Exact };
        self.tt_put(board, depth, best, best_move_local, bound);
        if let Some(mv) = best_move_local {
            let key = format!("{}", mv);
            *self.history.entry(key).or_insert(0) += (depth as i32) * (depth as i32);
            if self.use_killers && bound == Bound::Lower {
                self.update_killers(ply, mv);
            }
        }
        best
    }

    fn eval_terminal(&self, board: &Board, ply: i32) -> i32 {
        if !(board.checkers()).is_empty() { return -MATE_SCORE + ply; }
        DRAW_SCORE
    }
}

impl Searcher {
    fn tt_key(board: &Board) -> u64 { zobrist::compute(board) }
    fn tt_get(&self, board: &Board) -> Option<Entry> { self.tt.get(Self::tt_key(board)) }
    fn tt_put(&mut self, board: &Board, depth: u32, score: i32, best: Option<Move>, bound: Bound) {
        let e = Entry { key: Self::tt_key(board), depth, score, best, bound, gen: 0 };
        self.tt.put(e);
    }

    pub fn search_with_params(&mut self, board: &Board, params: SearchParams) -> SearchResult {
        self.nodes = 0;
        self.node_limit = params.max_nodes.unwrap_or(u64::MAX);
        if !params.use_tt { self.tt = Arc::new(Tt::new()); }
        self.order_captures = params.order_captures;
        self.use_history = params.use_history;
        self.threads = params.threads.max(1);
        self.use_aspiration = params.use_aspiration;
        self.use_lmr = params.use_lmr;
        self.use_killers = params.use_killers;
        self.use_nullmove = params.use_nullmove;
        self.killers = vec![[None, None]; 256];
        let mut best: Option<String> = None;
        let mut last_score = 0;
        self.deadline = params.movetime.map(|d| Instant::now() + d);
        // Generation bump per iteration for TT aging
        for d in 1..=params.depth {
            self.tt.bump_generation();
            let r = if self.use_aspiration && d > 1 {
                let window = params.aspiration_window_cp.max(10);
                let mut alpha = last_score - window;
                let mut beta = last_score + window;
                let mut res = self.search_depth_window(board, d, alpha, beta);
                if res.score_cp <= alpha || res.score_cp >= beta {
                    // fallback to full window
                    res = self.search_depth(board, d);
                }
                res
            } else {
                self.search_depth(board, d)
            };
            best = r.bestmove.clone();
            last_score = r.score_cp;
            if self.nodes >= self.node_limit { break; }
            if let Some(dl) = self.deadline { if Instant::now() >= dl { break; } }
        }
        SearchResult { bestmove: best, score_cp: last_score, nodes: self.nodes }
    }

    fn search_depth_window(&mut self, board: &Board, depth: u32, alpha0: i32, beta0: i32) -> SearchResult {
        let mut alpha = alpha0;
        let beta = beta0;
        let mut bestmove: Option<Move> = None;
        let mut best_score = -MATE_SCORE;

        if self.threads > 1 && depth > 1 { return self.search_depth(board, depth); }

        let mut any = false;
        let orig_alpha = alpha;
        board.generate_moves(|moves| {
            for m in moves {
                any = true;
                let mut child = board.clone();
                child.play(m);
                let score = -self.alphabeta(&child, depth.saturating_sub(1), -beta, -alpha, 1);
                if score > best_score { best_score = score; bestmove = Some(m); }
                if score > alpha { alpha = score; }
            }
            false
        });
        if !any { return SearchResult { bestmove: None, score_cp: self.eval_terminal(board, 0), nodes: self.nodes }; }
        let bestmove_uci = bestmove.map(|m| format!("{}", m));
        SearchResult { bestmove: bestmove_uci, score_cp: best_score, nodes: self.nodes }
    }

    fn is_capture(&self, board: &Board, m: Move) -> bool {
        let opp = if board.side_to_move() == cozy_chess::Color::White { cozy_chess::Color::Black } else { cozy_chess::Color::White };
        let opp_bb = board.colors(opp);
        let m_str = format!("{}", m);
        let to_str = &m_str[2..4];
        for sq in opp_bb { if format!("{}", sq) == to_str { return true; } }
        false
    }

    fn update_killers(&mut self, ply: i32, m: Move) {
        let p = ply as usize;
        if p >= self.killers.len() { return; }
        let slot = &mut self.killers[p];
        if slot[0] == Some(m) { return; }
        if slot[1] == Some(m) {
            slot[1] = slot[0];
            slot[0] = Some(m);
            return;
        }
        slot[1] = slot[0];
        slot[0] = Some(m);
    }

    fn killer_bonus(&self, ply: i32, m: Move) -> i32 {
        let p = ply as usize;
        if p >= self.killers.len() { return 0; }
        let slot = &self.killers[p];
        if slot[0] == Some(m) { 50 } else if slot[1] == Some(m) { 30 } else { 0 }
    }

    fn current_parent_key(&self, ply: i32) -> String {
        // For simplicity, use ply number as parent context; can be improved to actual parent UCI
        format!("ply{}", ply)
    }

    pub fn tt_probe(&self, board: &Board) -> Option<(u32, Bound)> {
        self.tt_get(board).map(|e| (e.depth, e.bound))
    }

    pub fn set_tt_capacity_mb(&mut self, mb: usize) {
        let mut tt = Tt::new();
        tt.set_capacity_mb(mb);
        self.tt = Arc::new(tt);
    }
    pub fn get_threads(&self) -> usize { self.threads }
}
