#![cfg(feature = "board-pleco")]
use pleco::{Board as PlecoBoard, BitMove as PMove, Player, PieceType, Piece, SQ};
use crate::search::tt_pleco::{TtPleco, Entry as TtEntry, Bound as TtBound};
use std::sync::Arc;
use std::time::{Duration, Instant};
use rayon::prelude::*;
use crate::search::eval::{MATE_SCORE, DRAW_SCORE};

pub struct PlecoSearcher {
    nodes: u64,
    deadline: Option<Instant>,
    tt: Arc<TtPleco>,
    killers: Vec<[Option<PMove>; 2]>,
    history: Vec<i32>,
    threads: usize,
    use_killers: bool,
    use_lmr: bool,
    use_nullmove: bool,
    use_aspiration: bool,
    aspiration_window_cp: i32,
}

impl Default for PlecoSearcher { fn default() -> Self { Self { nodes: 0, deadline: None, tt: Arc::new(TtPleco::default()), killers: vec![[None,None];256], history: vec![0; 64*64*5], threads: 1, use_killers: true, use_lmr: true, use_nullmove: true, use_aspiration: true, aspiration_window_cp: 30 } } }

impl PlecoSearcher {
    pub fn clear(&mut self) { self.nodes = 0; self.killers.iter_mut().for_each(|k| *k = [None, None]); self.history.fill(0); self.tt.bump_generation(); }
    pub fn set_tt_capacity_mb(&mut self, mb: usize) { Arc::get_mut(&mut self.tt).map(|t| t.set_capacity_mb(mb)); }
    pub fn set_threads(&mut self, t: usize) { self.threads = t.max(1); }

    pub fn search_movetime(&mut self, board: &mut PlecoBoard, millis: u64, depth: u32) -> (Option<PMove>, i32, u64) {
        self.nodes = 0;
        self.deadline = Some(Instant::now() + Duration::from_millis(millis));
        let mut best: Option<PMove> = None; let mut best_score = -MATE_SCORE;
        let max_depth = if depth == 0 { 99 } else { depth };
        let mut last_score = 0;
        for d in 1..=max_depth {
            self.tt.bump_generation();
            let (bm, sc) = if self.use_aspiration && d > 1 {
                let window = self.aspiration_window_cp.max(10);
                let mut alpha = last_score - window;
                let mut beta = last_score + window;
                let (b1, s1) = self.root_iter_window(board, d, alpha, beta);
                if s1 <= alpha || s1 >= beta { self.root_iter(board, d) } else { (b1, s1) }
            } else {
                self.root_iter(board, d)
            };
            best = bm; best_score = sc; last_score = sc;
            if let Some(dl) = self.deadline { if Instant::now() >= dl { break; } }
        }
        (best, best_score, self.nodes)
    }

    #[inline]
    fn killer_bonus(&self, ply: usize, m: PMove) -> i32 {
        if !self.use_killers { return 0; }
        let p = ply.min(self.killers.len()-1);
        let slot = &self.killers[p];
        if slot[0] == Some(m) { 50 } else if slot[1] == Some(m) { 30 } else { 0 }
    }

    fn order_moves(&self, board: &PlecoBoard, moves: &mut Vec<PMove>, tt_best: Option<PMove>, ply: usize) {
        if let Some(ttm) = tt_best { if let Some(pos) = moves.iter().position(|&x| x == ttm) { let mv = moves.remove(pos); moves.insert(0, mv); } }
        if moves.len() <= 1 { return; }
        moves[1..].sort_by_key(|&m| {
            let cap = if m.is_capture() { 1 } else { 0 };
            let mvv = if cap == 1 { self.mvv_lva(board, m) } else { 0 };
            let hist = 0; // history omitted for now
            let kb = self.killer_bonus(ply, m);
            -(cap * 10 + kb + hist + mvv)
        });
    }

    #[inline]
    fn piece_value_cp(pt: PieceType) -> i32 {
        match pt { PieceType::P => 100, PieceType::N => 320, PieceType::B => 330, PieceType::R => 500, PieceType::Q => 900, _ => 0 }
    }

    #[inline]
    fn mvv_lva(&self, board: &PlecoBoard, m: PMove) -> i32 {
        let to = m.get_dest(); let from = m.get_src();
        let v_piece = board.piece_at_sq(to);
        let a_piece = board.piece_at_sq(from);
        let v = if v_piece != Piece::None { Self::piece_value_cp(v_piece.type_of()) } else { 0 };
        let a = if a_piece != Piece::None { Self::piece_value_cp(a_piece.type_of()) } else { 0 };
        v * 10 - a
    }

    fn root_iter(&mut self, board: &mut PlecoBoard, depth: u32) -> (Option<PMove>, i32) {
        self.root_iter_window(board, depth, -MATE_SCORE, MATE_SCORE)
    }

    fn root_iter_window(&mut self, board: &mut PlecoBoard, depth: u32, alpha0: i32, beta: i32) -> (Option<PMove>, i32) {
        let mut alpha = alpha0;
        let mut ml: Vec<PMove> = board.generate_moves().iter().copied().collect();
        if ml.is_empty() { return (None, self.eval_terminal(board)); }
        let tt_best = self.tt.get(board.zobrist()).and_then(|e| e.best);
        self.order_moves(board, &mut ml, tt_best, 0);
        // Root SMP split
        if self.threads > 1 && depth >= 3 && ml.len() >= 12 {
            let shared_tt = self.tt.clone();
            // PV seed: first move
            let first = ml[0];
            let mut b1 = board.clone(); b1.apply_move(first);
            let mut seed = Self { tt: shared_tt.clone(), ..Self::default() };
            seed.threads = 1; seed.use_killers = self.use_killers; seed.use_lmr = self.use_lmr; seed.use_nullmove = self.use_nullmove; seed.use_aspiration = self.use_aspiration; seed.aspiration_window_cp = self.aspiration_window_cp; seed.deadline = self.deadline;
            let mut best_sc = -seed.alphabeta(&mut b1, depth - 1, -beta, -alpha);
            self.nodes += seed.nodes;
            let mut best = first;
            let tails: Vec<PMove> = ml.into_iter().skip(1).collect();
            let results: Vec<(PMove, i32, u64)> = tails.par_iter().map(|&m| {
                let mut c = board.clone(); c.apply_move(m);
                let mut w = Self { tt: shared_tt.clone(), ..Self::default() };
                w.threads = 1; w.use_killers = self.use_killers; w.use_lmr = self.use_lmr; w.use_nullmove = self.use_nullmove; w.use_aspiration = self.use_aspiration; w.aspiration_window_cp = self.aspiration_window_cp; w.deadline = self.deadline;
                let a = alpha;
                let score = -w.alphabeta(&mut c, depth - 1, -beta, -a);
                (m, score, w.nodes)
            }).collect();
            for (m, s, n) in results { self.nodes += n; if s > best_sc { best_sc = s; best = m; } }
            // Store root exact
            self.tt.put(TtEntry { key: board.zobrist(), depth, score: best_sc, best: Some(best), bound: TtBound::Exact, gen: 0 });
            return (Some(best), best_sc);
        }
        // Serial
        let mut best: Option<PMove> = None; let mut best_sc = -MATE_SCORE;
        for m in ml.iter() {
            board.apply_move(*m);
            let sc = -self.alphabeta(board, depth.saturating_sub(1), -beta, -alpha);
            board.undo_move();
            if sc > best_sc { best_sc = sc; best = Some(*m); }
            if sc > alpha { alpha = sc; }
            if let Some(dl) = self.deadline { if Instant::now() >= dl { break; } }
        }
        (best, best_sc)
    }

    fn alphabeta(&mut self, board: &mut PlecoBoard, depth: u32, mut alpha: i32, beta: i32) -> i32 {
        self.nodes += 1;
        if let Some(dl) = self.deadline { if Instant::now() >= dl { return self.eval(board); } }
        if depth == 0 { return self.qsearch(board, alpha, beta); }
        // Null-move pruning
        if self.use_nullmove && depth >= 3 && !board.in_check() {
            let mut nb = board.clone();
            // Pleco supports null moves via apply_null_move/undo_null_move if available
            let did_null = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe { nb.apply_null_move(); })).is_ok();
            if did_null {
                let r = 2 + (depth / 4) as u32;
                let sc = -self.alphabeta(&mut nb, depth - 1 - r, -beta, -beta + 1);
                if sc >= beta { return sc; }
            }
        }
        // TT probe
        if let Some(e) = self.tt.get(board.zobrist()) {
            if e.depth >= depth { match e.bound { TtBound::Exact => return e.score, TtBound::Lower => if e.score >= beta { return e.score; }, TtBound::Upper => if e.score <= alpha { return e.score; } } }
        }
        let mut ml: Vec<PMove> = board.generate_moves().iter().copied().collect();
        if ml.is_empty() { return self.eval_terminal(board); }
        let tt_best = self.tt.get(board.zobrist()).and_then(|e| e.best);
        self.order_moves(board, &mut ml, tt_best, (self.killers.len()-1).min(depth as usize));
        // In-tree split (jamboree-lite): PV seed + parallel tail
        if self.threads > 1 && depth >= 3 && ml.len() >= 12 {
            let shared_tt = self.tt.clone();
            // PV seed
            let first = ml[0];
            let mut b1 = board.clone(); b1.apply_move(first);
            let mut seed = Self { tt: shared_tt.clone(), ..Self::default() };
            seed.threads = 1; seed.use_killers = self.use_killers; seed.use_lmr = self.use_lmr; seed.use_nullmove = self.use_nullmove; seed.use_aspiration = self.use_aspiration; seed.aspiration_window_cp = self.aspiration_window_cp; seed.deadline = self.deadline;
            let mut best = -seed.alphabeta(&mut b1, depth - 1, -beta, -alpha);
            self.nodes += seed.nodes;
            let mut best_move_local: Option<PMove> = Some(first);
            use std::sync::atomic::{AtomicI32, Ordering};
            let alpha_shared = AtomicI32::new(best);
            let tails: Vec<PMove> = ml.into_iter().skip(1).collect();
            let results: Vec<(PMove, i32, u64)> = tails.par_iter().map(|&m| {
                let mut c = board.clone(); c.apply_move(m);
                let mut w = Self { tt: shared_tt.clone(), ..Self::default() };
                w.threads = 1; w.use_killers = self.use_killers; w.use_lmr = self.use_lmr; w.use_nullmove = self.use_nullmove; w.use_aspiration = self.use_aspiration; w.aspiration_window_cp = self.aspiration_window_cp; w.deadline = self.deadline;
                let a = alpha_shared.load(Ordering::Relaxed);
                let sc = -w.alphabeta(&mut c, depth - 1, -beta, -a);
                // update alpha
                let mut cur = a;
                while sc > cur {
                    match alpha_shared.compare_exchange(cur, sc, Ordering::Relaxed, Ordering::Relaxed) {
                        Ok(_) => break,
                        Err(obs) => { if obs >= sc { break; } cur = obs; }
                    }
                }
                (m, sc, w.nodes)
            }).collect();
            for (m, s, n) in results { self.nodes += n; if s > best { best = s; best_move_local = Some(m); } }
            // Store
            self.tt.put(TtEntry { key: board.zobrist(), depth, score: best, best: best_move_local, bound: TtBound::Exact, gen: 0 });
            return best;
        }

        let mut bestmove: Option<PMove> = None;
        for (i, m) in ml.iter().enumerate() {
            board.apply_move(*m);
            let sc = if self.use_lmr && depth >= 3 && !m.is_capture() && i >= 3 {
                let red = -self.alphabeta(board, depth - 2, -alpha - 1, -alpha);
                if red > alpha { -self.alphabeta(board, depth - 1, -beta, -alpha) } else { red }
            } else {
                -self.alphabeta(board, depth - 1, -beta, -alpha)
            };
            board.undo_move();
            if sc >= beta {
                self.tt.put(TtEntry { key: board.zobrist(), depth, score: sc, best: Some(*m), bound: TtBound::Lower, gen: 0 });
                if self.use_killers {
                    let ply = (self.killers.len()-1).min(depth as usize);
                    let k = &mut self.killers[ply]; if k[0] != Some(*m) { k[1] = k[0]; k[0] = Some(*m); }
                }
                return beta;
            }
            if sc > alpha { alpha = sc; bestmove = Some(*m); }
        }
        let bound = if bestmove.is_some() { TtBound::Exact } else { TtBound::Upper };
        self.tt.put(TtEntry { key: board.zobrist(), depth, score: alpha, best: bestmove, bound, gen: 0 });
        alpha
    }

    fn qsearch(&mut self, board: &mut PlecoBoard, mut alpha: i32, beta: i32) -> i32 {
        let stand = self.eval(board);
        if stand >= beta { return beta; }
        if stand > alpha { alpha = stand; }
        let mut caps: Vec<PMove> = board.generate_moves().iter().copied().filter(|m| m.is_capture()).collect();
        caps.sort_by_key(|&m| -self.mvv_lva(board, m));
        for m in caps.into_iter() {
            board.apply_move(m);
            let sc = -self.qsearch(board, -beta, -alpha);
            board.undo_move();
            if sc >= beta { return beta; }
            if sc > alpha { alpha = sc; }
        }
        alpha
    }

    fn eval(&self, board: &PlecoBoard) -> i32 {
        // Simple material count for prototype
        use pleco::PieceType::*;
        let mut score = 0i32;
        for &(p, v) in &[(PieceType::P,100),(PieceType::N,320),(PieceType::B,330),(PieceType::R,500),(PieceType::Q,900)] {
            score += board.count_piece(Player::White, p) as i32 * v;
            score -= board.count_piece(Player::Black, p) as i32 * v;
        }
        if board.turn() == Player::White { score } else { -score }
    }

    fn eval_terminal(&self, board: &PlecoBoard) -> i32 {
        if board.in_check() { -MATE_SCORE } else { DRAW_SCORE }
    }
}
