use cozy_chess::{Board, Move};
use crate::search::eval::{material_eval_cp, MATE_SCORE, DRAW_SCORE};

#[derive(Default, Debug, Clone, Copy)]
pub struct SearchParams {
    pub depth: u32,
}

#[derive(Default, Debug, Clone)]
pub struct SearchResult {
    pub bestmove: Option<String>,
    pub score_cp: i32,
}

#[derive(Default)]
pub struct Searcher {
    // Placeholder for future TT, time manager, etc.
}

impl Searcher {
    pub fn search_depth(&mut self, board: &Board, depth: u32) -> SearchResult {
        let mut alpha = -MATE_SCORE;
        let beta = MATE_SCORE;
        let mut bestmove: Option<Move> = None;
        let mut best_score = -MATE_SCORE;

        let mut any = false;
        board.generate_moves(|moves| {
            for m in moves {
                any = true;
                let mut child = board.clone();
                child.play(m);
                let score = -self.alphabeta(&child, depth.saturating_sub(1), -beta, -alpha, 1);
                if score > best_score {
                    best_score = score;
                    bestmove = Some(m);
                }
                if score > alpha { alpha = score; }
            }
            false
        });
        if !any {
            return SearchResult { bestmove: None, score_cp: self.eval_terminal(board, 0) };
        }

        let bestmove_uci = bestmove.map(|m| format!("{}", m));
        SearchResult { bestmove: bestmove_uci, score_cp: best_score }
    }

    fn alphabeta(&mut self, board: &Board, depth: u32, mut alpha: i32, beta: i32, ply: i32) -> i32 {
        // Terminal or leaf
        if depth == 0 {
            return material_eval_cp(board);
        }
        let mut has_any = false;
        let mut best = -MATE_SCORE;
        board.generate_moves(|moves| {
            for m in moves {
                has_any = true;
                let mut child = board.clone();
                child.play(m);
                let score = -self.alphabeta(&child, depth - 1, -beta, -alpha, ply + 1);
                if score > best { best = score; }
                if best > alpha { alpha = best; }
                if alpha >= beta { break; }
            }
            false
        });
        if !has_any { return self.eval_terminal(board, ply); }
        best
    }

    fn eval_terminal(&self, board: &Board, ply: i32) -> i32 {
        // No moves: checkmate or stalemate
        if !(board.checkers()).is_empty() {
            // Side to move is in check and has no moves -> mated
            return -MATE_SCORE + ply;
        }
        DRAW_SCORE
    }
}
