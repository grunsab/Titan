use cozy_chess::{Board, Color, Piece};

const PAWN: i32 = 100;
const KNIGHT: i32 = 320;
const BISHOP: i32 = 330;
const ROOK: i32 = 500;
const QUEEN: i32 = 900;

fn count_piece(board: &Board, color: Color, piece: Piece) -> i32 {
    let bb = board.colors(color) & board.pieces(piece);
    bb.into_iter().count() as i32
}

// Side-agnostic material in centipawns: positive means White has more material.
pub fn material_eval_cp_side_agnostic(board: &Board) -> i32 {
    let w = Color::White;
    let b = Color::Black;
    let score =
        (count_piece(board, w, Piece::Pawn) - count_piece(board, b, Piece::Pawn)) * PAWN +
        (count_piece(board, w, Piece::Knight) - count_piece(board, b, Piece::Knight)) * KNIGHT +
        (count_piece(board, w, Piece::Bishop) - count_piece(board, b, Piece::Bishop)) * BISHOP +
        (count_piece(board, w, Piece::Rook) - count_piece(board, b, Piece::Rook)) * ROOK +
        (count_piece(board, w, Piece::Queen) - count_piece(board, b, Piece::Queen)) * QUEEN;
    score
}

// Material from side-to-move perspective (negamax-friendly)
pub fn material_eval_cp(board: &Board) -> i32 {
    let base = material_eval_cp_side_agnostic(board);
    if board.side_to_move() == Color::White { base } else { -base }
}

// Mate scoring helpers
pub const MATE_SCORE: i32 = 30_000;
pub const DRAW_SCORE: i32 = 0;

