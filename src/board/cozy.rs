use cozy_chess::{Board as CozyBoard, Move as CozyMove, Color, GameStatus, Piece, Square};

#[derive(Clone, Debug)]
pub struct Position {
    board: CozyBoard,
}

impl Position {
    pub fn startpos() -> Self {
        Self { board: CozyBoard::default() }
    }

    pub fn from_fen(fen: &str) -> Result<Self, String> {
        CozyBoard::from_fen(fen, false).map(|b| Self { board: b }).map_err(|e| format!("FEN error: {e:?}"))
    }

    pub fn board(&self) -> &CozyBoard { &self.board }

    pub fn make_move_uci(&mut self, _mv_uci: &str) -> Result<(), String> {
        // TODO: Implement UCI parsing using cozy-chess once wired.
        Err("UCI move parsing not implemented yet".to_string())
    }

    pub fn legal_moves_count(&self) -> usize {
        let mut ct = 0usize;
        self.board.generate_moves(|moves| { ct += moves.len(); false });
        ct
    }

    pub fn side_to_move(&self) -> Color { self.board.side_to_move() }

    pub fn set_from_start_and_moves(moves: &[String]) -> Result<Self, String> {
        let mut pos = Self::startpos();
        for m in moves { pos.make_move_uci(m)?; }
        Ok(pos)
    }
}
