use anyhow::{Result, anyhow};
use ndarray::{Array3, Array1, s};
use chess::{Board, ChessMove, Piece, Color, Square, MoveGen, ALL_SQUARES, Rank, File};
use tch::{Tensor, Device, Kind};
use std::str::FromStr;

/// Parse game result string to integer
/// Returns 1 for white win, 0 for draw, -1 for black win
pub fn parse_result(result: &str) -> Result<i8> {
    match result {
        "1-0" => Ok(1),
        "1/2-1/2" => Ok(0),
        "0-1" => Ok(-1),
        _ => Err(anyhow!("Unexpected result string: {}", result)),
    }
}

/// Get piece type index for encoding
fn get_piece_plane_index(piece: Piece, color: Color) -> usize {
    let base = match piece {
        Piece::Pawn => 0,
        Piece::Rook => 2,
        Piece::Bishop => 4,
        Piece::Knight => 6,
        Piece::Queen => 8,
        Piece::King => 10,
    };
    base + if color == Color::Black { 1 } else { 0 }
}

/// Encode a chess position as a 16x8x8 tensor
/// First 12 planes represent different pieces
/// Last 4 planes represent castling rights
pub fn encode_position(board: &Board) -> Array3<f32> {
    let mut planes = Array3::<f32>::zeros((16, 8, 8));
    
    // Encode pieces
    for square in ALL_SQUARES.iter() {
        if let Some((piece, color)) = board.piece_on(*square).zip(board.color_on(*square)) {
            let plane_idx = get_piece_plane_index(piece, color);
            let rank = square.get_rank().to_index();
            let file = square.get_file().to_index();
            planes[[plane_idx, rank, file]] = 1.0;
        }
    }
    
    // Encode castling rights
    let castling = board.castle_rights(Color::White);
    if castling.has_kingside() {
        planes.slice_mut(s![12, .., ..]).fill(1.0);
    }
    if castling.has_queenside() {
        planes.slice_mut(s![14, .., ..]).fill(1.0);
    }
    
    let castling = board.castle_rights(Color::Black);
    if castling.has_kingside() {
        planes.slice_mut(s![13, .., ..]).fill(1.0);
    }
    if castling.has_queenside() {
        planes.slice_mut(s![15, .., ..]).fill(1.0);
    }
    
    planes
}

/// Convert a move to an index in the 72x8x8 policy representation
/// Returns (plane_index, from_rank, from_file)
pub fn move_to_idx(mv: ChessMove) -> (usize, usize, usize) {
    let from_square = mv.get_source();
    let to_square = mv.get_dest();
    
    let from_file_idx = from_square.get_file().to_index();
    let from_rank_idx = from_square.get_rank().to_index();
    let to_file_idx = to_square.get_file().to_index();
    let to_rank_idx = to_square.get_rank().to_index();
    
    let rank_diff = to_rank_idx as i32 - from_rank_idx as i32;
    let file_diff = to_file_idx as i32 - from_file_idx as i32;
    
    let plane_idx = match (rank_diff, file_diff) {
        // Horizontal moves (rook-like)
        (0, d) if d > 0 => d as usize - 1,  // Right: planes 0-6
        (0, d) if d < 0 => 7 + (-d) as usize,  // Left: planes 8-14
        
        // Vertical moves (rook-like)
        (d, 0) if d > 0 => 15 + d as usize,  // Up: planes 16-22
        (d, 0) if d < 0 => 23 + (-d) as usize,  // Down: planes 24-30
        
        // Diagonal moves (bishop-like)
        (d1, d2) if d1 == d2 && d1 > 0 => 31 + d1 as usize,  // Up-right: planes 32-38
        (d1, d2) if d1 == d2 && d1 < 0 => 39 + (-d1) as usize,  // Down-left: planes 40-46
        (d1, d2) if d1 == -d2 && d2 > 0 => 47 + d2 as usize,  // Up-left: planes 48-54
        (d1, d2) if d1 == -d2 && d2 < 0 => 55 + (-d2) as usize,  // Down-right: planes 56-62
        
        // Knight moves
        (2, 1) => 64,
        (1, 2) => 65,
        (-1, 2) => 66,
        (-2, 1) => 67,
        (2, -1) => 68,
        (1, -2) => 69,
        (-1, -2) => 70,
        (-2, -1) => 71,
        
        _ => panic!("Invalid move: {:?}", mv),
    };
    
    (plane_idx, from_rank_idx, from_file_idx)
}

/// Get a mask of legal moves in the 72x8x8 representation
pub fn get_legal_move_mask(board: &Board) -> Array3<i32> {
    let mut mask = Array3::<i32>::zeros((72, 8, 8));
    let movegen = MoveGen::new_legal(board);
    
    for mv in movegen {
        let (plane_idx, rank_idx, file_idx) = move_to_idx(mv);
        mask[[plane_idx, rank_idx, file_idx]] = 1;
    }
    
    mask
}

/// Mirror a move vertically (for handling black's perspective)
pub fn mirror_move(mv: ChessMove) -> ChessMove {
    let from_square = mv.get_source();
    let to_square = mv.get_dest();
    
    let from_rank_idx = from_square.get_rank().to_index();
    let to_rank_idx = to_square.get_rank().to_index();
    
    let new_from_rank = Rank::from_index(7 - from_rank_idx);
    let new_to_rank = Rank::from_index(7 - to_rank_idx);
    
    let new_from = Square::make_square(new_from_rank, from_square.get_file());
    let new_to = Square::make_square(new_to_rank, to_square.get_file());
    
    // Check if it's a promotion move
    if let Some(promotion) = mv.get_promotion() {
        ChessMove::new(new_from, new_to, Some(promotion))
    } else {
        ChessMove::new(new_from, new_to, None)
    }
}

/// Mirror a board position vertically
pub fn mirror_board(board: &Board) -> Board {
    // Get the FEN string
    let fen_str = board.to_string();
    let fen_parts: Vec<&str> = fen_str.split_whitespace().collect();
    
    let mut new_fen_parts = vec![];
    
    // Mirror the piece positions
    if !fen_parts.is_empty() {
        let board_part = fen_parts[0];
        let ranks: Vec<&str> = board_part.split('/').collect();
        let mut mirrored_ranks = Vec::new();
        
        // Reverse the ranks and flip colors
        for rank in ranks.iter().rev() {
            let mut new_rank = String::new();
            for ch in rank.chars() {
                if ch.is_ascii_digit() {
                    new_rank.push(ch);
                } else {
                    // Flip the color of the piece
                    if ch.is_ascii_uppercase() {
                        new_rank.push(ch.to_ascii_lowercase());
                    } else {
                        new_rank.push(ch.to_ascii_uppercase());
                    }
                }
            }
            mirrored_ranks.push(new_rank);
        }
        new_fen_parts.push(mirrored_ranks.join("/"));
    } else {
        new_fen_parts.push("8/8/8/8/8/8/8/8".to_string());
    }
    
    // Side to move (flipped)
    if fen_parts.len() > 1 {
        new_fen_parts.push(match fen_parts[1] {
            "w" => "b",
            "b" => "w",
            _ => "w"
        }.to_string());
    } else {
        new_fen_parts.push("w".to_string());
    }
    
    // Castling rights (flipped)
    if fen_parts.len() > 2 {
        let castling = fen_parts[2];
        let mut new_castling = String::new();
        for ch in castling.chars() {
            match ch {
                'K' => new_castling.push('k'),
                'Q' => new_castling.push('q'),
                'k' => new_castling.push('K'),
                'q' => new_castling.push('Q'),
                '-' => new_castling.push('-'),
                _ => {}
            }
        }
        if new_castling.is_empty() {
            new_castling.push('-');
        }
        new_fen_parts.push(new_castling);
    } else {
        new_fen_parts.push("-".to_string());
    }
    
    // En passant (mirrored if exists)
    if fen_parts.len() > 3 && fen_parts[3] != "-" {
        let ep = fen_parts[3];
        if ep.len() == 2 {
            let file = ep.chars().nth(0).unwrap();
            let rank = ep.chars().nth(1).unwrap();
            if let Some(rank_digit) = rank.to_digit(10) {
                let new_rank = 9 - rank_digit;
                new_fen_parts.push(format!("{}{}", file, new_rank));
            } else {
                new_fen_parts.push("-".to_string());
            }
        } else {
            new_fen_parts.push("-".to_string());
        }
    } else {
        new_fen_parts.push("-".to_string());
    }
    
    // Halfmove and fullmove clocks
    new_fen_parts.push("0".to_string());
    new_fen_parts.push("1".to_string());
    
    let fen_str = new_fen_parts.join(" ");
    Board::from_str(&fen_str).unwrap()
}

/// Encode a position for neural network inference
/// Returns (position_planes, legal_move_mask)
pub fn encode_position_for_inference(board: &Board) -> (Array3<f32>, Array3<i32>) {
    let board_to_encode = if board.side_to_move() == Color::Black {
        mirror_board(board)
    } else {
        *board
    };
    
    let position_planes = encode_position(&board_to_encode);
    let mask = get_legal_move_mask(&board_to_encode);
    
    (position_planes, mask)
}

/// Decode policy output from neural network
pub fn decode_policy_output(board: &Board, policy: &Array1<f32>) -> Result<Array1<f32>> {
    let mut move_probabilities = Array1::<f32>::zeros(200);
    let movegen = MoveGen::new_legal(board);
    let moves_vec: Vec<ChessMove> = movegen.collect();
    let num_moves = moves_vec.len();
    
    for (idx, mv) in moves_vec.iter().enumerate() {
        let mv_to_encode = if board.side_to_move() == Color::Black {
            mirror_move(*mv)
        } else {
            *mv
        };
        
        let (plane_idx, rank_idx, file_idx) = move_to_idx(mv_to_encode);
        let move_idx = plane_idx * 64 + rank_idx * 8 + file_idx;
        
        if move_idx >= policy.len() {
            return Err(anyhow!(
                "moveIdx {} is out of bounds for policy array of size {}",
                move_idx,
                policy.len()
            ));
        }
        
        move_probabilities[idx] = policy[move_idx];
    }
    
    Ok(move_probabilities.slice(s![..num_moves]).to_owned())
}

/// Call neural network on a single position
pub fn call_neural_network(
    board: &Board,
    network: &crate::network::AlphaZeroNet,
    device: Device,
) -> Result<(f32, Array1<f32>)> {
    let (position, mask) = encode_position_for_inference(board);
    
    // Convert to tensors
    let position_tensor = Tensor::from_slice(position.as_slice().unwrap())
        .reshape(&[1, 16, 8, 8])
        .to_device(device)
        .to_kind(Kind::Float);
    
    let mask_tensor = Tensor::from_slice(mask.as_slice().unwrap())
        .reshape(&[1, 72, 8, 8])
        .to_device(device)
        .to_kind(Kind::Float);
    
    // Flatten mask
    let mask_flat = mask_tensor.view([1, -1]);
    
    // Run inference
    let (value, policy) = network.forward(&position_tensor, Some(&mask_flat))?;
    
    // Extract results
    let value_scalar = value.double_value(&[0, 0]) as f32;
    let policy_vec = Vec::<f32>::try_from(policy.view([-1]).to_kind(Kind::Float))?;
    let policy_array = Array1::from_vec(policy_vec);
    
    let move_probabilities = decode_policy_output(board, &policy_array)?;
    
    Ok((value_scalar, move_probabilities))
}

/// Call neural network on a batch of positions
pub fn call_neural_network_batched(
    boards: &[Board],
    network: &crate::network::AlphaZeroNet,
    device: Device,
) -> Result<(Vec<f32>, Vec<Array1<f32>>)> {
    let num_boards = boards.len();
    
    // Prepare batch tensors
    let mut positions = Vec::new();
    let mut masks = Vec::new();
    
    for board in boards {
        let (pos, mask) = encode_position_for_inference(board);
        positions.push(pos);
        masks.push(mask);
    }
    
    // Stack into batch tensors
    let position_data: Vec<f32> = positions.iter()
        .flat_map(|p| p.as_slice().unwrap())
        .copied()
        .collect();
    let position_tensor = Tensor::from_slice(&position_data)
        .reshape(&[num_boards as i64, 16, 8, 8])
        .to_device(device)
        .to_kind(Kind::Float);
    
    let mask_data: Vec<f32> = masks.iter()
        .flat_map(|m| m.as_slice().unwrap().iter().map(|&x| x as f32))
        .collect();
    let mask_tensor = Tensor::from_slice(&mask_data)
        .reshape(&[num_boards as i64, 72, 8, 8])
        .to_device(device)
        .to_kind(Kind::Float);
    
    // Flatten masks
    let mask_flat = mask_tensor.view([num_boards as i64, -1]);
    
    // Run batch inference
    let (values, policies) = network.forward(&position_tensor, Some(&mask_flat))?;
    
    // Extract results
    let values_vec = Vec::<f32>::try_from(values.view([-1]).to_kind(Kind::Float))?;
    let policies_tensor = policies.to_kind(Kind::Float);
    
    let mut move_probabilities = Vec::new();
    for (i, board) in boards.iter().enumerate() {
        let policy_slice = policies_tensor.narrow(0, i as i64, 1).view([-1]);
        let policy_vec = Vec::<f32>::try_from(policy_slice)?;
        let policy_array = Array1::from_vec(policy_vec);
        let move_probs = decode_policy_output(board, &policy_array)?;
        move_probabilities.push(move_probs);
    }
    
    Ok((values_vec, move_probabilities))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_encode_position() {
        let board = Board::default();
        let encoded = encode_position(&board);
        
        // Check dimensions
        assert_eq!(encoded.shape(), &[16, 8, 8]);
        
        // Check initial position encoding
        // White pawns on rank 1 (index 1)
        for file in 0..8 {
            assert_eq!(encoded[[0, 1, file]], 1.0);
        }
        
        // Black pawns on rank 6 (index 6)
        for file in 0..8 {
            assert_eq!(encoded[[1, 6, file]], 1.0);
        }
    }
    
    #[test]
    fn test_move_to_idx() {
        // Test a simple pawn move e2-e4
        let mv = ChessMove::from_str("e2e4").unwrap();
        
        let (plane_idx, rank_idx, file_idx) = move_to_idx(mv);
        assert_eq!(rank_idx, 1);
        assert_eq!(file_idx, 4);
        assert_eq!(plane_idx, 17); // Vertical move up by 2
    }
}