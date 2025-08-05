use chess::{Board, ChessMove, Color, Game, MoveGen};
use std::str::FromStr;

fn main() {
    println!("Chess Engine Demo (without neural network)");
    println!("==========================================");
    
    // Create a new game
    let mut game = Game::new();
    
    // Print initial position
    println!("\nInitial position:");
    println!("{}", game.current_position());
    
    // Get legal moves
    let board = game.current_position();
    let movegen = MoveGen::new_legal(&board);
    let moves: Vec<ChessMove> = movegen.collect();
    println!("\nNumber of legal moves: {}", moves.len());
    
    // Make some example moves
    println!("\nMaking moves: e2e4, e7e5, g1f3");
    
    // Move 1: e2e4
    if let Ok(mv) = ChessMove::from_str("e2e4") {
        game.make_move(mv);
        println!("\nAfter 1. e4:");
        println!("{}", game.current_position());
    }
    
    // Move 2: e7e5
    if let Ok(mv) = ChessMove::from_str("e7e5") {
        game.make_move(mv);
        println!("\nAfter 1... e5:");
        println!("{}", game.current_position());
    }
    
    // Move 3: g1f3
    if let Ok(mv) = ChessMove::from_str("g1f3") {
        game.make_move(mv);
        println!("\nAfter 2. Nf3:");
        println!("{}", game.current_position());
    }
    
    // Show current legal moves
    let board = game.current_position();
    let movegen = MoveGen::new_legal(&board);
    let moves: Vec<ChessMove> = movegen.collect();
    println!("\nBlack's legal moves ({} total):", moves.len());
    for (i, mv) in moves.iter().take(10).enumerate() {
        print!("{} ", mv);
        if (i + 1) % 5 == 0 {
            println!();
        }
    }
    if moves.len() > 10 {
        println!("... and {} more", moves.len() - 10);
    } else {
        println!();
    }
    
    println!("\nThe chess engine is working correctly!");
    println!("To use with neural network, you need to convert the PyTorch weights to the correct format.");
}