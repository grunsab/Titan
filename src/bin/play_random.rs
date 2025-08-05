use chess::{Board, ChessMove, Color, Game, GameResult, MoveGen};
use std::io::{self, Write};
use rand::seq::SliceRandom;
use std::str::FromStr;

fn main() -> anyhow::Result<()> {
    println!("Chess Engine (Random Play Mode)");
    println!("===============================");
    println!("This demonstrates the chess engine working without neural network weights.");
    println!("The engine will make random legal moves.");
    println!();
    println!("Enter moves in UCI format (e.g., e2e4)");
    println!("Type 'quit' to exit");
    println!();

    let mut game = Game::new();
    let mut move_count = 0;

    loop {
        // Get current board
        let board = game.current_position();
        
        // Display the board
        println!("\n{}", board);
        
        // Check game result
        if let Some(result) = game.result() {
            match result {
                GameResult::WhiteCheckmates => println!("White wins by checkmate!"),
                GameResult::BlackCheckmates => println!("Black wins by checkmate!"),
                GameResult::WhiteResigns => println!("White resigns! Black wins!"),
                GameResult::BlackResigns => println!("Black resigns! White wins!"),
                GameResult::Stalemate => println!("It's a stalemate!"),
                GameResult::DrawAccepted => println!("Draw by agreement!"),
                GameResult::DrawDeclared => println!("It's a draw!"),
            }
            break;
        }

        // Get legal moves
        let movegen = MoveGen::new_legal(&board);
        let legal_moves: Vec<ChessMove> = movegen.collect();
        
        if legal_moves.is_empty() {
            println!("No legal moves available!");
            break;
        }

        // Player move (white)
        if move_count % 2 == 0 {
            print!("Your move: ");
            io::stdout().flush()?;
            
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            let input = input.trim();
            
            if input == "quit" {
                println!("Thanks for playing!");
                break;
            }
            
            match ChessMove::from_str(input) {
                Ok(mv) => {
                    if legal_moves.contains(&mv) {
                        game.make_move(mv);
                        move_count += 1;
                    } else {
                        println!("Illegal move! Legal moves are:");
                        for (i, mv) in legal_moves.iter().enumerate() {
                            print!("{} ", mv);
                            if (i + 1) % 10 == 0 {
                                println!();
                            }
                        }
                        println!();
                    }
                }
                Err(_) => {
                    println!("Invalid move format! Use UCI format like 'e2e4'");
                }
            }
        } else {
            // Engine move (black) - random
            let chosen_move = legal_moves.choose(&mut rand::thread_rng()).unwrap();
            println!("\nEngine plays: {}", chosen_move);
            game.make_move(*chosen_move);
            move_count += 1;
        }
    }

    Ok(())
}