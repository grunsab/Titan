use anyhow::Result;
use chess::{Board, ChessMove, Color, Game, GameResult, MoveGen};
use clap::Parser;
use std::str::FromStr;
use piebot::{mcts::Root, network::AlphaZeroNet, device_utils};
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about = "Play chess against the AlphaZero engine", long_about = None)]
struct Args {
    /// Path to model (.pt) file
    #[arg(long)]
    model: PathBuf,
    
    /// Operation mode: 's' for self play, 'p' for profile, 'h' for human
    #[arg(long, default_value = "h")]
    mode: String,
    
    /// Your color: 'w' for white, 'b' for black
    #[arg(long, default_value = "w")]
    color: String,
    
    /// The number of rollouts on computer's turn
    #[arg(long, default_value_t = 30)]
    rollouts: usize,
    
    /// Number of threads used per rollout
    #[arg(long, default_value_t = 60)]
    threads: usize,
    
    /// Print search statistics
    #[arg(long)]
    verbose: bool,
    
    /// Starting FEN position
    #[arg(long)]
    fen: Option<String>,
}

fn parse_color(color_str: &str) -> Result<Color> {
    match color_str.to_lowercase().as_str() {
        "w" | "white" => Ok(Color::White),
        "b" | "black" => Ok(Color::Black),
        _ => anyhow::bail!("Invalid color: use 'w' or 'b'"),
    }
}

fn print_board(board: &Board) {
    // Pretty print the position
    println!("\n{}", board);
}

fn get_human_move(board: &Board) -> Result<ChessMove> {
    let movegen = MoveGen::new_legal(board);
    let legal_moves: Vec<ChessMove> = movegen.collect();
    
    loop {
        print!("Enter your move (e.g., e2e4): ");
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        
        // Try to parse the move from UCI format
        if let Ok(mv) = ChessMove::from_str(input) {
            // Check if it's legal
            if legal_moves.contains(&mv) {
                return Ok(mv);
            } else {
                println!("Illegal move!");
            }
        } else {
            println!("Invalid move format! Use format like 'e2e4'");
        }
    }
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    
    // Parse arguments
    let mode = args.mode.chars().next().unwrap_or('h');
    let human_color = parse_color(&args.color)?;
    
    // Load model
    println!("Loading model from: {}", args.model.display());
    let (device, device_str) = device_utils::get_optimal_device();
    println!("Using device: {}", device_str);
    
    let network = AlphaZeroNet::load_from_file(&args.model, device)?;
    println!("Model loaded successfully!");
    
    // Create game
    let mut game = if let Some(fen_str) = args.fen {
        let board = Board::from_str(&fen_str)
            .map_err(|e| anyhow::anyhow!("Invalid FEN string: {:?}", e))?;
        Game::new_with_board(board)
    } else {
        Game::new()
    };
    
    // Main game loop
    loop {
        // Check if game is over
        if let Some(result) = game.result() {
            match result {
                GameResult::WhiteCheckmates => {
                    println!("\nCheckmate! White wins!");
                }
                GameResult::BlackCheckmates => {
                    println!("\nCheckmate! Black wins!");
                }
                GameResult::WhiteResigns => {
                    println!("\nWhite resigns! Black wins!");
                }
                GameResult::BlackResigns => {
                    println!("\nBlack resigns! White wins!");
                }
                GameResult::Stalemate => {
                    println!("\nGame is a stalemate!");
                }
                GameResult::DrawAccepted => {
                    println!("\nGame is a draw by agreement!");
                }
                GameResult::DrawDeclared => {
                    println!("\nGame is a draw!");
                }
            }
            break;
        }
        
        // Get current board position
        let board = game.current_position();
        
        // Print current position
        println!("\n{}'s turn", 
            if board.side_to_move() == Color::White { "White" } else { "Black" });
        print_board(&board);
        
        // Determine if it's human's turn
        let is_human_turn = mode == 'h' && board.side_to_move() == human_color;
        
        if is_human_turn {
            // Human move
            let mv = get_human_move(&board)?;
            game.make_move(mv);
        } else {
            // Computer move
            if args.verbose {
                println!("Thinking...");
            }
            
            let start_time = Instant::now();
            
            // Create MCTS root and perform rollouts
            let root = Root::new(&game, &network, device)?;
            
            for _ in 0..args.rollouts {
                root.parallel_rollouts(&game, &network, device, args.threads)?;
            }
            
            let elapsed = start_time.elapsed();
            
            // Get statistics
            let q = root.get_q();
            let n = root.get_n();
            let nps = n / elapsed.as_secs_f32();
            let same_paths = root.get_same_paths();
            
            if args.verbose {
                println!("{}", root.get_statistics_string());
                println!("Total rollouts: {}, Q: {:.3}, duplicate paths: {}, elapsed: {:.2}s, NPS: {:.2}",
                    n as i32, q, same_paths, elapsed.as_secs_f32(), nps);
            }
            
            // Select best move
            if let Some(edge) = root.max_n_select() {
                let best_move = edge.get_move();
                println!("Computer plays: {}", best_move);
                game.make_move(best_move);
            } else {
                println!("No legal moves available!");
                break;
            }
        }
        
        // In profile mode, exit after first move
        if mode == 'p' {
            break;
        }
    }
    
    Ok(())
}
