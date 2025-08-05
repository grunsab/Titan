use anyhow::Result;
use chess::{Board, BoardStatus, ChessMove};
use std::process::{Command, Stdio};
use std::io::{Write, BufRead, BufReader};
use std::path::Path;
use std::str::FromStr;
use piebot::{
    chess_openings::{get_opening, get_total_openings},
    mcts::Root,
    network::AlphaZeroNet,
    device_utils,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MoveComparison {
    position_fen: String,
    opening_name: String,
    move_number: usize,
    python_move: String,
    rust_move: String,
    python_q_value: f32,
    rust_q_value: f32,
    python_visit_count: f32,
    rust_visit_count: f32,
    moves_match: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct ComparisonReport {
    model_path: String,
    total_positions: usize,
    matching_moves: usize,
    match_rate: f32,
    comparisons: Vec<MoveComparison>,
}

/// Run the Python engine and get its move selection
fn get_python_move(
    python_path: &str,
    model_path: &str,
    fen: &str,
    rollouts: usize,
) -> Result<(String, f32, f32)> {
    let output = Command::new("python3")
        .arg(python_path)
        .arg("--model")
        .arg(model_path)
        .arg("--mode")
        .arg("p")
        .arg("--rollouts")
        .arg(&rollouts.to_string())
        .arg("--threads")
        .arg("1")
        .arg("--fen")
        .arg(fen)
        .arg("--verbose")
        .output()?;
    
    let stdout = String::from_utf8(output.stdout)?;
    
    // Parse the output to extract move, Q value, and visit count
    let mut best_move = String::new();
    let mut q_value = 0.0;
    let mut visit_count = 0.0;
    
    for line in stdout.lines() {
        if line.starts_with("best move ") {
            best_move = line.replace("best move ", "").trim().to_string();
        } else if line.contains("total rollouts") && line.contains(" Q ") {
            // Parse line like: "total rollouts 100 Q 0.523 duplicate paths 0 elapsed 0.12 nps 833.33"
            let parts: Vec<&str> = line.split_whitespace().collect();
            for (i, part) in parts.iter().enumerate() {
                if *part == "rollouts" && i + 1 < parts.len() {
                    visit_count = parts[i + 1].parse().unwrap_or(0.0);
                } else if *part == "Q" && i + 1 < parts.len() {
                    q_value = parts[i + 1].parse().unwrap_or(0.0);
                }
            }
        }
    }
    
    Ok((best_move, q_value, visit_count))
}

/// Run the Rust engine and get its move selection
fn get_rust_move(
    board: &Board,
    network: &AlphaZeroNet,
    device: tch::Device,
    rollouts: usize,
) -> Result<(String, f32, f32)> {
    let root = Root::new(board, network, device)?;
    
    for _ in 0..rollouts {
        root.rollout(board, network, device)?;
    }
    
    let q_value = root.get_q();
    let visit_count = root.get_n();
    
    if let Some(edge) = root.max_n_select() {
        let best_move = edge.get_move().to_string();
        Ok((best_move, q_value, visit_count))
    } else {
        anyhow::bail!("No legal moves found")
    }
}

/// Compare Python and Rust engines on a set of positions
fn compare_engines_on_positions(
    model_path: &str,
    python_script: &str,
    rollouts: usize,
    num_openings: Option<usize>,
) -> Result<ComparisonReport> {
    // Load Rust model
    let (device, _) = device_utils::get_optimal_device();
    let network = AlphaZeroNet::load_from_file(Path::new(model_path), device)?;
    
    let mut comparisons = Vec::new();
    let num_openings = num_openings.unwrap_or(get_total_openings());
    
    for opening_idx in 0..num_openings {
        let opening = get_opening(opening_idx);
        println!("Testing opening {}/{}: {}", opening_idx + 1, num_openings, opening.name);
        
        let mut board = Board::default();
        
        // Play through the opening moves
        for (move_idx, move_str) in opening.moves.iter().enumerate() {
            let chess_move = ChessMove::from_str(move_str)?;
            board = board.make_move_new(chess_move);
        }
        
        // Now test several positions after the opening
        for test_move in 0..5 {
            if board.status() != BoardStatus::Ongoing {
                break;
            }
            
            let fen = format!("{}", board);
            
            // Get Python move
            println!("  Position {}: Getting Python move...", test_move + 1);
            let (python_move, python_q, python_n) = 
                get_python_move(python_script, model_path, &fen, rollouts)?;
            
            // Get Rust move
            println!("  Position {}: Getting Rust move...", test_move + 1);
            let (rust_move, rust_q, rust_n) = 
                get_rust_move(&board, &network, device, rollouts)?;
            
            let moves_match = python_move == rust_move;
            println!("    Python: {} (Q={:.3}, N={:.0})", python_move, python_q, python_n);
            println!("    Rust:   {} (Q={:.3}, N={:.0})", rust_move, rust_q, rust_n);
            println!("    Match: {}", if moves_match { "YES" } else { "NO" });
            
            comparisons.push(MoveComparison {
                position_fen: fen,
                opening_name: opening.name.clone(),
                move_number: opening.moves.len() + test_move + 1,
                python_move: python_move.clone(),
                rust_move: rust_move.clone(),
                python_q_value: python_q,
                rust_q_value: rust_q,
                python_visit_count: python_n,
                rust_visit_count: rust_n,
                moves_match,
            });
            
            // Make the Python move to continue
            if let Ok(mv) = ChessMove::from_str(&python_move) {
                board = board.make_move_new(mv);
            } else {
                break;
            }
        }
    }
    
    let total_positions = comparisons.len();
    let matching_moves = comparisons.iter().filter(|c| c.moves_match).count();
    let match_rate = matching_moves as f32 / total_positions as f32;
    
    Ok(ComparisonReport {
        model_path: model_path.to_string(),
        total_positions,
        matching_moves,
        match_rate,
        comparisons,
    })
}

#[test]
#[ignore] // Run with: cargo test --test compare_engines -- --ignored
fn test_engine_consistency() -> Result<()> {
    // Configuration
    let model_path = "weights/AlphaZeroNet_20x256.pt";
    let python_script = "../PieBot_v1/playchess.py";
    let rollouts = 100;
    let num_openings = Some(5); // Test first 5 openings
    
    println!("Comparing Python and Rust engines...");
    println!("Model: {}", model_path);
    println!("Rollouts per position: {}", rollouts);
    
    let report = compare_engines_on_positions(
        model_path, 
        python_script, 
        rollouts,
        num_openings
    )?;
    
    // Print summary
    println!("\n=== COMPARISON SUMMARY ===");
    println!("Total positions tested: {}", report.total_positions);
    println!("Matching moves: {}", report.matching_moves);
    println!("Match rate: {:.1}%", report.match_rate * 100.0);
    
    // Print detailed mismatches
    println!("\n=== MISMATCHES ===");
    for comp in &report.comparisons {
        if !comp.moves_match {
            println!("\nOpening: {} (move {})", comp.opening_name, comp.move_number);
            println!("FEN: {}", comp.position_fen);
            println!("Python: {} (Q={:.3}, N={:.0})", 
                comp.python_move, comp.python_q_value, comp.python_visit_count);
            println!("Rust:   {} (Q={:.3}, N={:.0})", 
                comp.rust_move, comp.rust_q_value, comp.rust_visit_count);
        }
    }
    
    // Save report to JSON
    let report_json = serde_json::to_string_pretty(&report)?;
    std::fs::write("engine_comparison_report.json", report_json)?;
    println!("\nDetailed report saved to: engine_comparison_report.json");
    
    // Assert high match rate
    assert!(
        report.match_rate > 0.8, 
        "Match rate too low: {:.1}%", 
        report.match_rate * 100.0
    );
    
    Ok(())
}

#[test]
fn test_single_position() -> Result<()> {
    // Test a single position for debugging
    let model_path = "weights/AlphaZeroNet_20x256.pt";
    let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    let rollouts = 10;
    
    let board = Board::default();
    let (device, _) = device_utils::get_optimal_device();
    let network = AlphaZeroNet::load_from_file(Path::new(model_path), device)?;
    
    let (rust_move, rust_q, rust_n) = get_rust_move(&board, &network, device, rollouts)?;
    
    println!("Rust engine result:");
    println!("  Move: {}", rust_move);
    println!("  Q-value: {:.3}", rust_q);
    println!("  Visit count: {:.0}", rust_n);
    
    Ok(())
}