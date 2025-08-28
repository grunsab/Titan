use std::io::{self, BufRead, Write};
use std::time::Duration;
use crate::board::cozy::Position;
use crate::search::alphabeta::Searcher;

pub struct UciEngine {
    pos: Position,
    searcher: Searcher,
}

impl UciEngine {
    pub fn new() -> Self { Self { pos: Position::startpos(), searcher: Searcher::default() } }

    fn cmd_uci(&self) {
        println!("id name PieBot NNUE (skeleton)");
        println!("id author PieBot Team");
        println!("option name Threads type spin default 1 min 1 max 512");
        println!("option name Hash type spin default 64 min 1 max 16384");
        println!("uciok");
    }

    fn cmd_isready(&self) { println!("readyok"); }

    fn cmd_ucinewgame(&mut self) { self.pos = Position::startpos(); }

    fn cmd_position(&mut self, args: &str) {
        // Supports: 'position startpos [moves ...]' and 'position fen <fen> [moves ...]'
        let mut tokens = args.split_whitespace();
        match tokens.next() {
            Some("startpos") => {
                self.pos = Position::startpos();
                if let Some("moves") = tokens.next() {
                    let moves: Vec<String> = tokens.map(|s| s.to_string()).collect();
                    if let Ok(p) = Position::set_from_start_and_moves(&moves) {
                        self.pos = p;
                    }
                }
            }
            Some("fen") => {
                // FEN is 6 fields; collect them
                let fen_fields: Vec<&str> = tokens.by_ref().take(6).collect();
                if fen_fields.len() == 6 {
                    let fen = fen_fields.join(" ");
                    if let Ok(p) = Position::from_fen(&fen) { self.pos = p; }
                }
                if let Some("moves") = tokens.next() {
                    let moves: Vec<String> = tokens.map(|s| s.to_string()).collect();
                    if let Ok(p) = Position::set_from_start_and_moves(&moves) {
                        self.pos = p;
                    }
                }
            }
            _ => {}
        }
    }

    fn cmd_go(&mut self, args: &str) {
        // Support minimal: go depth N | go movetime T (fallback to depth)
        let mut depth: u32 = 3;
        let mut tokens = args.split_whitespace();
        while let Some(tok) = tokens.next() {
            match tok {
                "depth" => {
                    if let Some(d) = tokens.next().and_then(|s| s.parse::<u32>().ok()) { depth = d; }
                }
                _ => {}
            }
        }
        let res = self.searcher.search_depth(self.pos.board(), depth);
        if let Some(best) = res.bestmove { println!("bestmove {}", best); } else { println!("bestmove 0000"); }
    }

    pub fn run_loop(&mut self) {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            let line = match line { Ok(s) => s.trim().to_string(), Err(_) => break };
            if line.is_empty() { continue; }
            if line == "uci" { self.cmd_uci(); continue; }
            if line == "isready" { self.cmd_isready(); continue; }
            if line == "ucinewgame" { self.cmd_ucinewgame(); continue; }
            if line == "quit" { break; }
            if let Some(rest) = line.strip_prefix("position ") { self.cmd_position(rest); continue; }
            if let Some(rest) = line.strip_prefix("go ") { self.cmd_go(rest); continue; }
            if line == "stop" { /* ignore in skeleton */ continue; }
        }
    }
}
