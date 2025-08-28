use std::io::{self, BufRead};
use std::time::Duration;
use crate::board::cozy::Position;
use crate::search::alphabeta::{Searcher, SearchParams};

pub struct UciEngine {
    pos: Position,
    searcher: Searcher,
    hash_mb: usize,
    threads: usize,
}

impl UciEngine {
    pub fn new() -> Self { Self { pos: Position::startpos(), searcher: Searcher::default(), hash_mb: 64, threads: 1 } }

    fn cmd_uci(&self) {
        println!("id name PieBot NNUE (skeleton)");
        println!("id author PieBot Team");
        println!("option name Threads type spin default 1 min 1 max 512");
        println!("option name Hash type spin default 64 min 1 max 16384");
        println!("uciok");
    }

    fn cmd_isready(&self) { println!("readyok"); }

    fn cmd_ucinewgame(&mut self) { self.pos = Position::startpos(); }

    pub(crate) fn apply_setoption(&mut self, name: &str, value: &str) {
        match name.to_lowercase().as_str() {
            "hash" => {
                if let Ok(mb) = value.parse::<usize>() { self.hash_mb = mb; self.searcher.set_tt_capacity_mb(mb); }
            }
            "threads" => {
                if let Ok(t) = value.parse::<usize>() { self.threads = t.max(1); }
            }
            _ => {}
        }
    }

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

    fn cmd_setoption(&mut self, args: &str) {
        // setoption name <name> [value <value>]
        let mut tokens = args.split_whitespace();
        if tokens.next() != Some("name") { return; }
        let mut name_parts = Vec::new();
        let mut value: Option<String> = None;
        for tok in tokens {
            if tok == "value" {
                value = Some(String::new());
                continue;
            }
            if let Some(v) = value.as_mut() { if !v.is_empty() { v.push(' '); } v.push_str(tok); } else { name_parts.push(tok.to_string()); }
        }
        let name = name_parts.join(" ");
        let val = value.unwrap_or_else(|| "".to_string());
        self.apply_setoption(&name, &val);
    }

    fn cmd_go(&mut self, args: &str) {
        // Support minimal: go depth N | go movetime T
        let mut depth: u32 = 6;
        let mut movetime_ms: Option<u64> = None;
        let mut tokens = args.split_whitespace();
        while let Some(tok) = tokens.next() {
            match tok {
                "depth" => {
                    if let Some(d) = tokens.next().and_then(|s| s.parse::<u32>().ok()) { depth = d; }
                }
                "movetime" => {
                    if let Some(t) = tokens.next().and_then(|s| s.parse::<u64>().ok()) { movetime_ms = Some(t); }
                }
                _ => {}
            }
        }
        let mut params = SearchParams::default();
        params.depth = depth;
        params.use_tt = true;
        params.order_captures = true;
        params.use_history = true;
        params.movetime = movetime_ms.map(Duration::from_millis);
        params.threads = self.threads;
        let res = self.searcher.search_with_params(self.pos.board(), params);
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
            if let Some(rest) = line.strip_prefix("setoption ") { self.cmd_setoption(rest); continue; }
            if line == "quit" { break; }
            if let Some(rest) = line.strip_prefix("position ") { self.cmd_position(rest); continue; }
            if let Some(rest) = line.strip_prefix("go ") { self.cmd_go(rest); continue; }
            if line == "stop" { /* ignore in skeleton */ continue; }
        }
    }
}
