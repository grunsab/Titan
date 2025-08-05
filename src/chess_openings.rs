/// Common chess openings for testing MCTS implementations.
/// Each opening consists of 4 moves (8 half-moves) in UCI notation.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChessOpening {
    pub name: String,
    pub moves: Vec<String>,
}

/// Get all chess openings
pub fn get_chess_openings() -> Vec<ChessOpening> {
    vec![
        ChessOpening {
            name: "Italian Game".to_string(),
            moves: vec!["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "e1g1", "g8f6"]
                .into_iter().map(String::from).collect(),
        },
        ChessOpening {
            name: "Ruy Lopez".to_string(),
            moves: vec!["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6"]
                .into_iter().map(String::from).collect(),
        },
        ChessOpening {
            name: "Queen's Gambit".to_string(),
            moves: vec!["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7"]
                .into_iter().map(String::from).collect(),
        },
        ChessOpening {
            name: "Sicilian Defense - Dragon Variation".to_string(),
            moves: vec!["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6"]
                .into_iter().map(String::from).collect(),
        },
        ChessOpening {
            name: "French Defense".to_string(),
            moves: vec!["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "f8b4", "e4e5", "c7c5"]
                .into_iter().map(String::from).collect(),
        },
        ChessOpening {
            name: "King's Indian Defense".to_string(),
            moves: vec!["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6"]
                .into_iter().map(String::from).collect(),
        },
        ChessOpening {
            name: "English Opening".to_string(),
            moves: vec!["c2c4", "e7e5", "b1c3", "g8f6", "g2g3", "d7d5", "c4d5", "f6d5"]
                .into_iter().map(String::from).collect(),
        },
        ChessOpening {
            name: "Caro-Kann Defense".to_string(),
            moves: vec!["e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4", "c3e4", "c8f5"]
                .into_iter().map(String::from).collect(),
        },
        ChessOpening {
            name: "Scotch Game".to_string(),
            moves: vec!["e2e4", "e7e5", "g1f3", "b8c6", "d2d4", "e5d4", "f3d4", "f8c5"]
                .into_iter().map(String::from).collect(),
        },
        ChessOpening {
            name: "Queen's Indian Defense".to_string(),
            moves: vec!["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "b7b6", "g2g3", "c8b7"]
                .into_iter().map(String::from).collect(),
        },
        ChessOpening {
            name: "Nimzo-Indian Defense".to_string(),
            moves: vec!["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "e2e3", "e8g8"]
                .into_iter().map(String::from).collect(),
        },
        ChessOpening {
            name: "Pirc Defense".to_string(),
            moves: vec!["e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "g7g6", "f2f4", "f8g7"]
                .into_iter().map(String::from).collect(),
        },
        ChessOpening {
            name: "Alekhine's Defense".to_string(),
            moves: vec!["e2e4", "g8f6", "e4e5", "f6d5", "d2d4", "d7d6", "g1f3", "c8g4"]
                .into_iter().map(String::from).collect(),
        },
        ChessOpening {
            name: "Benoni Defense".to_string(),
            moves: vec!["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "e7e6", "b1c3", "e6d5"]
                .into_iter().map(String::from).collect(),
        },
        ChessOpening {
            name: "Dutch Defense".to_string(),
            moves: vec!["d2d4", "f7f5", "g2g3", "g8f6", "f1g2", "e7e6", "g1f3", "f8e7"]
                .into_iter().map(String::from).collect(),
        },
        ChessOpening {
            name: "Scandinavian Defense".to_string(),
            moves: vec!["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5", "d2d4", "g8f6"]
                .into_iter().map(String::from).collect(),
        },
        ChessOpening {
            name: "Vienna Game".to_string(),
            moves: vec!["e2e4", "e7e5", "b1c3", "g8f6", "f2f4", "d7d5", "f4e5", "f6e4"]
                .into_iter().map(String::from).collect(),
        },
        ChessOpening {
            name: "King's Gambit".to_string(),
            moves: vec!["e2e4", "e7e5", "f2f4", "e5f4", "g1f3", "g7g5", "h2h4", "g5g4"]
                .into_iter().map(String::from).collect(),
        },
        ChessOpening {
            name: "London System".to_string(),
            moves: vec!["d2d4", "d7d5", "g1f3", "g8f6", "c1f4", "c7c5", "e2e3", "b8c6"]
                .into_iter().map(String::from).collect(),
        },
        ChessOpening {
            name: "Catalan Opening".to_string(),
            moves: vec!["d2d4", "g8f6", "c2c4", "e7e6", "g2g3", "d7d5", "f1g2", "f8e7"]
                .into_iter().map(String::from).collect(),
        },
    ]
}

/// Get an opening by index, cycling through the list if necessary
pub fn get_opening(index: usize) -> ChessOpening {
    let openings = get_chess_openings();
    openings[index % openings.len()].clone()
}

/// Get the total number of openings available
pub fn get_total_openings() -> usize {
    get_chess_openings().len()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_get_openings() {
        let openings = get_chess_openings();
        assert_eq!(openings.len(), 20);
        
        // Check first opening
        assert_eq!(openings[0].name, "Italian Game");
        assert_eq!(openings[0].moves.len(), 8);
        assert_eq!(openings[0].moves[0], "e2e4");
    }
    
    #[test]
    fn test_get_opening_by_index() {
        let opening = get_opening(0);
        assert_eq!(opening.name, "Italian Game");
        
        // Test cycling
        let opening = get_opening(20);
        assert_eq!(opening.name, "Italian Game");
        
        let opening = get_opening(21);
        assert_eq!(opening.name, "Ruy Lopez");
    }
    
    #[test]
    fn test_total_openings() {
        assert_eq!(get_total_openings(), 20);
    }
}