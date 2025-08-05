use anyhow::Result;
use chess::{Board, ChessMove, Color, Game, GameResult, MoveGen};
use std::sync::{Arc, Mutex};
use ndarray::Array1;
use tch::Device;

use crate::{encoder, network::AlphaZeroNet};

const C_PUCT: f32 = 1.5;

/// Calculate the UCT formula for tree search
fn calc_uct(edge: &Edge, n_parent: f32) -> f32 {
    let q = edge.get_q();
    let n_child = edge.get_n();
    let p = edge.get_p();
    
    // Handle NaN values
    let p = if p.is_nan() { 1.0 / 200.0 } else { p };
    
    let uct = q + p * C_PUCT * (n_parent.sqrt()) / (1.0 + n_child);
    
    if uct.is_nan() { 0.0 } else { uct }
}

/// Thread-safe statistics for a node
#[derive(Debug)]
struct NodeStats {
    n: f32,
    sum_q: f32,
}

/// A node in the MCTS tree
pub struct Node {
    stats: Arc<Mutex<NodeStats>>,
    edges: Vec<Edge>,
}

impl Node {
    /// Create a new node
    pub fn new(game: &Game, new_q: f32, move_probabilities: Array1<f32>) -> Self {
        let stats = Arc::new(Mutex::new(NodeStats {
            n: 1.0,
            sum_q: if new_q.is_nan() { 0.5 } else { new_q },
        }));
        
        let mut edges = Vec::new();
        
        // Check if game is ongoing
        if game.result().is_none() {
            let board = game.current_position();
            let movegen = MoveGen::new_legal(&board);
            for (idx, mv) in movegen.into_iter().enumerate() {
                let prob = if idx < move_probabilities.len() {
                    move_probabilities[idx]
                } else {
                    1.0 / 200.0
                };
                edges.push(Edge::new(mv, prob));
            }
        }
        
        Self { stats, edges }
    }
    
    /// Get visit count
    pub fn get_n(&self) -> f32 {
        self.stats.lock().unwrap().n
    }
    
    /// Get average value
    pub fn get_q(&self) -> f32 {
        let stats = self.stats.lock().unwrap();
        stats.sum_q / stats.n
    }
    
    /// Select the edge that maximizes UCT
    pub fn uct_select(&self) -> Option<&Edge> {
        let n = self.get_n();
        self.edges
            .iter()
            .max_by(|a, b| {
                let uct_a = calc_uct(a, n);
                let uct_b = calc_uct(b, n);
                uct_a.partial_cmp(&uct_b).unwrap_or(std::cmp::Ordering::Equal)
            })
    }
    
    /// Select the edge with maximum visit count
    pub fn max_n_select(&self) -> Option<&Edge> {
        self.edges
            .iter()
            .max_by(|a, b| {
                let n_a = a.get_n();
                let n_b = b.get_n();
                n_a.partial_cmp(&n_b).unwrap_or(std::cmp::Ordering::Equal)
            })
    }
    
    /// Update node statistics (thread-safe)
    pub fn update_stats(&self, value: f32, from_child_perspective: bool) {
        let mut stats = self.stats.lock().unwrap();
        stats.n += 1.0;
        if from_child_perspective {
            stats.sum_q += 1.0 - value;
        } else {
            stats.sum_q += value;
        }
    }
    
    /// Check if node is terminal
    pub fn is_terminal(&self) -> bool {
        self.edges.is_empty()
    }
    
    /// Get statistics string for debugging
    pub fn get_statistics_string(&self) -> String {
        let mut s = String::from("|   move   |     P     |     N     |     Q     |    UCT    |\n");
        
        let n_parent = self.get_n();
        let mut sorted_edges: Vec<_> = self.edges.iter().collect();
        sorted_edges.sort_by(|a, b| b.get_n().partial_cmp(&a.get_n()).unwrap());
        
        for edge in sorted_edges.iter().take(10) {
            let mv = edge.get_move();
            let p = edge.get_p();
            let n = edge.get_n();
            let q = edge.get_q();
            let uct = calc_uct(edge, n_parent);
            
            s.push_str(&format!(
                "|{:^10}|{:^10.4}|{:^10.4}|{:^10.4}|{:^10.4}|\n",
                format!("{:?}", mv), p, n, q, uct
            ));
        }
        
        s
    }
}

/// Thread-safe statistics for an edge
#[derive(Debug)]
struct EdgeStats {
    virtual_losses: f32,
}

/// An edge in the MCTS tree
pub struct Edge {
    mv: ChessMove,
    p: f32,
    child: Arc<Mutex<Option<Arc<Node>>>>,
    stats: Arc<Mutex<EdgeStats>>,
}

impl Edge {
    /// Create a new edge
    pub fn new(mv: ChessMove, move_probability: f32) -> Self {
        let p = if move_probability.is_nan() || move_probability < 0.0 {
            1.0 / 200.0
        } else {
            move_probability
        };
        
        Self {
            mv,
            p,
            child: Arc::new(Mutex::new(None)),
            stats: Arc::new(Mutex::new(EdgeStats {
                virtual_losses: 0.0,
            })),
        }
    }
    
    /// Check if edge has a child
    pub fn has_child(&self) -> bool {
        self.child.lock().unwrap().is_some()
    }
    
    /// Get visit count
    pub fn get_n(&self) -> f32 {
        let virtual_losses = self.stats.lock().unwrap().virtual_losses;
        
        if let Some(ref child) = *self.child.lock().unwrap() {
            child.get_n() + virtual_losses
        } else {
            virtual_losses
        }
    }
    
    /// Get average value
    pub fn get_q(&self) -> f32 {
        if let Some(ref child) = *self.child.lock().unwrap() {
            let virtual_losses = self.stats.lock().unwrap().virtual_losses;
            let child_sum_q = child.stats.lock().unwrap().sum_q;
            let child_n = child.get_n();
            1.0 - ((child_sum_q + virtual_losses) / (child_n + virtual_losses))
        } else {
            0.0
        }
    }
    
    /// Get prior probability
    pub fn get_p(&self) -> f32 {
        self.p
    }
    
    /// Get the move
    pub fn get_move(&self) -> ChessMove {
        self.mv
    }
    
    /// Expand the edge with a new child node
    pub fn expand(&self, game: &Game, new_q: f32, move_probabilities: Array1<f32>) -> bool {
        let mut child_guard = self.child.lock().unwrap();
        if child_guard.is_none() {
            *child_guard = Some(Arc::new(Node::new(game, new_q, move_probabilities)));
            true
        } else {
            false
        }
    }
    
    /// Get the child node
    pub fn get_child(&self) -> Option<Arc<Node>> {
        self.child.lock().unwrap().clone()
    }
    
    /// Add virtual loss (for parallel search)
    pub fn add_virtual_loss(&self) {
        self.stats.lock().unwrap().virtual_losses += 1.0;
    }
    
    /// Clear virtual losses
    pub fn clear_virtual_loss(&self) {
        self.stats.lock().unwrap().virtual_losses = 0.0;
    }
}

/// Root node of the MCTS tree
pub struct Root {
    node: Arc<Node>,
    same_paths: Arc<Mutex<usize>>,
}

impl Root {
    /// Create a new root node
    pub fn new(game: &Game, network: &AlphaZeroNet, device: Device) -> Result<Self> {
        let board = game.current_position();
        let (value, move_probabilities) = encoder::call_neural_network(&board, network, device)?;
        let q = value / 2.0 + 0.5;
        
        Ok(Self {
            node: Arc::new(Node::new(game, q, move_probabilities)),
            same_paths: Arc::new(Mutex::new(0)),
        })
    }
    
    /// Perform selection phase of MCTS
    fn select_leaf(
        &self,
        game: &mut Game,
        node_path: &mut Vec<Arc<Node>>,
        edge_path: &mut Vec<Arc<Edge>>,
    ) {
        let mut current_node = self.node.clone();
        
        loop {
            node_path.push(current_node.clone());
            
            // Find the best edge to follow
            let selected_edge = {
                let edges = &current_node.edges;
                let n = current_node.get_n();
                edges
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        let uct_a = calc_uct(a, n);
                        let uct_b = calc_uct(b, n);
                        uct_a.partial_cmp(&uct_b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(idx, _)| idx)
            };
            
            if let Some(edge_idx) = selected_edge {
                let edge = &current_node.edges[edge_idx];
                edge.add_virtual_loss();
                
                // Store Arc reference to the edge
                edge_path.push(Arc::new(Edge {
                    mv: edge.mv,
                    p: edge.p,
                    child: edge.child.clone(),
                    stats: edge.stats.clone(),
                }));
                
                game.make_move(edge.get_move());
                
                if let Some(child) = edge.get_child() {
                    current_node = child;
                } else {
                    // Unexpanded node
                    break;
                }
            } else {
                // Terminal node
                break;
            }
        }
    }
    
    /// Perform a single rollout
    pub fn rollout(&self, game: &Game, network: &AlphaZeroNet, device: Device) -> Result<()> {
        let mut game_copy = game.clone();
        let mut node_path = Vec::new();
        let mut edge_path = Vec::new();
        
        self.select_leaf(&mut game_copy, &mut node_path, &mut edge_path);
        
        let new_q = if let Some(edge) = edge_path.last() {
            // Expand the leaf
            let board = game_copy.current_position();
            let (value, move_probs) = encoder::call_neural_network(&board, network, device)?;
            let q = value / 2.0 + 0.5;
            
            let expanded = edge.expand(&game_copy, q, move_probs);
            if !expanded {
                *self.same_paths.lock().unwrap() += 1;
            }
            
            1.0 - q
        } else {
            // Terminal node
            let result = game_copy.result();
            let winner = match result {
                Some(GameResult::WhiteCheckmates) => 1,
                Some(GameResult::BlackCheckmates) => -1,
                Some(GameResult::WhiteResigns) => -1,
                Some(GameResult::BlackResigns) => 1,
                _ => 0, // Draw or ongoing (shouldn't be ongoing here)
            };
            
            (winner as f32) / 2.0 + 0.5
        };
        
        // Backpropagate
        let last_node_idx = node_path.len() - 1;
        for (i, node) in node_path.iter().enumerate().rev() {
            let from_child = (last_node_idx - i) % 2 == 1;
            node.update_stats(new_q, from_child);
        }
        
        // Clear virtual losses
        for edge in edge_path {
            edge.clear_virtual_loss();
        }
        
        Ok(())
    }
    
    /// Perform parallel rollouts
    pub fn parallel_rollouts(
        &self,
        game: &Game,
        network: &AlphaZeroNet,
        device: Device,
        num_rollouts: usize,
    ) -> Result<()> {
        // For now, do rollouts sequentially
        // TODO: Implement proper parallel rollouts with batching
        for _ in 0..num_rollouts {
            self.rollout(game, network, device)?;
        }
        Ok(())
    }
    
    /// Get total visit count
    pub fn get_n(&self) -> f32 {
        self.node.get_n()
    }
    
    /// Get average value
    pub fn get_q(&self) -> f32 {
        self.node.get_q()
    }
    
    /// Get the edge with maximum visit count
    pub fn max_n_select(&self) -> Option<&Edge> {
        self.node.max_n_select()
    }
    
    /// Get statistics string
    pub fn get_statistics_string(&self) -> String {
        self.node.get_statistics_string()
    }
    
    /// Get number of duplicate paths
    pub fn get_same_paths(&self) -> usize {
        *self.same_paths.lock().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_node_creation() {
        let board = Board::default();
        let move_probs = Array1::zeros(200);
        let node = Node::new(&board, 0.5, move_probs);
        
        assert_eq!(node.get_n(), 1.0);
        assert_eq!(node.get_q(), 0.5);
        assert!(!node.is_terminal());
    }
    
    #[test]
    fn test_edge_creation() {
        let mv = ChessMove::from_uci("e2e4").unwrap();
        let edge = Edge::new(mv, 0.1);
        
        assert_eq!(edge.get_p(), 0.1);
        assert_eq!(edge.get_n(), 0.0);
        assert!(!edge.has_child());
    }
}