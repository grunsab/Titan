use cozy_chess::Board;

#[test]
fn nullmove_reduces_nodes_midgame() {
    use piebot::search::alphabeta::{Searcher, SearchParams};
    let fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3";
    let b = Board::from_fen(fen, false).unwrap();
    let mut s1 = Searcher::default();
    let mut p1 = SearchParams::default();
    p1.depth = 4; p1.use_tt = true; p1.order_captures = true; p1.use_history = true; p1.threads = 1;
    p1.use_nullmove = false; p1.use_aspiration = true; p1.aspiration_window_cp = 50;
    let r1 = s1.search_with_params(&b, p1);

    let mut s2 = Searcher::default();
    let mut p2 = p1; p2.use_nullmove = true;
    let r2 = s2.search_with_params(&b, p2);
    assert!((r2.score_cp - r1.score_cp).abs() <= 100, "nullmove changed score too much: {} vs {}", r2.score_cp, r1.score_cp);
    assert!(r2.nodes <= r1.nodes, "nullmove did not reduce nodes: {} vs {}", r2.nodes, r1.nodes);
}

#[test]
fn nullmove_disabled_in_check() {
    use piebot::search::alphabeta::{Searcher, SearchParams};
    // Black in check from rook on a1
    let fen = "k7/8/8/8/8/8/8/R3K3 b - - 0 1";
    let b = Board::from_fen(fen, false).unwrap();
    let mut s1 = Searcher::default();
    let mut p1 = SearchParams::default();
    p1.depth = 3; p1.use_tt = true; p1.order_captures = true; p1.use_history = true; p1.threads = 1;
    p1.use_nullmove = false;
    let r1 = s1.search_with_params(&b, p1);

    let mut s2 = Searcher::default();
    let mut p2 = p1; p2.use_nullmove = true;
    let r2 = s2.search_with_params(&b, p2);
    assert_eq!(r2.score_cp, r1.score_cp, "nullmove in check should not change score");
}
