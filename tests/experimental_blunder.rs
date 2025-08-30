use cozy_chess::Board;

// Regression: experimental search (alphabeta_temp) should not blunder Qxh2 in this FEN.
// FEN: rnb1kbnr/4pppp/3q4/1P6/p7/P3P3/3P1PPP/RNBQKBNR b KQkq - 0 7
#[test]
fn experimental_avoids_losing_queen_sac_on_h2() {
    let fen = "rnb1kbnr/4pppp/3q4/1P6/p7/P3P3/3P1PPP/RNBQKBNR b KQkq - 0 7";
    let board = Board::from_fen(fen, false).expect("valid FEN");
    // Depth-2 should be sufficient to see Rxh2 wins the queen.
    let mut s = piebot::search::alphabeta_temp::Searcher::default();
    let res = s.search_depth(&board, 2);
    if let Some(bm) = res.bestmove {
        assert_ne!(bm.as_str(), "d6h2", "experimental chose losing queen sac Qxh2 at depth 2");
    }
}

// Sanity check: SEE detects that Qxh2 is a losing capture for Black.
#[test]
fn see_flags_qxh2_as_losing() {
    let fen = "rnb1kbnr/4pppp/3q4/1P6/p7/P3P3/3P1PPP/RNBQKBNR b KQkq - 0 7";
    let board = Board::from_fen(fen, false).expect("valid FEN");
    // Locate Qd6h2
    let mut qxh2 = None;
    board.generate_moves(|ml| {
        for m in ml { if format!("{}", m) == "d6h2" { qxh2 = Some(m); break; } }
        qxh2.is_some()
    });
    let m = qxh2.expect("Qxh2 should be legal");
    let see = piebot::search::see::see_gain_cp(&board, m).expect("SEE gain must exist");
    assert!(see < -300, "expected Qxh2 to be a large losing capture (SEE), got {}", see);
}

// Baseline regression: avoid hanging quiet move Nc3 which loses to ...Qxc3.
#[test]
fn baseline_avoids_hanging_nc3() {
    let fen = "rn2kbnr/2qb1ppp/8/1P1Pp3/p7/P4N2/4PPPP/RNBQKB1R w KQkq - 1 10";
    let board = Board::from_fen(fen, false).expect("valid FEN");
    let mut s = piebot::search::alphabeta::Searcher::default();
    let res = s.search_depth(&board, 2);
    if let Some(bm) = res.bestmove {
        assert_ne!(bm.as_str(), "b1c3", "baseline chose hanging quiet move Nc3 at depth 2");
    }
}

// Experimental regression: avoid queen sac Qxd3 in this FEN.
// FEN: rn2k1nr/3b1ppp/3b4/1P1Pp3/p3P3/P2B1N2/3Q1PPP/1qBK3R b kq - 4 14
#[test]
fn experimental_avoids_losing_qxd3() {
    let fen = "rn2k1nr/3b1ppp/3b4/1P1Pp3/p3P3/P2B1N2/3Q1PPP/1qBK3R b kq - 4 14";
    let board = Board::from_fen(fen, false).expect("valid FEN");
    let mut s = piebot::search::alphabeta_temp::Searcher::default();
    let res = s.search_depth(&board, 2);
    if let Some(bm) = res.bestmove {
        assert_ne!(bm.as_str(), "b1d3", "experimental chose losing queen capture Qxd3 at depth 2");
    }
}
