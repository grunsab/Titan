#[test]
fn threads_param_propagates() {
    use cozy_chess::Board;
    use piebot::search::alphabeta::{Searcher, SearchParams};
    let b = Board::default();
    let mut s = Searcher::default();
    let mut p = SearchParams::default();
    p.depth = 2;
    p.use_tt = false;
    p.threads = 4;
    let _ = s.search_with_params(&b, p);
    assert_eq!(s.get_threads(), 4, "threads param did not propagate to searcher");
}

