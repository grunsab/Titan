use cozy_chess::Board;

#[test]
fn nnue_stub_evaluate_returns_zero() {
    use piebot::eval::nnue::Nnue;
    let nn = Nnue::load("nonexistent.nnue").unwrap();
    let b = Board::default();
    assert_eq!(nn.evaluate(&b), 0);
}
