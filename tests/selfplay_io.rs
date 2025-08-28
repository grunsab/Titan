use piebot::selfplay::{SelfPlayParams, generate_games, write_shards, read_shard, RECORD_SIZE, SHARD_MAGIC};
use std::fs::{read_dir, remove_file, create_dir_all};

#[test]
fn write_and_read_shard() {
    let params = SelfPlayParams {
        games: 3, max_plies: 8, threads: 1, use_engine: false, depth: 2, movetime_ms: None, seed: 123,
        temperature_tau: 0.0, temp_cp_scale: 200.0, dirichlet_alpha: 0.3, dirichlet_epsilon: 0.0,
        dirichlet_plies: 0, temperature_moves: 0, openings_path: None, temperature_tau_final: 0.1
    };
    let games = generate_games(&params);
    let outdir = std::path::Path::new("target/selfplay_test");
    create_dir_all(outdir).unwrap();
    let shards = write_shards(&games, outdir, 10).unwrap();
    assert!(!shards.is_empty());
    let recs = read_shard(&shards[0]).unwrap();
    assert!(!recs.is_empty());
}
