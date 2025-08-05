use criterion::{black_box, criterion_group, criterion_main, Criterion};
use chess::Board;
use piebot::{mcts::Root, network::AlphaZeroNet, device_utils};
use std::path::Path;

fn benchmark_mcts_rollouts(c: &mut Criterion) {
    // Skip benchmark if model doesn't exist
    let model_path = Path::new("weights/AlphaZeroNet_20x256.pt");
    if !model_path.exists() {
        eprintln!("Skipping benchmark: model not found at {:?}", model_path);
        return;
    }
    
    let (device, _) = device_utils::get_optimal_device();
    let network = AlphaZeroNet::load_from_file(model_path, device)
        .expect("Failed to load model");
    
    let board = Board::default();
    
    c.bench_function("mcts_single_rollout", |b| {
        b.iter(|| {
            let root = Root::new(&board, &network, device).unwrap();
            root.rollout(black_box(&board), &network, device).unwrap();
        })
    });
    
    c.bench_function("mcts_10_rollouts", |b| {
        b.iter(|| {
            let root = Root::new(&board, &network, device).unwrap();
            for _ in 0..10 {
                root.rollout(black_box(&board), &network, device).unwrap();
            }
        })
    });
    
    c.bench_function("mcts_100_rollouts", |b| {
        b.iter(|| {
            let root = Root::new(&board, &network, device).unwrap();
            for _ in 0..100 {
                root.rollout(black_box(&board), &network, device).unwrap();
            }
        })
    });
}

fn benchmark_encoder(c: &mut Criterion) {
    use piebot::encoder;
    
    let board = Board::default();
    
    c.bench_function("encode_position", |b| {
        b.iter(|| {
            encoder::encode_position(black_box(&board))
        })
    });
    
    c.bench_function("encode_position_for_inference", |b| {
        b.iter(|| {
            encoder::encode_position_for_inference(black_box(&board))
        })
    });
    
    c.bench_function("get_legal_move_mask", |b| {
        b.iter(|| {
            encoder::get_legal_move_mask(black_box(&board))
        })
    });
}

criterion_group!(benches, benchmark_mcts_rollouts, benchmark_encoder);
criterion_main!(benches);