use tch::{nn, Device};
use piebot::network::AlphaZeroNet;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let mut vs = nn::VarStore::new(device);
    
    // Create model with 10 blocks, 128 filters (matching the weights)
    let model = AlphaZeroNet::new(&vs.root(), 10, 128, device);
    
    println!("Created model with {} variables", vs.variables().len());
    
    // Try to load the converted weights
    let weight_path = Path::new("weights/AlphaZeroNet_10x128_rust.pt");
    println!("Loading weights from: {:?}", weight_path);
    
    match vs.load(weight_path) {
        Ok(_) => {
            println!("Successfully loaded weights!");
            
            // Test forward pass
            let input = tch::Tensor::randn(&[1, 16, 8, 8], (tch::Kind::Float, device));
            let mask = tch::Tensor::ones(&[1, 4608], (tch::Kind::Float, device));
            
            match model.forward(&input, Some(&mask)) {
                Ok((value, policy)) => {
                    println!("Forward pass successful!");
                    println!("Value shape: {:?}", value.size());
                    println!("Policy shape: {:?}", policy.size());
                    println!("Value: {:?}", value);
                }
                Err(e) => {
                    println!("Forward pass failed: {}", e);
                }
            }
        }
        Err(e) => {
            println!("Failed to load weights: {}", e);
            println!("\nExpected variables:");
            for (name, _) in vs.variables().iter().take(10) {
                println!("  {}", name);
            }
        }
    }
    
    Ok(())
}