use tch::{nn, Device, Tensor};
use piebot::network::AlphaZeroNet;

fn main() -> anyhow::Result<()> {
    println!("Creating and saving AlphaZero model with random weights...");
    
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    
    // Create model
    let model = AlphaZeroNet::new(&vs.root(), 20, 256, device);
    
    // Test forward pass
    let batch_size = 1;
    let input = Tensor::randn(&[batch_size, 16, 8, 8], (tch::Kind::Float, device));
    let (value, policy) = model.forward(&input, None)?;
    
    println!("Model created successfully!");
    println!("Value output shape: {:?}", value.size());
    println!("Policy output shape: {:?}", policy.size());
    
    // Save the model
    let output_path = "weights/rust_alphazero.pt";
    vs.save(output_path)?;
    println!("Model saved to: {}", output_path);
    
    // Print variable count
    let var_count = vs.variables().len();
    println!("Total variables: {}", var_count);
    
    Ok(())
}