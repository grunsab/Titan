use tch::{nn, Device};
use piebot::network::AlphaZeroNet;

fn main() {
    // Test with 10 blocks, 128 filters
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let _model = AlphaZeroNet::new(&vs.root(), 10, 128, device);
    
    println!("=== Variables for 10x128 model ===");
    let mut vars: Vec<_> = vs.variables().into_iter().collect();
    vars.sort_by_key(|(name, _)| name.clone());
    
    for (name, tensor) in vars {
        println!("{} -> shape: {:?}", name, tensor.size());
    }
    
    // Also test with 20 blocks, 256 filters
    let vs2 = nn::VarStore::new(device);
    let _model2 = AlphaZeroNet::new(&vs2.root(), 20, 256, device);
    
    println!("\n=== Variables for 20x256 model ===");
    let mut vars2: Vec<_> = vs2.variables().into_iter().collect();
    vars2.sort_by_key(|(name, _)| name.clone());
    
    // Only show first few and last few
    for (i, (name, tensor)) in vars2.iter().enumerate() {
        if i < 10 || i >= vars2.len() - 10 {
            println!("{} -> shape: {:?}", name, tensor.size());
        } else if i == 10 {
            println!("... ({} more variables) ...", vars2.len() - 20);
        }
    }
}