use tch::{nn, Device};
use piebot::network::AlphaZeroNet;

fn main() {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    
    // Create model
    let _model = AlphaZeroNet::new(&vs.root(), 20, 256, device);
    
    // Print all variable names
    println!("Model variables:");
    for (name, _tensor) in vs.variables() {
        println!("  {}", name);
    }
    
    println!("\nTotal variables: {}", vs.variables().len());
}