use anyhow::Result;
use tch::{nn, nn::Module, Device, Kind, Tensor};
use std::path::Path;

/// Convolutional block with conv -> batch norm -> relu
#[derive(Debug)]
pub struct ConvBlock {
    conv: nn::Conv2D,
    bn: nn::BatchNorm,
    _relu: bool,
}

impl ConvBlock {
    pub fn new(vs: &nn::Path, input_channels: i64, num_filters: i64) -> Self {
        let conv = nn::conv2d(vs, input_channels, num_filters, 3, nn::ConvConfig { padding: 1, ..Default::default() });
        let bn = nn::batch_norm2d(vs, num_filters, Default::default());
        
        Self {
            conv,
            bn,
            _relu: true,
        }
    }
}

impl Module for ConvBlock {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.conv)
            .apply_t(&self.bn, true)
            .relu()
    }
}

/// Residual block
#[derive(Debug)]
pub struct ResidualBlock {
    conv1: nn::Conv2D,
    bn1: nn::BatchNorm,
    conv2: nn::Conv2D,
    bn2: nn::BatchNorm,
}

impl ResidualBlock {
    pub fn new(vs: &nn::Path, num_filters: i64) -> Self {
        let conv1 = nn::conv2d(vs, num_filters, num_filters, 3, nn::ConvConfig { padding: 1, ..Default::default() });
        let bn1 = nn::batch_norm2d(vs, num_filters, Default::default());
        let conv2 = nn::conv2d(vs, num_filters, num_filters, 3, nn::ConvConfig { padding: 1, ..Default::default() });
        let bn2 = nn::batch_norm2d(vs, num_filters, Default::default());
        
        Self {
            conv1,
            bn1,
            conv2,
            bn2,
        }
    }
}

impl Module for ResidualBlock {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let residual = xs.shallow_clone();
        
        let out = xs
            .apply(&self.conv1)
            .apply_t(&self.bn1, true)
            .relu()
            .apply(&self.conv2)
            .apply_t(&self.bn2, true);
        
        (out + residual).relu()
    }
}

/// Value head for predicting game outcome
#[derive(Debug)]
pub struct ValueHead {
    conv: nn::Conv2D,
    bn: nn::BatchNorm,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl ValueHead {
    pub fn new(vs: &nn::Path, input_channels: i64) -> Self {
        let conv = nn::conv2d(vs, input_channels, 1, 1, nn::ConvConfig::default());
        let bn = nn::batch_norm2d(vs, 1, Default::default());
        let fc1 = nn::linear(vs, 64, 256, Default::default());
        let fc2 = nn::linear(vs, 256, 1, Default::default());
        
        Self {
            conv,
            bn,
            fc1,
            fc2,
        }
    }
}

impl Module for ValueHead {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let batch_size = xs.size()[0];
        
        xs.apply(&self.conv)
            .apply_t(&self.bn, true)
            .relu()
            .view([batch_size, 64])
            .apply(&self.fc1)
            .relu()
            .apply(&self.fc2)
            .tanh()
    }
}

/// Policy head for predicting move probabilities
#[derive(Debug)]
pub struct PolicyHead {
    conv: nn::Conv2D,
    bn: nn::BatchNorm,
    fc: nn::Linear,
}

impl PolicyHead {
    pub fn new(vs: &nn::Path, input_channels: i64) -> Self {
        let conv = nn::conv2d(vs, input_channels, 2, 1, nn::ConvConfig::default());
        let bn = nn::batch_norm2d(vs, 2, Default::default());
        let fc = nn::linear(vs, 128, 4608, Default::default());
        
        Self {
            conv,
            bn,
            fc,
        }
    }
}

impl Module for PolicyHead {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let batch_size = xs.size()[0];
        
        xs.apply(&self.conv)
            .apply_t(&self.bn, true)
            .relu()
            .view([batch_size, 128])
            .apply(&self.fc)
    }
}

/// AlphaZero neural network
pub struct AlphaZeroNet {
    conv_block: ConvBlock,
    residual_blocks: Vec<ResidualBlock>,
    value_head: ValueHead,
    policy_head: PolicyHead,
    device: Device,
}

impl AlphaZeroNet {
    pub fn new(vs: &nn::Path, num_blocks: i64, num_filters: i64, device: Device) -> Self {
        // Initial convolutional block
        let conv_block = ConvBlock::new(&vs.sub("conv_block"), 16, num_filters);
        
        // Residual blocks
        let mut residual_blocks = Vec::new();
        for i in 0..num_blocks {
            let block = ResidualBlock::new(&vs.sub(&format!("res_block_{}", i)), num_filters);
            residual_blocks.push(block);
        }
        
        // Value and policy heads
        let value_head = ValueHead::new(&vs.sub("value_head"), num_filters);
        let policy_head = PolicyHead::new(&vs.sub("policy_head"), num_filters);
        
        Self {
            conv_block,
            residual_blocks,
            value_head,
            policy_head,
            device,
        }
    }
    
    /// Forward pass through the network
    /// Returns (value, policy) tensors
    pub fn forward(&self, input: &Tensor, policy_mask: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        // Initial conv block
        let mut x = self.conv_block.forward(input);
        
        // Residual blocks
        for block in &self.residual_blocks {
            x = block.forward(&x);
        }
        
        // Split into value and policy heads
        let value = self.value_head.forward(&x);
        let policy = self.policy_head.forward(&x);
        
        // Apply policy mask if provided (for inference)
        let policy_out = if let Some(mask) = policy_mask {
            // Apply mask and softmax
            let policy_exp = policy.exp();
            let masked_exp = &policy_exp * mask;
            let sum = masked_exp.sum_dim_intlist(&[1i64][..], true, Kind::Float);
            &masked_exp / &sum
        } else {
            // Training mode - return raw logits
            policy
        };
        
        Ok((value, policy_out))
    }
    
    /// Load model from a PyTorch .pt file
    pub fn load_from_file(path: &Path, device: Device) -> Result<Self> {
        let mut vs = nn::VarStore::new(device);
        
        // Create model with default architecture (20 blocks, 256 filters)
        let model = Self::new(&vs.root(), 20, 256, device);
        
        // Load weights
        vs.load(path)?;
        
        // Set to eval mode
        vs.freeze();
        
        Ok(model)
    }
    
    /// Save model to a file
    pub fn save(&self, vs: &nn::VarStore, path: &Path) -> Result<()> {
        vs.save(path)?;
        Ok(())
    }
}

/// Helper function to create and load a model
pub fn load_model(model_path: &Path, device: Device) -> Result<AlphaZeroNet> {
    AlphaZeroNet::load_from_file(model_path, device)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_network_forward() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);
        let net = AlphaZeroNet::new(&vs.root(), 2, 64, device);
        
        // Create dummy input
        let input = Tensor::randn(&[1, 16, 8, 8], (Kind::Float, device));
        let mask = Tensor::ones(&[1, 4608], (Kind::Float, device));
        
        // Forward pass
        let result = net.forward(&input, Some(&mask));
        assert!(result.is_ok());
        
        let (value, policy) = result.unwrap();
        assert_eq!(value.size(), vec![1, 1]);
        assert_eq!(policy.size(), vec![1, 4608]);
    }
}