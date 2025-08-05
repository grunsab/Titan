use anyhow::Result;
use tch::{Device, Cuda};
use log::info;

/// Automatically detect and return the best available device for PyTorch operations
pub fn get_optimal_device() -> (Device, String) {
    // Check for CUDA availability
    if Cuda::is_available() {
        let device = Device::Cuda(0);
        let gpu_count = Cuda::device_count();
        let gpu_name = format!("CUDA GPU (count: {})", gpu_count);
        
        info!("Using CUDA device: {}", gpu_name);
        
        // Note: tch-rs doesn't provide direct access to GPU names or memory info
        // like the Python version does
        
        (device, gpu_name)
    } else {
        // Fallback to CPU
        let device = Device::Cpu;
        let device_str = "CPU".to_string();
        
        info!("Using CPU device");
        
        (device, device_str)
    }
}

/// Get the number of available GPUs
pub fn get_gpu_count() -> i64 {
    if Cuda::is_available() {
        Cuda::device_count()
    } else {
        0
    }
}

/// Check if CUDA is available
pub fn cuda_is_available() -> bool {
    Cuda::is_available()
}

/// Get device by index (for multi-GPU setups)
pub fn get_device_by_index(index: usize) -> Result<Device> {
    if index == 0 && !Cuda::is_available() {
        Ok(Device::Cpu)
    } else if (index as i64) < get_gpu_count() {
        Ok(Device::Cuda(index))
    } else {
        anyhow::bail!("Device index {} not available", index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_get_optimal_device() {
        let (device, device_str) = get_optimal_device();
        
        // This test will pass regardless of whether CUDA is available
        match device {
            Device::Cuda(_) => assert!(device_str.contains("CUDA")),
            Device::Cpu => assert_eq!(device_str, "CPU"),
            _ => panic!("Unexpected device type"),
        }
    }
    
    #[test]
    fn test_get_gpu_count() {
        let count = get_gpu_count();
        assert!(count >= 0);
    }
}