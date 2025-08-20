#!/usr/bin/env python3
"""
Quick test for parameter extraction functionality
"""

import torch
import numpy as np
from model import GPT, GPTConfig
from web.dashboard import DashboardBroadcaster

def test_parameter_extraction():
    print('Testing parameter extraction...')
    
    # Create a small test model
    config = GPTConfig(
        block_size=64,
        vocab_size=100,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        bias=False
    )
    
    model = GPT(config)
    print(f'✅ Created test model with {model.get_num_params()} parameters')
    
    # Test parameter extraction
    broadcaster = DashboardBroadcaster(enabled=True)  # Enable for testing
    parameter_data = broadcaster.extract_parameters(model, max_samples=10)
    
    if parameter_data:
        print('✅ Parameter extraction successful')
        print(f'   Token embeddings: {len(parameter_data["token_embeddings"])}')
        print(f'   Position embeddings: {len(parameter_data["position_embeddings"])}')
        print(f'   Layer weights: {len(parameter_data["layer_weights"])}')
        print(f'   Metadata: {parameter_data["metadata"]}')
        
        # Test validation
        if broadcaster._validate_parameter_data(parameter_data):
            print('✅ Parameter data validation passed')
        else:
            print('❌ Parameter data validation failed')
            
        # Test compression
        compressed = broadcaster._compress_parameter_data(parameter_data)
        if compressed:
            print('✅ Parameter data compression successful')
            print(f'   Compressed token embeddings: {len(compressed["token_embeddings"])}')
        else:
            print('❌ Parameter data compression failed')
            
        # Test sampling strategy
        for iteration in [10, 50, 100, 1000, 5000]:
            should_extract = broadcaster.should_extract_parameters(iteration)
            print(f'   Iteration {iteration}: extract = {should_extract}')
            
        return True
    else:
        print('❌ Parameter extraction failed')
        return False

if __name__ == '__main__':
    test_parameter_extraction()