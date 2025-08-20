#!/usr/bin/env python3
"""
Dashboard simulator to test parameter visualization without training
"""

import time
import json
import numpy as np
from web.dashboard import DashboardBroadcaster

def simulate_parameter_data():
    """Generate mock parameter data for testing"""
    return {
        'token_embeddings': [
            {
                'token_id': i,
                'values': np.random.randn(32).tolist(),
                'magnitude': float(np.random.rand() * 5),
                'type': 'token_embedding'
            } for i in range(20)
        ],
        'position_embeddings': [
            {
                'position': i,
                'values': np.random.randn(32).tolist(),
                'magnitude': float(np.random.rand() * 3),
                'type': 'position_embedding'
            } for i in range(10)
        ],
        'layer_weights': [
            {
                'layer': i % 4,
                'component': ['attention', 'mlp'][i % 2],
                'index': i,
                'value': float(np.random.randn() * 0.1),
                'magnitude': float(abs(np.random.randn() * 0.1)),
                'type': 'layer_weight'
            } for i in range(30)
        ],
        'metadata': {
            'vocab_size': 1000,
            'n_embd': 32,
            'n_layer': 4,
            'block_size': 64
        }
    }

def test_dashboard_functionality():
    print('🧪 Testing Dashboard Parameter Visualization')
    print('=' * 50)
    
    # Initialize dashboard
    broadcaster = DashboardBroadcaster(port=8080, enabled=True)
    
    # Test parameter data generation
    print('✅ Generating mock parameter data...')
    parameter_data = simulate_parameter_data()
    
    # Test validation
    print('✅ Testing parameter validation...')
    is_valid = broadcaster._validate_parameter_data(parameter_data)
    print(f'   Validation result: {is_valid}')
    
    # Test compression
    print('✅ Testing parameter compression...')
    compressed = broadcaster._compress_parameter_data(parameter_data)
    original_size = len(json.dumps(parameter_data))
    compressed_size = len(json.dumps(compressed))
    print(f'   Original size: {original_size} bytes')
    print(f'   Compressed size: {compressed_size} bytes')
    print(f'   Compression ratio: {compressed_size/original_size:.2f}')
    
    # Test sampling strategy
    print('✅ Testing sampling strategy...')
    test_iterations = [10, 50, 100, 500, 1000, 5000, 10000]
    
    # Simulate connected clients
    broadcaster.websocket_clients.add("mock_client")
    
    for iteration in test_iterations:
        should_extract = broadcaster.should_extract_parameters(iteration)
        print(f'   Iteration {iteration:5d}: extract = {should_extract}')
    
    # Test broadcast (without actual WebSocket)
    print('✅ Testing parameter broadcast...')
    try:
        broadcaster.broadcast_parameters(parameter_data, 100)
        print('   Broadcast test completed (no actual clients)')
    except Exception as e:
        print(f'   Broadcast test failed: {e}')
    
    print('\n🎉 Dashboard functionality tests completed!')
    print('\nTo test the full dashboard:')
    print('1. Run: python train.py --dataset=shakespeare_char --max_iters=50 --compile=False --device=cpu')
    print('2. Open: http://localhost:8080')
    print('3. Switch to "Parameter Visualization" tab')
    
    return True

if __name__ == '__main__':
    test_dashboard_functionality()