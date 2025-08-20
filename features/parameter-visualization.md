# Model Parameter Visualization Integration

## Overview

Create a bbycroft-style 3D parameter visualizer that shows live evolution of nanoGPT model weights during training, integrated with the existing web dashboard. This feature will provide real-time insights into how model parameters change during training, making the learning process more transparent and educational.

## Motivation

- **Educational Value**: Visualize how neural network parameters evolve during training
- **Training Insights**: Identify convergence patterns, dead neurons, and optimization dynamics
- **Debugging Aid**: Spot parameter issues like exploding/vanishing gradients visually
- **Research Tool**: Compare parameter evolution across different hyperparameter settings
- **Inspired by bbycroft.net/llm**: Leverage proven 3D visualization techniques for transformer models

## Phase 1: Parameter Extraction & Data Pipeline

### 1.1 Training Loop Integration

**Location**: Modify `train.py` to add parameter extraction hooks

**Key Integration Points**:
- After each training step (configurable frequency)
- During validation phases
- At checkpoint saves
- On training completion

**Parameters to Extract**:
```python
# Token and position embeddings (most visually impactful)
- transformer.wte.weight  # Token embeddings [vocab_size, n_embd]
- transformer.wpe.weight  # Position embeddings [block_size, n_embd]

# Per-layer parameters (focus on first/middle/last layers)
- transformer.h[i].ln_1.weight    # Pre-attention layer norm
- transformer.h[i].attn.c_attn.weight  # Attention projection
- transformer.h[i].attn.c_proj.weight  # Attention output projection
- transformer.h[i].ln_2.weight    # Pre-MLP layer norm
- transformer.h[i].mlp.c_fc.weight     # MLP first linear layer
- transformer.h[i].mlp.c_proj.weight   # MLP second linear layer

# Final layer
- transformer.ln_f.weight  # Final layer norm
- lm_head.weight          # Language modeling head
```

### 1.2 Parameter Sampling Strategy

**Adaptive Sampling Rate**:
- Iterations 0-100: Every iteration (rapid initial changes)
- Iterations 100-1000: Every 10 iterations
- Iterations 1000+: Every 100 iterations (or configurable)

**Layer Prioritization**:
1. **High Priority**: Token embeddings (most interpretable)
2. **Medium Priority**: First and last transformer layers
3. **Low Priority**: Middle transformer layers (sample subset)

**Dimension Reduction Techniques**:
```python
class ParameterSampler:
    def sample_weight_matrix(self, weight, max_samples=1000):
        """Sample representative elements from large weight matrices"""
        if weight.numel() <= max_samples:
            return weight.detach().cpu().numpy()
        
        # Use stratified sampling to preserve distribution
        return self._stratified_sample(weight, max_samples)
    
    def compute_statistics(self, weight):
        """Compute summary statistics for visualization"""
        return {
            'mean': weight.mean().item(),
            'std': weight.std().item(),
            'min': weight.min().item(),
            'max': weight.max().item(),
            'l2_norm': weight.norm().item(),
            'sparsity': (weight.abs() < 1e-6).float().mean().item()
        }
```

### 1.3 Data Processing Pipeline

**File**: `web/parameter_extractor.py`

```python
class ParameterVisualizer:
    def __init__(self, dashboard_broadcaster, sample_rate='adaptive'):
        self.broadcaster = dashboard_broadcaster
        self.sample_rate = sample_rate
        self.iteration_count = 0
        self.last_extraction = 0
        
    def should_extract(self, iteration):
        """Determine if parameters should be extracted this iteration"""
        if self.sample_rate == 'adaptive':
            if iteration < 100:
                return True
            elif iteration < 1000:
                return iteration % 10 == 0
            else:
                return iteration % 100 == 0
        return iteration % self.sample_rate == 0
    
    def extract_parameters(self, model, iteration, optimizer=None):
        """Extract and process model parameters for visualization"""
        if not self.should_extract(iteration):
            return
            
        param_data = {
            'iteration': iteration,
            'timestamp': time.time(),
            'embeddings': self._extract_embeddings(model),
            'layers': self._extract_layer_params(model),
            'statistics': self._compute_model_stats(model)
        }
        
        if optimizer:
            param_data['gradients'] = self._extract_gradient_info(model, optimizer)
        
        # Compress and transmit
        compressed_data = self._compress_data(param_data)
        self.broadcaster.broadcast_parameters(compressed_data)
    
    def _extract_embeddings(self, model):
        """Extract token and position embeddings"""
        wte = model.transformer.wte.weight.detach().cpu()
        wpe = model.transformer.wpe.weight.detach().cpu()
        
        return {
            'token_embeddings': {
                'data': self.sample_weight_matrix(wte, max_samples=5000),
                'shape': list(wte.shape),
                'stats': self.compute_statistics(wte)
            },
            'position_embeddings': {
                'data': self.sample_weight_matrix(wpe, max_samples=1000),
                'shape': list(wpe.shape), 
                'stats': self.compute_statistics(wpe)
            }
        }
    
    def _compress_data(self, param_data):
        """Compress parameter data for efficient transmission"""
        # Use techniques like:
        # - Float32 -> Float16 conversion
        # - Delta compression (send only changes)
        # - Statistical binning for large matrices
        # - JSON compression
        pass
```

## Phase 2: 3D Visualization Engine

### 2.1 Three.js Integration

**Files**:
- `web/static/js/parameter-viz.js` - Main visualization logic
- `web/static/js/parameter-shaders.js` - Custom WebGL shaders
- `web/static/css/parameter-viz.css` - Styling

**Setup**:
```html
<!-- Add to dashboard.html -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r150/three.min.js"></script>
<script src="js/parameter-viz.js"></script>
```

### 2.2 Parameter Representation Strategies

**1. Token Embeddings as Point Cloud**:
```javascript
class TokenEmbeddingVisualizer {
    constructor(container) {
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, width/height, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer();
        this.pointCloud = null;
    }
    
    updateEmbeddings(embeddingData) {
        // Create point cloud where each token is a 3D point
        // Color represents embedding magnitude
        // Position in 3D space represents embedding values (using PCA/t-SNE)
        const positions = new Float32Array(embeddingData.length * 3);
        const colors = new Float32Array(embeddingData.length * 3);
        
        embeddingData.forEach((embedding, i) => {
            // Use first 3 dimensions or PCA projection
            positions[i*3] = embedding[0];
            positions[i*3+1] = embedding[1]; 
            positions[i*3+2] = embedding[2];
            
            // Color based on magnitude
            const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val*val, 0));
            const color = this.magnitudeToColor(magnitude);
            colors[i*3] = color.r;
            colors[i*3+1] = color.g;
            colors[i*3+2] = color.b;
        });
        
        this.updatePointCloud(positions, colors);
    }
}
```

**2. Weight Matrices as 3D Heatmaps**:
```javascript
class WeightHeatmapVisualizer {
    createHeatmap(weightMatrix, layerIndex) {
        const geometry = new THREE.PlaneGeometry(
            weightMatrix.shape[0] / 10, 
            weightMatrix.shape[1] / 10
        );
        
        // Create texture from weight values
        const texture = this.createWeightTexture(weightMatrix.data);
        const material = new THREE.MeshBasicMaterial({
            map: texture,
            transparent: true
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.z = layerIndex * 2; // Stack layers in 3D
        return mesh;
    }
    
    createWeightTexture(weights) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        // Convert weight values to color pixels
        const imageData = ctx.createImageData(width, height);
        weights.forEach((weight, i) => {
            const color = this.weightToColor(weight);
            imageData.data[i*4] = color.r;
            imageData.data[i*4+1] = color.g;
            imageData.data[i*4+2] = color.b;
            imageData.data[i*4+3] = 255;
        });
        
        ctx.putImageData(imageData, 0, 0);
        return new THREE.CanvasTexture(canvas);
    }
}
```

**3. Parameter Evolution Animation**:
```javascript
class ParameterEvolutionAnimator {
    constructor() {
        this.parameterHistory = [];
        this.currentFrame = 0;
        this.isPlaying = false;
    }
    
    addParameterSnapshot(paramData) {
        this.parameterHistory.push({
            iteration: paramData.iteration,
            embeddings: paramData.embeddings,
            layers: paramData.layers,
            timestamp: paramData.timestamp
        });
        
        // Limit history size for memory management
        if (this.parameterHistory.length > 1000) {
            this.parameterHistory.shift();
        }
    }
    
    playEvolution(speed = 1.0) {
        this.isPlaying = true;
        const animate = () => {
            if (!this.isPlaying) return;
            
            this.currentFrame = (this.currentFrame + speed) % this.parameterHistory.length;
            const frameData = this.parameterHistory[Math.floor(this.currentFrame)];
            
            this.updateVisualization(frameData);
            requestAnimationFrame(animate);
        };
        animate();
    }
}
```

### 2.3 Interactive Features

**Camera Controls**:
- Orbit controls for 3D navigation
- Zoom to focus on specific layers
- Reset view button

**Parameter Filtering**:
```javascript
class ParameterFilter {
    constructor() {
        this.visibleLayers = new Set(Array.from({length: 12}, (_, i) => i));
        this.visibleTypes = new Set(['embeddings', 'attention', 'mlp']);
        this.magnitudeThreshold = 0.0;
    }
    
    toggleLayer(layerIndex) {
        if (this.visibleLayers.has(layerIndex)) {
            this.visibleLayers.delete(layerIndex);
        } else {
            this.visibleLayers.add(layerIndex);
        }
        this.updateVisibility();
    }
    
    setMagnitudeThreshold(threshold) {
        this.magnitudeThreshold = threshold;
        this.updateVisibility();
    }
}
```

## Phase 3: Dashboard Integration

### 3.1 WebSocket Extensions

**Extend `web/dashboard.py`**:
```python
def broadcast_parameters(self, param_data):
    """Broadcast parameter updates to connected clients"""
    try:
        message = {
            "type": "parameter_update",
            "data": {
                "iteration": param_data['iteration'],
                "timestamp": param_data['timestamp'],
                "embeddings": param_data['embeddings'],
                "layer_stats": param_data['statistics'],
                "gradient_info": param_data.get('gradients', {})
            }
        }
        
        self._broadcast_to_clients(message)
        
    except Exception as e:
        print(f"Warning: Failed to broadcast parameter data: {e}")

def handle_parameter_request(self, layer_index=None, param_type=None):
    """Handle client requests for specific parameter data"""
    # Allow clients to request detailed data for specific layers
    pass
```

### 3.2 UI Layout Enhancement

**Dashboard Layout Update**:
```html
<div class="content">
    <!-- Existing metrics sidebar -->
    <section class="metrics-section" id="metrics">
        <!-- Keep existing training metrics -->
    </section>
    
    <!-- Main visualization area with tabs -->
    <section class="visualization-section">
        <div class="viz-tabs">
            <button id="loss-tab" class="tab-button active">Training Loss</button>
            <button id="params-tab" class="tab-button">Model Parameters</button>
        </div>
        
        <div id="loss-panel" class="viz-panel active">
            <!-- Existing loss chart -->
        </div>
        
        <div id="params-panel" class="viz-panel">
            <!-- Parameter visualization controls -->
            <div class="param-controls">
                <div class="layer-selector">
                    <label>Layers:</label>
                    <!-- Layer checkboxes -->
                </div>
                <div class="param-type-selector">
                    <label>Parameters:</label>
                    <!-- Parameter type checkboxes -->
                </div>
                <div class="evolution-controls">
                    <button id="play-evolution">▶️ Play Evolution</button>
                    <input type="range" id="evolution-slider" min="0" max="100">
                </div>
            </div>
            
            <!-- 3D visualization container -->
            <div id="parameter-viz-container"></div>
        </div>
    </section>
</div>
```

### 3.3 Control Panel Features

**Parameter Type Selection**:
- ☑️ Token Embeddings
- ☑️ Position Embeddings  
- ☑️ Attention Weights
- ☑️ MLP Weights
- ☑️ Layer Norms

**Layer Selection**:
- Individual layer toggles (0-11 for 12-layer model)
- "Select All" / "Deselect All" buttons
- Layer grouping (Early/Middle/Late layers)

**Evolution Controls**:
- Play/Pause parameter evolution
- Speed control slider
- Jump to specific iteration
- Export parameter snapshots

## Phase 4: Performance Optimization

### 4.1 Data Efficiency

**Smart Delta Compression**:
```python
class ParameterDeltaCompressor:
    def __init__(self):
        self.last_params = {}
        self.compression_threshold = 1e-6
    
    def compress_parameters(self, new_params, iteration):
        """Only send parameters that changed significantly"""
        if iteration == 0:
            self.last_params = new_params
            return new_params
        
        delta_params = {}
        for layer_name, params in new_params.items():
            if layer_name in self.last_params:
                delta = params - self.last_params[layer_name]
                if np.abs(delta).max() > self.compression_threshold:
                    delta_params[layer_name] = {
                        'delta': delta,
                        'compression': 'delta'
                    }
            else:
                delta_params[layer_name] = {
                    'full': params,
                    'compression': 'full'
                }
        
        self.last_params = new_params
        return delta_params
```

**Binary Data Transmission**:
```python
import msgpack

def serialize_parameters(param_data):
    """Use efficient binary serialization"""
    # Convert numpy arrays to binary
    binary_data = {}
    for key, value in param_data.items():
        if isinstance(value, np.ndarray):
            binary_data[key] = {
                'data': value.tobytes(),
                'dtype': str(value.dtype),
                'shape': value.shape
            }
        else:
            binary_data[key] = value
    
    return msgpack.packb(binary_data)
```

### 4.2 Rendering Performance

**Level of Detail (LOD)**:
```javascript
class ParameterLODManager {
    constructor() {
        this.lodLevels = {
            high: { maxPoints: 10000, updateRate: 30 },
            medium: { maxPoints: 5000, updateRate: 15 },
            low: { maxPoints: 1000, updateRate: 5 }
        };
        this.currentLOD = 'high';
    }
    
    updateLOD(cameraDistance, parameterCount) {
        let targetLOD = 'high';
        
        if (cameraDistance > 50 || parameterCount > 50000) {
            targetLOD = 'medium';
        }
        if (cameraDistance > 100 || parameterCount > 100000) {
            targetLOD = 'low';
        }
        
        if (targetLOD !== this.currentLOD) {
            this.currentLOD = targetLOD;
            this.applyLOD();
        }
    }
}
```

**WebGL Optimization**:
```javascript
class OptimizedParameterRenderer {
    constructor() {
        // Use instanced rendering for repeated elements
        this.instancedMesh = new THREE.InstancedMesh(
            geometry, 
            material, 
            maxInstances
        );
        
        // Reuse geometries and materials
        this.geometryCache = new Map();
        this.materialCache = new Map();
    }
    
    updateInstances(parameterData) {
        // Update instance matrices in batches
        const matrix = new THREE.Matrix4();
        parameterData.forEach((param, i) => {
            matrix.setPosition(param.x, param.y, param.z);
            matrix.scale(param.scale, param.scale, param.scale);
            this.instancedMesh.setMatrixAt(i, matrix);
        });
        
        this.instancedMesh.instanceMatrix.needsUpdate = true;
    }
}
```

## Phase 5: Educational Features

### 5.1 Guided Parameter Tours

**Parameter Explanation System**:
```javascript
class ParameterEducator {
    constructor() {
        this.explanations = {
            'token_embeddings': {
                title: 'Token Embeddings',
                description: 'Convert tokens into dense vectors. Each row represents a token from the vocabulary.',
                insights: [
                    'Similar tokens should have similar embeddings',
                    'Embeddings evolve to capture semantic relationships',
                    'Initial random values become meaningful representations'
                ]
            },
            'attention_weights': {
                title: 'Attention Weights',
                description: 'Determine which tokens to focus on when processing each position.',
                insights: [
                    'Learn to attend to relevant context',
                    'Different heads specialize in different relationships',
                    'Patterns emerge like attending to previous tokens'
                ]
            }
        };
    }
    
    showParameterExplanation(paramType, element) {
        const explanation = this.explanations[paramType];
        this.showTooltip(element, explanation);
    }
}
```

### 5.2 Training Phase Recognition

**Automatic Phase Detection**:
```python
class TrainingPhaseDetector:
    def __init__(self):
        self.phases = {
            'initialization': (0, 100),
            'rapid_learning': (100, 1000),
            'fine_tuning': (1000, 5000),
            'convergence': (5000, float('inf'))
        }
        
    def detect_phase(self, iteration, loss_history):
        """Detect current training phase based on iteration and loss"""
        # Basic iteration-based detection
        for phase, (start, end) in self.phases.items():
            if start <= iteration < end:
                base_phase = phase
                break
        
        # Refine based on loss dynamics
        if len(loss_history) > 50:
            recent_slope = self._compute_loss_slope(loss_history[-50:])
            if abs(recent_slope) < 1e-5:
                return 'convergence'
            elif recent_slope < -1e-3:
                return 'rapid_learning'
        
        return base_phase
    
    def get_phase_insights(self, phase):
        insights = {
            'initialization': 'Parameters are adjusting from random initialization',
            'rapid_learning': 'Model is learning fundamental patterns rapidly',
            'fine_tuning': 'Model is refining learned representations',
            'convergence': 'Parameters are stabilizing around optimal values'
        }
        return insights.get(phase, '')
```

## Technical Architecture

### File Structure
```
web/
├── static/
│   ├── dashboard.html (extended)
│   ├── js/
│   │   ├── parameter-viz.js (new - main visualization)
│   │   ├── parameter-shaders.js (new - WebGL shaders)
│   │   ├── parameter-controls.js (new - UI controls)
│   │   ├── parameter-animator.js (new - evolution animation)
│   │   └── three.min.js (new - 3D library)
│   └── css/
│       └── parameter-viz.css (new - visualization styling)
├── dashboard.py (extended with parameter broadcasting)
├── parameter_extractor.py (new - parameter extraction logic)
└── parameter_compressor.py (new - data compression utilities)

# Modified training files
train.py (add parameter extraction hooks)
```

### Data Flow Architecture
```
Training Loop (train.py)
    ↓ (parameter extraction hooks)
Parameter Extractor (parameter_extractor.py)
    ↓ (compress & format)
Dashboard Broadcaster (dashboard.py)
    ↓ (WebSocket)
Browser Client (dashboard.html)
    ↓ (JavaScript)
Parameter Visualizer (parameter-viz.js)
    ↓ (Three.js)
3D Visualization (WebGL)
```

### WebSocket Message Format
```json
{
    "type": "parameter_update",
    "data": {
        "iteration": 1500,
        "timestamp": 1673123456.789,
        "training_phase": "rapid_learning",
        "embeddings": {
            "token_embeddings": {
                "sampled_data": [...],
                "shape": [50257, 768],
                "statistics": {
                    "mean": 0.001,
                    "std": 0.12,
                    "l2_norm": 45.2
                }
            }
        },
        "layers": {
            "0": {
                "attention": {...},
                "mlp": {...},
                "layer_norm": {...}
            }
        },
        "gradient_info": {
            "global_grad_norm": 0.5,
            "layer_grad_norms": [...]
        }
    }
}
```

## Success Metrics

### Performance Targets
- **Latency**: Parameter updates transmitted within 100ms
- **Memory**: Browser memory usage < 1GB for 1000 parameter snapshots
- **Frame Rate**: 3D visualization maintains 30fps during updates
- **Scalability**: Handle GPT-2 XL (1.5B parameters) with selective visualization

### Educational Value
- **Clarity**: Users can identify training phases visually
- **Insights**: Reveal parameter evolution patterns not visible in loss curves
- **Interactivity**: Users can explore specific layers and parameter types
- **Understanding**: Improve comprehension of transformer training dynamics

### User Experience
- **Intuitive**: Controls are self-explanatory
- **Responsive**: UI remains interactive during heavy computation
- **Informative**: Tooltips and explanations provide context
- **Customizable**: Users can configure visualization preferences

## Implementation Phases

### Phase 1: Foundation (Week 1-2)
- ✅ Parameter extraction from training loop
- ✅ Basic data compression and transmission
- ✅ WebSocket protocol extensions
- ✅ Three.js integration setup

### Phase 2: Core Visualization (Week 3-4)
- ✅ Token embedding point cloud visualization
- ✅ Weight matrix heatmap rendering
- ✅ Basic parameter evolution animation
- ✅ Camera controls and navigation

### Phase 3: Advanced Features (Week 5-6)
- ✅ Interactive parameter filtering
- ✅ Training phase detection and visualization
- ✅ Performance optimization (LOD, caching)
- ✅ Educational tooltips and explanations

### Phase 4: Polish & Testing (Week 7-8)
- ✅ Comprehensive testing with different model sizes
- ✅ Performance tuning and memory optimization
- ✅ Documentation and user guides
- ✅ Integration with existing dashboard features

## Future Enhancements

### Advanced Visualizations
- **Gradient Flow**: Visualize backpropagation through the network
- **Attention Patterns**: Show attention matrices as connection graphs
- **Parameter Clusters**: Group similar parameters using clustering
- **Cross-Layer Correlations**: Show relationships between layers

### Machine Learning Insights
- **Parameter Importance**: Highlight most important parameters for specific tasks
- **Convergence Analysis**: Predict convergence based on parameter dynamics
- **Hyperparameter Effects**: Compare parameter evolution across different settings
- **Transfer Learning**: Visualize parameter changes during fine-tuning

### Collaborative Features
- **Parameter Snapshots**: Save and share interesting parameter states
- **Collaborative Annotations**: Allow team members to annotate parameter behaviors
- **Training Comparisons**: Compare parameter evolution across multiple training runs
- **Integration with wandb**: Sync with existing experiment tracking tools

This comprehensive plan provides a roadmap for creating an educational and insightful parameter visualization system that will significantly enhance the nanoGPT training experience.