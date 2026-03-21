# Robot Evaluation System - Enterprise Edition

A modular, enterprise-grade robot evaluation system with clean architecture and separation of concerns.

## Architecture Overview

The system is organized into the following modules:

### Core Modules

1. **config_manager.py** - Configuration management
   - `RobotEvaluationConfig`: Main configuration container
   - `SmoothingConfig`: Action smoothing parameters
   - `JointSmoothingConfig`: Per-joint smoothing parameters
   - `SpeedLimitConfig`: Speed limiting parameters
   - `PolicyConfig`: Policy service configuration
   - `EvaluationConfig`: Evaluation execution configuration

2. **robot_controller.py** - Robot control abstraction
   - `RobotController`: Manages robot connection and control
   - Provides synchronous and asynchronous observation retrieval
   - Handles action sending to the robot

3. **observation_prefetcher.py** - Asynchronous observation prefetching
   - `ObservationPrefetcher`: Background prefetching of observations
   - Reduces latency by pre-fetching next observation while processing current action

4. **policy_service.py** - Policy inference service
   - `PolicyService`: Handles communication with policy server
   - Manages observation formatting and action parsing
   - Supports both sync and async operation modes

5. **action_processor.py** - Action processing pipeline
   - `ActionSmoother`: Multiple smoothing algorithms (EMA, Moving Average, SavGol, DCT)
   - `ActionInterpolator`: Smooth interpolation between actions
   - `SpeedLimiter`: Enforces speed limits on robot movements
   - `ActionProcessor`: Orchestrates the complete action processing pipeline

6. **performance_monitor.py** - Performance monitoring and metrics
   - `PerformanceMonitor`: Tracks execution metrics
   - `PerformanceMetrics`: Data class for performance statistics
   - Provides real-time performance feedback

7. **evaluation_engine.py** - Main evaluation orchestration
   - `EvaluationEngine`: Coordinates all components
   - Manages initialization, execution, and shutdown
   - Supports both sync and async evaluation modes

8. **utils.py** - Utility functions
   - Logging helpers
   - Text formatting utilities
   - Audio feedback functions

## Usage

### Basic Usage

```python
import asyncio
from core import (
    RobotEvaluationConfig,
    EvaluationEngine,
    setup_logging
)
from lerobot.robots import RobotConfig

async def main():
    setup_logging()
    
    config = RobotEvaluationConfig(
        robot_config=RobotConfig(
            type="so101_follower",
            port="/dev/ttyACM0",
            id="lil_guy"
        ),
        # ... other configurations
    )
    
    engine = EvaluationEngine(config)
    
    try:
        await engine.initialize()
        await engine.run_async_evaluation()
    finally:
        await engine.shutdown()

asyncio.run(main())
```

### Command Line Usage

```bash
python eval_lerobot_refactored.py \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=lil_guy \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 9, width: 640, height: 480, fps: 30}}" \
    --policy_host=localhost \
    --lang_instruction="Grab pens and place into pen holder."
```

## Key Features

### 1. Modular Design
- Clear separation of concerns
- Each module has a single responsibility
- Easy to test and maintain

### 2. Performance Optimization
- Asynchronous observation prefetching
- Efficient action processing pipeline
- Real-time performance monitoring

### 3. Flexibility
- Multiple smoothing algorithms (EMA, Moving Average, SavGol, DCT)
- Configurable per-joint smoothing parameters
- Adjustable speed limits
- Support for both sync and async modes

### 4. Enterprise-Grade Quality
- Comprehensive error handling
- Detailed logging
- Configuration validation
- Clean shutdown procedures

## Configuration

### Smoothing Methods

- **EMA (Exponential Moving Average)**: Fast response with some smoothing
- **Moving Average**: Simple and effective smoothing
- **SavGol (Savitzky-Golay)**: Preserves signal features while smoothing
- **DCT (Discrete Cosine Transform)**: Advanced frequency-domain smoothing

### Key Parameters

- `smoothing_method`: Algorithm to use for action smoothing
- `smoothing_window_size`: History window for smoothing
- `dct_keep_ratio`: Low-frequency coefficient retention for DCT (0.1-0.9)
- `enable_interpolation`: Enable action interpolation
- `interpolation_steps`: Number of interpolation steps between actions
- `max_delta_pos`: Maximum joint angle change per control period
- `ctrl_period`: Control period in seconds

## File Structure

```
examples/SO-101/
├── core/
│   ├── __init__.py
│   ├── config_manager.py
│   ├── robot_controller.py
│   ├── observation_prefetcher.py
│   ├── policy_service.py
│   ├── action_processor.py
│   ├── performance_monitor.py
│   ├── evaluation_engine.py
│   └── utils.py
├── eval_lerobot.py                    # Original file
├── eval_lerobot_refactored.py         # Refactored main file
└── example_usage.py                   # Usage examples
```

## Benefits of Refactoring

1. **Maintainability**: Each module is focused and easy to understand
2. **Testability**: Components can be tested independently
3. **Reusability**: Modules can be used in other projects
4. **Scalability**: Easy to add new features or modify existing ones
5. **Performance**: Optimized async operations and efficient resource usage
6. **Reliability**: Better error handling and resource management

## Migration Guide

To migrate from the original `eval_lerobot.py` to the refactored version:

1. Import from the new `core` module instead of using inline classes
2. Use `RobotEvaluationConfig` for configuration
3. Use `EvaluationEngine` for running evaluations
4. See `example_usage.py` for detailed examples

## Contributing

When adding new features:

1. Add new modules to the `core` directory
2. Update `core/__init__.py` to export new classes
3. Maintain consistent coding style
4. Add appropriate error handling
5. Update this README with new features

## License

Apache-2.0
