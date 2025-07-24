# UR3 Hybrid System - Test Suite

This directory contains all organized tests for the UR3 Hybrid Deep Learning System.

## Test Files

### Host GPU System Tests
- **`gpu_integration_test.py`** - Complete host GPU system integration test
  - Neural network functionality
  - Data pipeline processing
  - GPU server operation
  - Training pipeline components

### VM Simulation System Tests  
- **`vm_integration_test.py`** - Complete VM simulation system integration test
  - Robot controller functionality
  - Camera handler operation
  - Webots simulation bridge
  - Network communication

### End-to-End Tests
- **`integration_test.py`** - Full system integration test
  - Host-VM communication
  - Complete data pipeline
  - Neural network inference
  - Robot control verification

### Deployment Tests
- **`simple_deployment_test.py`** - Quick deployment verification
  - System configuration checks
  - Network connectivity
  - Basic functionality verification

## Running Tests

```bash
# Navigate to tests directory
cd tests

# Run individual test suites
python3 gpu_integration_test.py       # Host system
python3 vm_integration_test.py        # VM system  
python3 integration_test.py           # End-to-end
python3 simple_deployment_test.py     # Deployment

# Run all tests (if you have a test runner)
# python3 -m pytest .  # If pytest is installed
```

## Test Results

All tests should pass with:
- ✅ **GPU Integration**: Neural network and GPU components verified
- ✅ **VM Integration**: Simulation and robot control verified
- ✅ **End-to-End**: Complete pipeline communication verified
- ✅ **Deployment**: Production readiness confirmed

## Archived Tests

- `end_to_end_test.py` - Replaced by organized integration tests
- `real_deployment_test.py` - Replaced by simple_deployment_test.py
- `test_gpu_setup.py` - Replaced by gpu_integration_test.py
- `test_vm_setup.py` - Replaced by vm_integration_test.py
- `final_deployment_verification.py` - Functionality merged into other tests

## Support

For test issues or questions, see:
- `../QUICK_START.md` - System setup and testing guide
- `../DEPLOYMENT_GUIDE.md` - Detailed deployment instructions
- `../README.md` - Project overview and documentation
