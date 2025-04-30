# Procedural Pipeline Generation

## Generation Algorithm

1. Select N operations (where N scales with difficulty)
2. Generate parameters for each operation
4. Add path for the Intermediary output 

## Data State Validation
- After each operation, create a checkpoint file containing a hash of the data state
- Provide a validation tool that checks if the current data state matches expected checkpoints
- The agent must pass validation at each step before proceeding

## Evaluation Metrics

1. **Correctness**: Does each step produce the expected output?
2. **Robustness**: How well does the agent handle unexpected data patterns?
3. Breaking point: At what pipeline complexity does the agent fail?