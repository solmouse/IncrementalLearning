cat > README.md << 'EOF'
# TFLite with Flex Delegate

TensorFlow Lite with SELECT_TF_OPS for on-device training.

## Usage
```bash
kubectl run --rm -it \
  --image sgs-registry.snucse.org/ws-5y8frda38hqz1/tflite-flex:latest \
  tflite-test
```

## Build

Check Actions tab for build status.
EOF