import onnx

model = onnx.load("/home/hatci/openpilot/selfdrive/modeld/models/big_driving_vision.onnx")
print("Inputs:")
for i in model.graph.input:
    print(i.name)

print("\nOutputs:")
for o in model.graph.output:
    print(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])