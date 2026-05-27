from src.models.gru_model import GRUModel

model_builder = GRUModel()

model = model_builder.build_model()

assert model is not None

print("Training test passed")