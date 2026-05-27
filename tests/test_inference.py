from src.pipelines.unified_prediction_pipeline import (
    UnifiedPredictionPipeline
)

pipeline = UnifiedPredictionPipeline()

result = pipeline.predict(
    "Amazing movie",
    model_type="GRU"
)

assert isinstance(result, dict)

print("Inference test passed")