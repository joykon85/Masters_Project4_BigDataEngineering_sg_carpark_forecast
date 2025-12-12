from kfp import compiler
from LSTM_pipeline import pipeline 

# Compile the pipeline
compiler.Compiler().compile(
    pipeline_func=pipeline,
    package_path="pipeline.json"  # output filename
)