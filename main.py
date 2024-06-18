import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer

real_data = pd.read_csv('study_performance_real.csv')

print(real_data.to_string()) 

from sdv.metadata import SingleTableMetadata

metadata = SingleTableMetadata()

metadata.detect_from_dataframe(real_data)

print(metadata)

from sdv.single_table import GaussianCopulaSynthesizer

synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(data=real_data)

synthetic_data = synthesizer.sample(num_rows=1500)

print(synthetic_data)

from sdv.evaluation.single_table import evaluate_quality

quality_report = evaluate_quality(
    real_data,
    synthetic_data,
    metadata)

concatenated_data = pd.concat([real_data, synthetic_data], ignore_index=True, axis=0)

concatenated_data.to_csv('study_performance.csv', index=False) 