# HPI-DHC @ BioCreative VIII–SympTEMIST

Detection and Normalization of Symptom Mentions with [SpanMarker](https://github.com/tomaarsen/SpanMarkerNER) and [xMEN](https://github.com/hpi-dhc/xmen)

## Preparation

- Get the recent version of the SympTEMIST data from Zenodo: https://zenodo.org/records/8413866
- For subtask 2 (EL), get access to the [UMLS 2023AA metathesaurus](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html)


### Subtask 1 (NER)

- Training: [notebooks/00_NER_Training.ipynb](notebooks/00_NER_Training.ipynb)
- Inference for SympTEMIST Test Set: [notebooks/01_NER_Inference.ipynb](notebooks/01_NER_Inference.ipynb)
- Result Analysis and Background Set Predictions: [notebooks/02_NER_Analysis.ipynb](notebooks/02_NER_Analysis.ipynb)

### Subtask 2 (EL)

- see [notebooks/03_EL_xMEN.ipynb](notebooks/03_EL_xMEN.ipynb)

## Citation

```bibtex
@inproceedings{borchert2023symptemist,
  author = {Borchert, Florian and Llorca, Ignacio and Schapranow, Matthieu-P.},
  title = {HPI-DHC @ BC8 SympTEMIST Track: Detection and Normalization of Symptom Mentions with SpanMarker and xMEN},
  booktitle = {Proceedings of the BioCreative VIII Challenge and Workshop: Curation and Evaluation in the Era of Generative Models},
  editor = {Islamaj, Rezarta and Arighi, Cecilia and Campbell, Ian and Gonzalez-Hernandez, Graciela and Hirschman, Lynette and Krallinger, Martin and Lima-López, Salvador and Weissenbacher, Davy and Lu, Zhiyong},
  address = {New Orleans, LA},
  year = 2023
}
```
