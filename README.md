# Hybrid Quantum k - nearest neighbors (Hybrid QkNN)

This source is used in all experiments in the paper "New approach of KNN algorithm in quantum computing based on new design of Quantum Circuits".

_Authors: Vu Tuan Hai, Phan Hoang Chuong, Pham The Bao_

Open access: https://doi.org/10.31449/inf.v46i5.3608

Email to: haivt@uit.edu.vn if you have any questions.

Please cite the below Bibitem when used

```
@article{hai_chuong_bao_2022, 
  title={New approach of KNN algorithm in quantum computing based on new design of Quantum Circuits}, 
  volume={46}, 
  DOI={10.31449/inf.v46i5.3608}, 
  number={5}, 
  journal={Informatica}, 
  author={Hai, Vu Tuan and Chuong, Phan Hoang and Bao, Pham The}, 
  year={2022}, pages={95–103}
}
```


# How to run this code?

Package: Qiskit

The function of files:

- base:

  - encoding.py: use in the integrated swap test (isc) circuit.

  - swaptest.py: create and run isc circuit.

  - knn.py: do the classical task in the origin kNN

- jupyter:

  - qknn.ipynb: call functions from knn.py and swaptest.py, load dataset Iris and excute method predict().

  - knn.ipynb: run kNN classifier that provided from sklearn.
