# Hybrid QkNN

This algorithm use both quantum and classical quantum to run steps like in kNN. Specially, it use a new design circuit that I called Integrated swap test circuit.

![](images/algorithm.png?raw=true)

# Integrated swap test circuit

Swap test is a procedure in quantum computing which you can see its circuit below:

![](images/swap_test_circuit.png?raw=true)

Because of simplicity, it can only calculate the fidelity between two 1 - qubit (or 2 - dimensional states). When you want to apply it in real life (4,8,16,... - dimensional state), swap test must be more complex. There are two problem that new version of swap test has:

- Load N components in the state into circuit by using n qubits with n = log2(N)
- Using n Fredkin gates and 2 Hadamard gates to perform swap test on two N - dimensional states.

So this is our new idea:

*For calculate the fidelity between 2 - dimensional states.*
![](images/integrated_swap_test/2-dimensional.png?raw=true)

*For calculate the fidelity between 4 - dimensional states.*
![](images/integrated_swap_test/4-dimensional.png?raw=true)

*For calculate the fidelity between 8 - dimensional states.*
![](images/integrated_swap_test/8-dimensional.png?raw=true)

*For calculate the fidelity between 16 - dimensional states.*
![](images/integrated_swap_test/16-dimensional.png?raw=true)

Don't scare when our circuit uses 2(N - 1) + 1 qubits, ancilla took up 2(N - 1) + 1 - 2log2(N) - 1 = 2(N - 1 - log2(N)). Details on this circuit explained in our paper which will be publish soon. A parts of it is from [this code](https://github.com/adjs/dcsp) [18]

# Experiments

### 1. On ingrated swap test circuit

![](images/se_re.png?raw=true)

Standard error and relative error when use this circuit to calculate fidelity between two states.

### 2. Hybrid QkNN on Iris dataset

Accuracy

![](images/accuracy.png?raw=true)

Confusion matrix

![](images/confusion_matrix.png?raw=true)


# References

[1] Häffner, H., Roos, C. F., & Blatt, R. (2008). Quantum computing with trapped ions. Physics reports, 469(4), 155-203.

[2] Kok, P., Munro, W. J., Nemoto, K., Ralph, T. C., Dowling, J. P., & Milburn, G. J. (2007). Linear optical quantum computing with photonic qubits. Reviews of modern physics, 79(1), 135.

[3] LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. nature, 521(7553), 436-444.

[4] Lu, S., Huang, S., Li, K., Li, J., Chen, J., Lu, D., ... & Zeng, B. (2018). Separability-entanglement classifier via machine learning. Physical Review A, 98(1), 012315.

[5] Lloyd, S., Mohseni, M., & Rebentrost, P. (2013). Quantum algorithms for super-vised and unsupervised machine learning. arXiv preprint arXiv:1307.0411.

[6] Rebentrost, P., Mohseni, M., & Lloyd, S. (2014). Quantum support vector ma-chine for big data classification. Physical review letters, 113(13), 130503.

[7] Lloyd, S., Mohseni, M., & Rebentrost, P. (2014). Quantum principal component analysis. Nature Physics, 10(9), 631-633.

[8] Ruan, Y., Xue, X., Liu, H., Tan, J., & Li, X. (2017). Quantum algorithm for k-nearest neighbors classification based on the metric of hamming distance. Interna-tional Journal of Theoretical Physics, 56(11), 3496-3507.

[9] Basheer, A., & Goyal, S. K. (2020). Quantum k-nearest neighbor machine learning algorithm. arXiv preprint arXiv:2003.09187.

[10] Fastovets, D. V., Bogdanov, Y. I., Bantysh, B. I., & Lukichev, V. F. (2019, March). Machine learning methods in quantum computing theory. In International Conference on Micro-and Nano-Electronics 2018 (Vol. 11022, p. 110222S). Interna-tional Society for Optics and Photonics.

[11] Ruan, Y., Xue, X., Liu, H., Tan, J., & Li, X. (2017). Quantum algorithm for k-nearest neighbors classification based on the metric of hamming distance. Interna-tional Journal of Theoretical Physics, 56(11), 3496-3507.

[12] Kok, D. J., Caron, S., & Acun, A. (2021). Building a quantum kNN classifier with Qiskit: theoretical gains put to practice.

[13] Araujo, I. F., Park, D. K., Petruccione, F., & da Silva, A. J. (2021). A divide-and-conquer algorithm for quantum state preparation. Scientific Reports, 11(1), 1-12.

[14] Cunningham, P., & Delany, S. J. (2020). k-Nearest Neighbour Classifiers--. arXiv preprint arXiv:2004.04523.

[15] Jozsa, R. (1994). Fidelity for mixed quantum states. Journal of modern optics, 41(12), 2315-2323.

[16] Zhang, S., Li, X., Zong, M., Zhu, X., & Cheng, D. (2017). Learning k for knn clas-sification. ACM Transactions on Intelligent Systems and Technology (TIST), 8(3), 1-19.

[17] Barenco, A., Berthiaume, A., Deutsch, D., Ekert, A., Jozsa, R., & Macchiavello, C. (1997). Stabilization of quantum computations by symmetrization. SIAM Journal on Computing, 26(5), 1541-1557.

[18] Araujo, I. F., Park, D. K., Petruccione, F., & da Silva, A. J. (2021). A divide-and-conquer algorithm for quantum state preparation. Scientific Reports, 11(1), 1-12.

[19] Park, D. K., Sinayskiy, I., Fingerhuth, M., Petruccione, F., & Rhee, J. K. K. (2019). Parallel quantum trajectories via forking for sampling without redundancy. New Journal of Physics, 21(8), 083024.

[20] Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). Introduction to Algorithms, (International Edition).

[21] Wiebe, N., Kapoor, A., & Svore, K. (2014). Quantum algorithms for nearest-neighbor methods for supervised and unsupervised learning. arXiv preprint arXiv:1401.2142.

[22] C. L. Blake and C. J. Merz. (1998). UCI Repository of machine learning data-bases [http://www.ics.uci.edu/~mlearn/MLRepository.html]. Irvine, CA: University of California, Department of Information and Computer Science

[23] Mitarai, K., Kitagawa, M., & Fujii, K. (2019). Quantum analog-digital conver-sion. Physical Review A, 99(1), 012301.


