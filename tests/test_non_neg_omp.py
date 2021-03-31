from bolsonaro.models.nn_omp import NonNegativeOrthogonalMatchingPursuit
import numpy as np

def test_binary_classif_omp():
    N = 1000
    L = 100

    T = np.random.rand(N, L)
    w_star = np.zeros(L)
    w_star[:L//2] = np.abs(np.random.rand(L//2))

    T /= np.linalg.norm(T, axis=0)
    y = T @ w_star

    requested_solutions = list(range(10, L, 10))
    print()
    print(len(requested_solutions))
    print(L//2)
    nn_omp = NonNegativeOrthogonalMatchingPursuit(max_iter=L, intermediate_solutions_sizes=requested_solutions, fill_with_final_solution=False)
    nn_omp.fit(T, y)

    lst_predict = nn_omp.predict(T)
    print(len(lst_predict))

    # solutions = nn_omp(T, y, L, requested_solutions)
    #
    # for idx_sol, w in enumerate(solutions):
    #     solution = T @ w
    #     non_zero = w.astype(bool)
    #     print(requested_solutions[idx_sol], np.sum(non_zero), np.linalg.norm(solution - y)/np.linalg.norm(y))

    # assert isinstance(results, np.ndarray)
