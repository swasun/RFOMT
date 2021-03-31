from copy import deepcopy

from scipy.optimize import nnls
import numpy as np
from sklearn.linear_model.base import _preprocess_data

from bolsonaro import LOG_PATH

from bolsonaro.error_handling.logger_factory import LoggerFactory


class NonNegativeOrthogonalMatchingPursuit:
    """
    Input needs to be normalized

    """
    def __init__(self, max_iter, intermediate_solutions_sizes, fill_with_final_solution=True):
        assert all(type(elm) == int for elm in intermediate_solutions_sizes), "All intermediate solution must be size specified as integers."

        self.max_iter = max_iter
        self.requested_intermediate_solutions_sizes = intermediate_solutions_sizes
        self.fill_with_final_solution = fill_with_final_solution
        self._logger = LoggerFactory.create(LOG_PATH, __name__)
        self.lst_intermediate_solutions = list()
        self.lst_intercept = list()

    def _set_intercept(self, X_offset, y_offset, X_scale):
        """Set the intercept_
        """
        for sol in self.lst_intermediate_solutions:
            sol /= X_scale
            intercept = y_offset - np.dot(X_offset, sol.T)
            self.lst_intercept.append(intercept)
        # self.coef_ = self.coef_ / X_scale
        # self.intercept_ = y_offset - np.dot(X_offset, self.coef_.T)

    def fit(self, T, y):
        """
        Ref: Sparse Non-Negative Solution of a Linear System of Equations is Unique

        T: (N x L)
        y: (N x 1)
        max_iter: the max number of iteration. If requested_intermediate_solutions_sizes is None. Return the max_iter-sparse solution.
        requested_intermediate_solutions_sizes: a list of the other returned intermediate solutions than with max_iter (they are returned in a list with same indexes)

        Return the list of intermediate solutions. If the perfect solution is found before the end, the list may not be full.
        """
        # this is copied from sklearn preprocessing hope this works fine but I am a believer
        T, y, T_offset, y_offset, T_scale = _preprocess_data( T, y, fit_intercept=True, normalize=False, copy=False, return_mean=True, check_input=True)

        iter_intermediate_solutions_sizes = iter(self.requested_intermediate_solutions_sizes)

        lst_intermediate_solutions = []
        bool_arr_selected_indexes = np.zeros(T.shape[1], dtype=bool)
        residual = y
        i = 0
        next_solution = next(iter_intermediate_solutions_sizes, None)
        while i < self.max_iter and next_solution != None and not np.isclose(np.linalg.norm(residual), 0):
            # if logger is not None: logger.debug("iter {}".format(i))
            # compute all correlations between atoms and residual
            dot_products = T.T @ residual

            idx_max_dot_product = np.argmax(dot_products)
            # only positively correlated results can be taken
            if dot_products[idx_max_dot_product] <= 0:
                self._logger.warning("No other atoms is positively correlated with the residual. End prematurely with {} atoms.".format(i + 1))
                break

            # selection of atom with max correlation with residual
            bool_arr_selected_indexes[idx_max_dot_product] = True

            tmp_T = T[:, bool_arr_selected_indexes]
            sol = nnls(tmp_T, y)[0]  # non negative least square
            residual = y - tmp_T @ sol
            int_used_atoms = np.sum(sol.astype(bool))
            if  int_used_atoms != i+1:
                self._logger.warning("Atom found but not used. {} < {}".format(int_used_atoms, i+1))

            if i + 1 == next_solution:
                final_vec = np.zeros(T.shape[1])
                final_vec[bool_arr_selected_indexes] = sol  # solution is full of zero but on selected indices
                lst_intermediate_solutions.append(final_vec)
                next_solution = next(iter_intermediate_solutions_sizes, None)

            i += 1

        if len(lst_intermediate_solutions) == 0 and np.isclose(np.linalg.norm(residual), 0):
            final_vec = np.zeros(T.shape[1])
            final_vec[bool_arr_selected_indexes] = sol  # solution is full of zero but on selected indices
            lst_intermediate_solutions.append(final_vec)

        nb_missing_solutions = len(self.requested_intermediate_solutions_sizes) - len(lst_intermediate_solutions)

        if nb_missing_solutions > 0:
            if self.fill_with_final_solution:
                self._logger.warning("nn_omp ended prematurely and found less solution than expected: "
                               "expected {}. found {}".format(len(self.requested_intermediate_solutions_sizes), len(lst_intermediate_solutions)))
                lst_intermediate_solutions.extend([deepcopy(lst_intermediate_solutions[-1]) for _ in range(nb_missing_solutions)])
            else:
                self._logger.warning("nn_omp ended prematurely and found less solution than expected: "
                                     "expected {}. found {}. But fill with the last solution".format(len(self.requested_intermediate_solutions_sizes), len(lst_intermediate_solutions)))

        self.lst_intermediate_solutions = lst_intermediate_solutions
        self._set_intercept(T_offset, y_offset, T_scale)

    def predict(self, X, forest_size=None):
        if forest_size is not None:
            idx_prediction = self.requested_intermediate_solutions_sizes.index(forest_size)
            return X @ self.lst_intermediate_solutions[idx_prediction] + self.lst_intercept[idx_prediction]
        else:
            predictions = []
            for idx_sol, sol in enumerate(self.lst_intermediate_solutions):
                predictions.append(X @ sol + self.lst_intercept[idx_sol])
            return predictions

    def get_coef(self, forest_size=None):
        """
        return the intermediate solution corresponding to requested forest size if not None.

        Else return the list of intermediate solution.
        :param forest_size:
        :return:
        """
        if forest_size is not None:
            idx_prediction = self.requested_intermediate_solutions_sizes.index(forest_size)
            return self.lst_intermediate_solutions[idx_prediction]
        else:
            return self.lst_intermediate_solutions