import numpy as np
from src.simplex import simplex

def test_ppl_geleia():
    # Coeficientes da função objetivo
    c = np.array([-5, -7])

    # Restrições lineares
    A = np.array([[3, 0], [0, 1.5], [0.25, 0.5]])
    b = np.array([250, 100, 50])

    # Execução
    solution, dual_prices, feasibility, optimal_profit = simplex(c, A, b)

    # Resultados esperados
    solution_expected = np.array([12.5, 58.33333333, 83.33333333])
    dual_prices_expected = np.array([0., 0., 0.5, 0., 14.])
    feasibility_expected = True
    optimal_profit_expected = 825.0

    # Asserções
    assert(np.testing.assert_allclose(solution, solution_expected)) == None
    assert(np.testing.assert_allclose(dual_prices, dual_prices_expected)) == None
    assert(feasibility) == feasibility_expected
    assert(optimal_profit) == optimal_profit_expected

def test_ppl_metalurgica():
    # Coeficientes da função objetivo
    c = np.array([-3000, -5000])

    # Restrições lineares
    A = np.array([[0.5, 0.2], [0.25, 0.3], [0.25, 0.5]])
    b = np.array([16, 11, 15])

    # Execução
    solution, dual_prices, feasibility, optimal_profit = simplex(c, A, b)

    # Resultados esperados
    solution_expected = np.array([2., 20., 20.])
    dual_prices_expected = np.array([0., 0., 0., 5000., 7000.])
    feasibility_expected = True
    optimal_profit_expected = 160000.0

    # Asserções
    assert(np.testing.assert_allclose(solution, solution_expected)) == None
    assert(np.testing.assert_allclose(dual_prices, dual_prices_expected)) == None
    assert(feasibility) == feasibility_expected
    assert(optimal_profit) == optimal_profit_expected

def test_ppl_petroleo():
    # Coeficientes da função objetivo
    c = np.array([-20, -15])

    # Restrições lineares
    A = np.array([[0.3, 0.4], [0.4, 0.2], [0.2, 0.3], [1, 0], [0, 1]])
    b = np.array([2000, 1500, 500, 9000, 6000])

    # Execução
    solution, dual_prices, feasibility, optimal_profit = simplex(c, A, b)

    # Resultados esperados
    solution_expected = np.array([12.5, 58.33333333, 83.33333333])
    dual_prices_expected = np.array([0., 0., 0.5, 0., 14.])
    feasibility_expected = True
    optimal_profit_expected = 825.0

    # Asserções
    assert(np.testing.assert_allclose(solution, solution_expected)) == None
    assert(np.testing.assert_allclose(dual_prices, dual_prices_expected)) == None
    assert(feasibility) == feasibility_expected
    assert(optimal_profit) == optimal_profit_expected

    

    