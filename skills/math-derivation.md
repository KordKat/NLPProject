---
name: math-derivation
title: Mathematical derivations and verified calculations
description: Solve and explain mathematics step by step — algebra, calculus, linear algebra, probability, ODEs — verifying every numeric result with eval and symbolic work with sympy via run_python.
triggers: derive, derivation, prove, proof, integral, integrate, derivative, differentiate, solve, equation, limit, series, matrix, eigenvalue, probability, expectation, differential equation, ode, simplify, calculate
tools: eval, run_python
---

# Mathematical derivations

## Workflow
1. Restate the problem precisely: variables, domain, what is asked (value, closed form, proof).
2. Choose the method and say why (substitution, integration by parts, diagonalisation, …).
3. Derive step by step in LaTeX; one transformation per line using `\begin{aligned} … \end{aligned}` inside `$$…$$`.
4. **Verify** before answering:
   - numbers → `eval`, e.g. `integrate('x**2*exp(-x)', 'x', 0, 5)`, `derivative('sin(x)**2', 'x', 1.2)`;
   - symbolic results → run_python with sympy and print the result.
5. State the final answer clearly (boxed: `$$\boxed{...}$$`) with units/conditions, and mention the check you performed.

## sympy patterns (run_python)
```python
import sympy as sp
x, t = sp.symbols("x t", real=True)
print(sp.integrate(x**2 * sp.exp(-x), (x, 0, sp.oo)))            # definite integral
print(sp.simplify(sp.diff(sp.sin(x)**2, x)))                      # derivative
print(sp.series(sp.exp(x), x, 0, 6))                              # Taylor series
y = sp.Function("y")
print(sp.dsolve(sp.Eq(y(t).diff(t, 2) + y(t), 0)))                # ODE
M = sp.Matrix([[2, 1], [1, 2]]); print(M.eigenvals(), M.eigenvects())
print(sp.solve(sp.Eq(x**2 - 5*x + 6, 0), x))                      # equation
```

## Rules
- Never skip the verification step for a numeric answer; if eval and your derivation disagree, find the error.
- Keep notation consistent with the user's (and their preferred notation from preferences).
- For proofs: state assumptions, then each logical step with justification; end with ∎.
- Do not search papers for routine mathematics unless the user asks for sources.
