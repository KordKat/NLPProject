"""eval — restricted mathematical expression evaluator (AST whitelist, no Python execution)."""
from __future__ import annotations

import ast
import math
import operator
from typing import Any, Callable

from app.tools.base import ToolContext, ToolResult, ToolSpec
from app.tools.registry import compact_json, registry

MAX_POW = 10_000
MAX_FACT = 2_000
MAX_SEQ = 10_000


class MathError(ValueError):
    pass


def _safe_pow(a, b):
    if isinstance(b, (int, float)) and abs(b) > MAX_POW:
        raise MathError("exponent too large")
    if isinstance(a, int) and isinstance(b, int) and b > 0 and abs(a) > 1 and b * math.log2(abs(a)) > 100_000:
        raise MathError("result too large")
    return operator.pow(a, b)


def _safe_factorial(n):
    if not float(n).is_integer() or n < 0 or n > MAX_FACT:
        raise MathError(f"factorial argument must be an integer in [0, {MAX_FACT}]")
    return math.factorial(int(n))


BIN_OPS: dict[type, Callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: _safe_pow,
}
UNARY_OPS: dict[type, Callable] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
CMP_OPS: dict[type, Callable] = {
    ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne,
}

FUNCTIONS: dict[str, Callable] = {
    name: getattr(math, name)
    for name in (
        "sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
        "exp", "log", "log10", "log2", "log1p", "expm1", "floor", "ceil", "gcd", "lcm", "hypot", "degrees", "radians",
        "comb", "perm", "erf", "erfc", "gamma", "lgamma", "fabs", "isqrt", "trunc", "copysign", "fmod", "dist", "prod",
    )
    if hasattr(math, name)
}
FUNCTIONS.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": _safe_pow, "factorial": _safe_factorial, "ln": math.log})
CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf}


def _eval(node: ast.AST, env: dict[str, float]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval(node.body, env)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)) and not isinstance(node.value, bool):
            return node.value
        raise MathError(f"constant {node.value!r} not allowed")
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        if node.id in CONSTANTS:
            return CONSTANTS[node.id]
        raise MathError(f"unknown name '{node.id}'")
    if isinstance(node, ast.BinOp) and type(node.op) in BIN_OPS:
        return BIN_OPS[type(node.op)](_eval(node.left, env), _eval(node.right, env))
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPS:
        return UNARY_OPS[type(node.op)](_eval(node.operand, env))
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in CMP_OPS:
        return CMP_OPS[type(node.ops[0])](_eval(node.left, env), _eval(node.comparators[0], env))
    if isinstance(node, (ast.List, ast.Tuple)):
        if len(node.elts) > MAX_SEQ:
            raise MathError("sequence too long")
        return [_eval(e, env) for e in node.elts]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        fname = node.func.id
        if fname in ("integrate", "derivative"):
            return _calculus(fname, node.args, env)
        if fname not in FUNCTIONS:
            raise MathError(f"function '{fname}' not allowed")
        args = [_eval(a, env) for a in node.args]
        return FUNCTIONS[fname](*args)
    raise MathError(f"unsupported syntax: {type(node).__name__}")


def _compile_fn(expr: str, var: str) -> Callable[[float], float]:
    tree = ast.parse(expr.replace("^", "**"), mode="eval")
    return lambda x: float(_eval(tree, {var: x}))


def _str_arg(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return node.id
    raise MathError("expression and variable must be given as strings, e.g. integrate('x**2', 'x', 0, 3)")


def _calculus(fname: str, args: list[ast.AST], env: dict[str, float]) -> float:
    if fname == "integrate":
        if len(args) != 4:
            raise MathError("usage: integrate('expr', 'x', a, b)")
        expr, var = _str_arg(args[0]), _str_arg(args[1])
        a, b = _eval(args[2], env), _eval(args[3], env)
        f = _compile_fn(str(expr), str(var))
        n = 2000  # composite Simpson's rule (even n)
        a, b = float(a), float(b)
        h = (b - a) / n
        s = f(a) + f(b)
        for i in range(1, n):
            s += (4 if i % 2 else 2) * f(a + i * h)
        return s * h / 3
    if len(args) != 3:
        raise MathError("usage: derivative('expr', 'x', x0)")
    expr, var = _str_arg(args[0]), _str_arg(args[1])
    x0 = float(_eval(args[2], env))
    f = _compile_fn(expr, var)
    h = 1e-5 * max(1.0, abs(float(x0)))
    return (f(x0 - 2 * h) - 8 * f(x0 - h) + 8 * f(x0 + h) - f(x0 + 2 * h)) / (12 * h)


def safe_eval(expression: str) -> Any:
    expr = expression.strip().replace("^", "**").replace("×", "*").replace("÷", "/")
    if len(expr) > 2000:
        raise MathError("expression too long")
    tree = ast.parse(expr, mode="eval")
    result = _eval(tree, {})
    if isinstance(result, float):
        if result.is_integer() and abs(result) < 1e15:
            return result
        return float(f"{result:.12g}")
    return result


async def eval_tool(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    expr = str(args["expression"])
    try:
        result = safe_eval(expr)
    except (MathError, SyntaxError, ValueError, ZeroDivisionError, OverflowError, TypeError) as exc:
        return ToolResult.failure(f"Could not evaluate '{expr}': {exc}")
    if isinstance(result, complex):
        result = str(result)
    return ToolResult(content=compact_json({"expression": expr, "result": result}), data={"expression": expr, "result": result})


registry.register(
    ToolSpec(
        name="eval",
        description=(
            "Safely evaluate a mathematical expression. Supports + - * / // % ** (or ^), sqrt, sin, cos, tan, log (natural), "
            "log10, log2, exp, factorial, abs, round, min, max, sum, comb, gamma, pi, e, plus numeric calculus helpers "
            "integrate('expr','x',a,b) and derivative('expr','x',x0)."
        ),
        parameters={"properties": {"expression": {"type": "string", "description": "e.g. sqrt(2) * sin(pi / 4)"}}, "required": ["expression"]},
        handler=eval_tool,
        category="compute",
        timeout=15,
        when_to_use="Any arithmetic, numeric check, unit conversion arithmetic, definite integral or derivative value.",
        when_not_to_use="Symbolic algebra, plotting, data processing or multi-step programs (use run_python).",
        returns='JSON {"expression", "result"} or {"error"}.',
        example_args={"expression": "integrate('x**2', 'x', 0, 3)"},
        example_result='{"expression": "integrate(\'x**2\', \'x\', 0, 3)", "result": 9.0}',
        interpretation="Report the numeric result and explain the derivation analytically (e.g. ∫₀³ x² dx = x³/3 |₀³ = 9).",
    )
)
