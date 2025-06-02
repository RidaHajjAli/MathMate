import sympy
from sympy.parsing.mathematica import parse_mathematica  # Alternative parser
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import re
import logging

ALLOWED_SYMBOLS = {
    'pi': sympy.pi,
    'π': sympy.pi,
    'e': sympy.E,
    'E': sympy.E,
    'i': sympy.I,
    'I': sympy.I,
    '∞': sympy.oo,
    'inf': sympy.oo,
    'infinity': sympy.oo,
    # Greek letters
    'α': sympy.Symbol('alpha'),
    'β': sympy.Symbol('beta'),
    'γ': sympy.Symbol('gamma'),
    'δ': sympy.Symbol('delta'),
    'θ': sympy.Symbol('theta'),
    'λ': sympy.Symbol('lambda'),
    'μ': sympy.Symbol('mu'),
    'σ': sympy.Symbol('sigma'),
    'φ': sympy.Symbol('phi'),
    'ω': sympy.Symbol('omega'),
}

ALLOWED_FUNCTIONS = {
    # Basic arithmetic
    'abs': sympy.Abs,
    'sign': sympy.sign,
    'floor': sympy.floor,
    'ceil': sympy.ceiling,
    'round': lambda x: sympy.Integer(round(x)) if x.is_number else sympy.floor(x + sympy.S.Half),
    
    # Exponential and logarithms
    'exp': sympy.exp,
    'log': sympy.log,
    'ln': sympy.ln,
    'log10': lambda x: sympy.log(x, 10),
    'log2': lambda x: sympy.log(x, 2),
    
    # Trigonometric
    'sin': sympy.sin,
    'cos': sympy.cos,
    'tan': sympy.tan,
    'cot': sympy.cot,
    'sec': sympy.sec,
    'csc': sympy.csc,
    'asin': sympy.asin,
    'acos': sympy.acos,
    'atan': sympy.atan,
    'atan2': sympy.atan2,
    
    # Hyperbolic
    'sinh': sympy.sinh,
    'cosh': sympy.cosh,
    'tanh': sympy.tanh,
    'asinh': sympy.asinh,
    'acosh': sympy.acosh,
    'atanh': sympy.atanh,
    
    # Calculus
    'diff': sympy.Derivative,
    'integrate': sympy.Integral,
    'limit': sympy.Limit,
    'sum': sympy.Sum,
    'product': sympy.Product,
    
    # Special functions
    'factorial': sympy.factorial,
    'gamma': sympy.gamma,
    'erf': sympy.erf,
    'zeta': sympy.zeta,
    'beta': sympy.beta,
    
    # Complex numbers
    're': sympy.re,
    'im': sympy.im,
    'conjugate': sympy.conjugate,
    'arg': sympy.arg,
    
    # Polynomials
    'expand': sympy.expand,
    'factor': sympy.factor,
    'simplify': sympy.simplify,
    'solve': lambda eq, var: sympy.solve(eq, var),
    
    # Statistics
    'mean': lambda args: sympy.Add(*args)/len(args),
    'stddev': lambda args: sympy.sqrt(
        sympy.Add(*[(x - (sympy.Add(*args)/len(args)))**2 for x in args]) / len(args)
    ),
    'variance': lambda args: sympy.Add(*(x**2 for x in args))/len(args) - (sympy.Add(*args)/len(args))**2,
    
    # Matrix operations
    'det': sympy.det,
    #'inv': sympy.inv,
    'transpose': sympy.transpose,
    
    # Boolean
    'and': lambda x, y: x & y,
    'or': lambda x, y: x | y,
    'not': lambda x: ~x,
    'xor': lambda x, y: x ^ y,
}

# Combine allowed symbols and functions for the parser's local dictionary
SAFE_DICT = {**ALLOWED_SYMBOLS, **ALLOWED_FUNCTIONS}

# Regex to check for potentially unsafe attributes or functions
UNSAFE_PATTERN = re.compile(r"(__|import|exec|eval|open|lambda)")

def contains_invalid_chars(expression: str) -> bool:
    """Checks for characters not typically part of a safe math expression, allowing Unicode math symbols."""
    allowed_chars_pattern = re.compile(
        r"^[a-zA-Z0-9\s\.\+\-\*\/\(\)\^\%\,\_\=\<\>\!\&\|\{\}\[\]π∞αβγδθλμσφω]+$"
    )
    if not allowed_chars_pattern.fullmatch(expression):
        return True
    return False

def calculate_sympy(expression: str) -> str:
    """
    Safely evaluates a mathematical expression string using SymPy.
    Returns the result as a string, or an error message prefixed with "Error:".
    """
    stripped_expression = expression.strip()

    if not stripped_expression:
        return "Error: Expression cannot be empty."

    if UNSAFE_PATTERN.search(stripped_expression.lower()):
        return "Error: Expression contains potentially unsafe constructs."

    if contains_invalid_chars(stripped_expression):
        return "Error: Expression contains invalid characters."

    try:
        # Preprocessing for common notations
        preprocessed = stripped_expression

        # Regex-based replacements for notations
        # 1. Replace 5! with factorial(5)
        preprocessed = re.sub(r'(\d+)!', r'factorial(\1)', preprocessed)
        # 2. Replace n° with n*pi/180 (degrees to radians)
        preprocessed = re.sub(r'(\d+)°', r'(\1*pi/180)', preprocessed)
        # 3. Replace π with pi, ∞ with oo
        preprocessed = preprocessed.replace('π', 'pi').replace('∞', 'oo')
        # 4. Replace √x with sqrt(x)
        preprocessed = re.sub(r'√\s*\(?([^\)]+)\)?', r'sqrt(\1)', preprocessed)
        # 5. Replace ∑, ∫, ∏, ∂ with corresponding function names
        preprocessed = preprocessed.replace('∑', 'Sum').replace('∫', 'Integral').replace('∏', 'Product').replace('∂', 'diff')
        # 6. Replace ÷ and ×
        preprocessed = preprocessed.replace('÷', '/').replace('×', '*')
        # 7. Replace ^ with **
        preprocessed = preprocessed.replace('^', '**')
        # 8. Replace mod with %
        preprocessed = preprocessed.replace('mod', '%')
        # 9. Replace ||x|| with Abs(x)
        preprocessed = re.sub(r'\|\|([^\|]+)\|\|', r'Abs(\1)', preprocessed)

        # Handle percentage expressions
        if '%' in preprocessed:
            parts = preprocessed.split('%')
            if len(parts) == 2 and parts[1] == '':
                preprocessed = f"{parts[0]}/100"
            else:
                # Handle expressions like "50% of 200"
                preprocessed = re.sub(r'(\d+)\s*%\s*of\s*(\d+)', r'(\1/100)*\2', preprocessed)

        # Handle square brackets for function arguments (Mathematica style)
        preprocessed = preprocessed.replace('[', '(').replace(']', ')')

        # Handle special functions like max, min
        if 'max(' in preprocessed or 'min(' in preprocessed:
            preprocessed = preprocessed.replace('max', 'Max').replace('min', 'Min')
            SAFE_DICT['Max'] = sympy.Max
            SAFE_DICT['Min'] = sympy.Min

        # Use SymPy's implicit multiplication (no manual regex for 2x, sin2x, etc.)
        transformations = standard_transformations + (implicit_multiplication_application,)

        # Parse the expression safely using a limited scope
        parsed_expr = parse_expr(preprocessed, local_dict=SAFE_DICT, transformations=transformations, global_dict={})

        # For expressions with variables, try to solve if possible
        if parsed_expr.free_symbols:
            try:
                # Attempt to solve if it's an equation
                if '=' in preprocessed:
                    # Split only on the first '=' to handle multiple equals
                    eq_parts = preprocessed.split('=')
                    if len(eq_parts) < 2:
                        return "Error: Invalid equation format."
                    lhs = eq_parts[0]
                    rhs = '='.join(eq_parts[1:])
                    lhs_expr = parse_expr(lhs, local_dict=SAFE_DICT, transformations=transformations)
                    rhs_expr = parse_expr(rhs, local_dict=SAFE_DICT, transformations=transformations)
                    solution = sympy.solve(lhs_expr - rhs_expr, dict=True)
                    if solution:
                        return ", ".join([str(sol) for sol in solution])
                    else:
                        return "Error: No solution found for the equation."
                # Otherwise try to simplify
                simplified = sympy.simplify(parsed_expr)
                if simplified != parsed_expr:
                    return f"Simplified: {simplified}"
                return f"Symbolic expression: {parsed_expr}"
            except Exception as e:
                return f"Symbolic expression: {parsed_expr}"

        # Evaluate the expression to a numerical value
        result = parsed_expr.evalf()

        # Check if the result is a symbolic expression that couldn't be fully evaluated
        if result.has(sympy.Symbol):
            try:
                mathematica_expr = parse_mathematica(stripped_expression)
                result_mathematica = mathematica_expr.evalf()
                if result_mathematica.has(sympy.Symbol):
                    return f"Error: Expression '{stripped_expression}' could not be fully evaluated to a number. Result: {result}"
                result = result_mathematica
            except Exception:
                return f"Error: Expression '{stripped_expression}' could not be fully evaluated to a number. Result: {result}"

        # Format the result
        if result == sympy.zoo:  # Complex infinity
            return "Error: Result is complex infinity."
        if result == sympy.oo:  # Positive infinity
            return "Error: Result is positive infinity."
        if result == -sympy.oo:  # Negative infinity
            return "Error: Result is negative infinity."
        if result == sympy.nan:  # Not a Number
            return "Error: Result is Not a Number (NaN)."

        # If the result is an integer, display it without decimal places
        if result.is_Integer:
            return str(int(result))
        # If the result is a rational, display as fraction or decimal
        elif result.is_Rational and not result.is_Integer:
            return str(float(result))  # Convert to float for consistent decimal output
        else:
            # For floats and other numerical types
            return str(float(result))  # Ensure standard float string representation

    except (SyntaxError, TypeError, ValueError) as e:
        return f"Error: Invalid expression or syntax. Details: {e}"
    except ZeroDivisionError:
        return "Error: Division by zero."
    except AttributeError as e:
        return f"Error: Could not evaluate. Attribute error: {e}"
    except Exception as e:
        return f"Error: Could not evaluate expression '{stripped_expression}'. {type(e).__name__}: {str(e)}"

if __name__ == '__main__':
    test_expressions = [
        "2 + 2",
        "10 / (2 + 3) - 1",
        "sqrt(16) + 2^3",
        "pi * 2",
        "sin(pi/2)",
        "log(E)",
        "log(10, 2)",
        "1/0",
        "factorial(5)",
        "Abs(-5.5)",
        "2*x + y",
        "integrate(x, x)",
        "solve(x**2 - 1, x)",
        "nonsense_function(5)",
        "1 + ",
        "import os",
        "exec('print(1)')",
        "sqrt(gamma(5))",
        "E^(I*pi) + 1",
        "conjugate(2+3*I)",
        "re(4-5*I)",
        "im(6+7*I)",
        "50% of 200",
        "3π",
        "2°",
        "max(3,5)",
        "x^2 + 2x + 1 = 0",
        "∑(i, (i, 1, 10))"
    ]
    for expr_str in test_expressions:
        print(f"'{expr_str}': {calculate_sympy(expr_str)}")

    print(f"'N[Log[10,2], 10]': {calculate_sympy('N[Log[10,2], 10]')}")
    print(f"'Log[10,2].evalf()': {calculate_sympy('Log[10,2].evalf()')}")