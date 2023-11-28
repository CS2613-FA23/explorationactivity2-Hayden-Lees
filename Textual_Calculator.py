#Hayden Lees 3649173 Textual Calculator V2
import re
import inspect
from functools import reduce
import torch as tc
import numpy as np

#COMMAND COMPREHENSION
CMD_REG = re.compile(r"""
                     ^\s*
                     (?P<exit>exit|close|quit|esc(ape)?|$)|
                     (?P<help>help|explain\ (normal|lang|equation|matrix))|
                     (?P<normal>norm(al)?|eval(uate)?|comp(are)?)|
                     (?P<lang>shapes)|
                     (?P<equation>equation)|
                     (?P<matrix>matrix|vector)\s*$
                     """, re.X)
NORM_REG = re.compile(r"""
                      ^\s*
                      (?P<return>b(ack\ out|o)|c(hange\ modes?|m)|\.)|
                      ((?P<first>.+?)
                      ((?P<second><=?|>=?|!=|==|and|\||&)
                      (?P<third>.+))?)
                      $""", re.X)
DECOMPOSE_REG = re.compile(r"""
                          ((?P<func_name>[a-z][a-z0-9_]*)\((?P<func_value>.+?\)?)\))|
                          (?P<operand>(\d+(\.\d+)?)|[a-z][a-z0-9_]*)|
                          (?P<open>\()|
                          (?P<single>[!])|
                          (?P<dual>[+\-/%*^])|
                          (?P<close>\))
                          """, re.X)
MATRIX_EV_REG = re.compile(r"""
                           ((?P<func_name>[a-z][a-z0-9_]*)\((?P<func_value>.+?\)?)\))|
                           (?P<open>\()|(?P<close>\))|
                           (?P<mono>\^t|\^-1)|
                           (?P<bin>\.|\#|[+\-*/^]|\ )|
                           (?P<var>[a-zA-Z][a-z0-9_]*)|(?P<raw>(\[.+?\])|(\d+(\.\d+)?))
                           #MATRIX MONO OPERATORS (det, trans, inver, trace)
                           #MATRIX BINARY OPERATORS (dot, cross, ewise operations)
                           """, re.X)
CUSTOM_VAR_REG = re.compile(r"""\s*
                            (?P<var>[a-z][a-z0-9_]*)
                            \s*=\s*
                            (?P<rest>.+)\s*
                            """, re.X)
TO_TENSOR_REG = re.compile(r"""\s*
                           (?P<var>[A-Z]\d*)\s*=\s*
                           \[
                           (?P<rest>.+?)
                           \]\s*
                            """, re.X)

IS_DEBUG = False
DEBUG = lambda x: print(f"DEBUG ({inspect.stack()[1].lineno}): {x}") if IS_DEBUG else x
ERROR_DELTA = 1e-15
ERR_UNKNOWN_CMD = "ERROR: Unknown cmd"
ERR_UNKNOWN_VAR = "ERROR: Unknown var - "
ERR_UNKNOWN_QSTN = "ERROR: Unknown question"
ERR_UNBALANCED = "ERROR: Unbalanced ()"
ERR_BAD_FORMAT = "ERROR: Invalid formating"
ERR_UNKNOWN_FN = "ERROR: Unknown Function - "
ERR_BAD_VAR_NAME = "ERROR: Unable to overwrite - "
ERR_BAD_EVAL = "ERROR: Unable to evaluate - "
WARN_FACTORIAL = "WARNING: The current factorial function (!) truncates the value given to it into a int"
HELP_STRINGS = {
    "help": """This is where you can input which mode you would like (currently only normal mode or matrix mode)
you can also ask for an explaination of each mode by typing \"explain <MODE>\", 
you can exit the program by inputing [exit, close, quit, escape, esc] or by inputing nothing, 
to exit a mode by inputing [back, change mode, or .]""",
    "explain normal": """In the normal mode you can assign variables (<VAR> = ...), 
evaluate expressions (5^2, sin(pi/2), ...), 
and compare expressions (\"sin(2) and cos(2)\" returns \"sin(2) > cos(2)\", \"e^-1 < pi^-1\" returns \"e^-1 < pi^-1 is False\").
currently supports
    numpy trig (including hyperbolic), 
    inverse trig (including hyperbolic),
    exponential and logarithm,
    factorial (positive integers only),
    modulus,
    basic operations (+ - * /)""",
    "explain lang": "This is currently a w.i.p, it will be able to take verbos language questions and evaluate them (what is the area of a regular pentagon with a side length of 1)",
    "explain equation": "This is currently a w.i.p, it will be able to take in equations like (x^2-1 = 0) and return the answers which would be (x = 1 and -1)",
    "explain matrix": """In the matrix mode you can assign matrix variables (<A-Z>[<num>] = [x11,x21,x31;x21,x22,x23;x31,x32,x33]),
evaluate matrix expressions (3 + A, V.V, M M^-1, ...)
currently supports
    determinants, inverse, trace, transpose
    vector products (dot \".\" and cross \"#\")
    matrix multiplication ([...] [...])
    element wise operations (+ - * / ^)"""
}
special_values = {
    "pi": np.pi,
    "e": np.e,
    "ans": 0,
    "I2": tc.tensor([[1,0],[0,1]])
}
IMMUTABLE_values = [
    "pi",
    "e",
    "ans",
    "I2"
]
MATH_FUNCS = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "csc": lambda x: 1/np.sin(x),
    "sec": lambda x: 1/np.cos(x),
    "cot": lambda x: 1/np.tan(x),
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "csch": lambda x: 1/np.sinh(x),
    "sech": lambda x: 1/np.cosh(x),
    "coth": lambda x: 1/np.tanh(x),
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "arccsc": lambda x: np.arcsin(1/x),
    "arcsec": lambda x: np.arccos(1/x),
    "arccot": lambda x: np.arctan(1/x),
    "arcsinh": np.arcsinh,
    "arccosh": np.arccosh,
    "arctanh": np.arctanh,
    "arccsch": lambda x: np.arcsinh(1/x),
    "arcsech": lambda x: np.arccosh(1/x),
    "arccoth": lambda x: np.arctanh(1/x),
    "ln": np.log,
    "log_": lambda x,b: np.log(x)/np.log(b)
}

def add(x: float|int, y: float|int) -> float|int: return x + y
def minus(x: float|int, y: float|int) -> float|int: return x - y
def mult(x: float|int, y: float|int) -> float|int: return x * y
def divid(x: float|int, y: float|int) -> float|int: return x / y
def mod(x: float|int, y: float|int) -> float|int: return x % y
def power(x: float|int, y: float|int) -> float|int: return x ** y
def factorial(x: float|int):
        if type(x) == float:
            print(WARN_FACTORIAL)
            x = int(x)
        out = 1
        for z in range(x): out *= x-z
        return out
def calcEquals(x: float|int, y: float|int) -> bool: return (abs(x - y) < ERROR_DELTA)
def calcNotEq(x: float|int, y: float|int) -> bool: return not calcEquals(x,y)
def calcLess(x: float|int, y: float|int) -> bool: return calcNotEq(x,y)*(x<y)
def calcLessEq(x: float|int, y: float|int) -> bool: return calcEquals(x,y) or calcLess(x,y)
def calcGreater(x: float|int, y: float|int) -> bool: return calcNotEq(x,y)*(x>y)
def calcGreaterEq(x: float|int, y: float|int) -> bool: return calcEquals(x,y) or calcGreater(x,y)
def calcComp(x: float|int, y: float|int) -> int: return calcLess(x,y)*(-1)+calcGreater(x,y)*(1)
def transpose(x: tc.Tensor): return tc.transpose(x,0,1)
def inver(x: tc.Tensor): 
    try: return tc.linalg.inv(x)
    except RuntimeError: return ERR_BAD_EVAL + "Determinant is 0"
OPERATORS = {
    "+": add,
    "-": minus,
    "*": mult,
    "/": divid,
    "%": mod,
    "^": power,
    "!": factorial
}
COMPARATORS = {
    "==": calcEquals,
    "!=": calcNotEq,
    "<": calcLess,
    "<=": calcLessEq,
    ">": calcGreater,
    ">=": calcGreaterEq,
    "and": calcComp,
    "|": calcComp,
    "&": calcComp
}
ORDER = {
    "+": 1,
    "-": 1,
    "*": 2,
    "/": 2,
    "%": 2,
    "^": 3,
    "!": 4,
}
MATRIX_OPS = {
    "^t": transpose,
    "^-1": inver,
    ".": tc.linalg.vecdot,
    "#": tc.linalg.cross,
    "+": tc.add,
    "-": tc.sub,
    "*": tc.mul,
    "/": tc.div,
    "^": tc.pow
}
MATRIX_MONO_OPS = ["^t","^-1"]
VECTOR_OPS = [".","#"]
MATRIX_FUNC = {
    "det": tc.det,
    "tr": tc.trace
}


def evaluate(s: str):
    operators = DECOMPOSE_REG.finditer(s)
    in_open = 0
    ind_open = []
    substr = []
    for x in operators:
        if x["open"]:
            in_open += 1
            ind_open.append(x.span("open")[1])
        if in_open == 0:
            if x["func_name"] and x["func_value"]:
                substr.append(x["func_name"]+":"+x["func_value"])
            if x["operand"]:
                substr.append(x["operand"])
            if x["single"]:
                if x.span("single")[0] == 0: return ERR_BAD_FORMAT
                substr.append(x["single"])
            if x["dual"]:
                if x.span("dual")[0] == 0: 
                    if x["dual"] not in "-+": return ERR_BAD_FORMAT
                substr.append(x["dual"])
        if x["close"]:
            if in_open == 0: return ERR_UNBALANCED
            in_open -= 1
            end = x.span("close")[0]
            substr.append(s[ind_open[-1]:end])
    if len(substr) == 1:
        match substr[0]:
            case _ if substr[0].isnumeric():
                return float(substr[0])
            case _ if substr[0] in special_values:
                return special_values[substr[0]]
            case _ if ":" in substr[0]:
                l = substr[0].split(":")
                v = evaluate(l[1])
                if type(v) == str: return v
                if "log_" in l[0]: 
                    base = l[0].split("_")[1]
                    if base.isnumeric(): base = float(base)
                    else: return ERR_UNKNOWN_FN + l[0]
                    l[0] = "log_"
                    return MATH_FUNCS[l[0]](v,base)
                if l[0] not in MATH_FUNCS: return ERR_UNKNOWN_FN + l[0]
                return MATH_FUNCS[l[0]](v)
            case _:
                return ERR_BAD_EVAL + substr[0]
    for x in range(len(substr)):
        if substr[x] in "+-/*^%!": continue
        if ":" in substr[x]: 
            l = substr[x].split(":")
            v = evaluate(l[1])
            if type(v) == str: return v
            if "log_" in l[0]: 
                base = l[0].split("_")[1]
                if base.isnumeric(): base = float(base)
                else: return ERR_UNKNOWN_FN + l[0]
                l[0] = "log_"
                return MATH_FUNCS[l[0]](v,base)
            if l[0] not in MATH_FUNCS: return ERR_UNKNOWN_FN + l[0]
            substr[x] = MATH_FUNCS[l[0]](v)
            continue
        if substr[x] in special_values: 
            substr[x] = special_values[substr[x]] #MAKE SURE THE SPECIAL VALUE IS EVALUATED...
            continue
        if not substr[x].isnumeric():
            substr[x] = evaluate(substr[x])
            continue
        substr[x] = float(substr[x])
    while len(substr) > 1:
        max_val = 0
        ind_of = 0
        if type(substr[0]) == str: #if it starts with + or -
            substr[1] = OPERATORS[substr[0]](0,substr[1])
            substr.pop(0)
            continue
        for x in range(len(substr)):
            if substr[x] in ORDER:
                if ORDER[substr[x]] > max_val: 
                    max_val = ORDER[substr[x]]
                    ind_of = x
        if substr[ind_of] == "!": 
            substr[ind_of-1] = OPERATORS[substr[ind_of]](substr[ind_of-1])
            substr.pop(ind_of)
            continue
        if type(substr[ind_of+1]) == str:
            if substr[ind_of+1] in "+-":
                substr[ind_of+1] = OPERATORS[substr[ind_of+1]](0,substr[ind_of+2])
                substr.pop(ind_of+2)
            else: return ERR_BAD_FORMAT
        substr[ind_of-1] = OPERATORS[substr[ind_of]](substr[ind_of-1],substr[ind_of+1])
        substr.pop(ind_of)
        substr.pop(ind_of)
    if type(substr[0]) == str: return ERR_BAD_FORMAT
    return substr[0]

def matrix_ev(s: str):
    DEBUG(s)
    decomp = [*MATRIX_EV_REG.finditer(s)]
    DEBUG(decomp)
    layer = 0
    openInd = []
    chunks = []
    multi_op = False
    for m in decomp:
        DEBUG(m.groupdict())
        if m["open"]:
            if layer == 0: openInd.append(m.span("open")[1])
            layer += 1
            continue
        if m["close"]:
            if layer == 0: return ERR_UNBALANCED
            layer -= 1
            if layer == 0: 
                val = matrix_ev(s[openInd[-1]:m.span("close")[0]])
                if type(val) == str: return val
                chunks.append(val)
            continue
        if layer != 0: continue
        if m["func_name"] and m["func_value"]:
            if m["func_name"] in MATRIX_FUNC:
                val = matrix_ev(m["func_value"])
                if type(val) == str: return val
                if tc.is_tensor(val):
                    chunks.append(MATRIX_FUNC[m["func_name"]](val))
                else: return ERR_BAD_EVAL
            else: return ERR_UNKNOWN_FN + m["func_name"]
        match m["mono"]:
            case None: pass
            case "^t"|"^-1" if m.span("mono")[0] == 0 or multi_op: return ERR_BAD_FORMAT
            case "^t"|"^-1": 
                chunks.append(m["mono"])
                multi_op = False
                continue
        match m["bin"]:
            case None: pass
            case " ": continue
            case _ if m.span("bin")[1] == len(s): return ERR_BAD_FORMAT 
            case "+"|"-":
                chunks.append(m["bin"])
                multi_op = True 
                continue
            case _ if m.span("bin")[0] == 0 or multi_op: return ERR_BAD_FORMAT #
            case _ if m["bin"]: 
                chunks.append(m["bin"])
                continue
        if m["raw"]:
            if "[" in m["raw"]:
                rows = re.finditer(r".+?(?=;|$)",m["raw"][1:-1])
                construct = []
                for row in rows: construct.append([*map(lambda x: float(x[0]),[*re.finditer(r"((\+|\-)?\d+(\.\d+)?)(?=,|$)",row[0])])])
                chunks.append(tc.tensor(construct))
                continue
            else:
                chunks.append(float(m["raw"]))
                continue
        if m["var"]:
            if m["var"] in special_values:
                chunks.append(special_values[m["var"]])
            else: return ERR_UNKNOWN_VAR + m["var"]
    if layer != 0: return ERR_UNBALANCED
    has_mono = True
    has_back = True
    has_vect = True
    DEBUG(chunks)
    if len(chunks) == 1: return chunks[0]
    while len(chunks) > 1:
        if has_mono:
            ind = -1
            for x in MATRIX_MONO_OPS:
                try: 
                    tmp = chunks.index(x)
                    if ind == -1: ind = tmp
                    elif ind > tmp: ind = tmp
                except ValueError: continue
            DEBUG(ind)
            if ind != -1:
                if not tc.is_tensor(chunks[ind-1]): return ERR_BAD_EVAL
                chunks[ind-1] = MATRIX_OPS[chunks[ind]](chunks[ind-1])
                chunks.pop(ind)
                continue
            else:
                has_mono = False
                continue
        elif has_back:
            ind = -1
            for i in range(1,len(chunks)):
                if chunks[i] in MATRIX_OPS or chunks[i-1] in MATRIX_OPS: continue
                else:
                    ind = i
                    break
            if ind != -1:
                dtypes = (type(chunks[ind]),type(chunks[ind-1]))
                if dtypes[1] == tc.Tensor:
                    if dtypes[0] not in [int,float,tc.Tensor]: return ERR_BAD_EVAL
                    if dtypes[0] == tc.Tensor:
                        chunks[ind-1] = chunks[ind-1] @ chunks[ind]
                    else:
                        chunks[ind-1] = tc.mul(chunks[ind-1],chunks[ind])
                    chunks.pop(ind)
                    continue
                elif dtypes[0] == tc.Tensor:
                    chunks[ind-1] = tc.mul(chunks[ind],chunks[ind-1])
                    chunks.pop(ind)
                    continue
                else:
                    chunks[ind-1] *= chunks[ind]
                    chunks.pop(ind)
            else: 
                has_back = False
                continue
        elif has_vect:
            DEBUG(chunks) 
            ind = -1
            for x in VECTOR_OPS:
                try: 
                    tmp = chunks.index(x)
                    if ind == -1: ind = tmp
                    elif ind > tmp: ind = tmp
                except ValueError: continue
            if ind != -1:
                if not tc.is_tensor(chunks[ind-1]) or not tc.is_tensor(chunks[ind+1]): return ERR_BAD_EVAL
                chunks[ind-1] = MATRIX_OPS[chunks[ind]](chunks[ind-1],chunks[ind+1])
                chunks.pop(ind)
                chunks.pop(ind)
                continue
            else:
                has_vect = False
                continue
        else:
            DEBUG(chunks)
            max_val = 0
            ind_of = 0
            if type(chunks[0]) == str: #if it starts with + or -
                chunks[1] = MATRIX_OPS[chunks[0]](0,chunks[1])
                chunks.pop(0)
            for x in range(len(chunks)):
                if chunks[x] in ORDER:
                    if ORDER[chunks[x]] > max_val: 
                        max_val = ORDER[chunks[x]]
                        ind_of = x
            if type(chunks[ind_of+1]) == str:
                chunks[ind_of+1] = MATRIX_OPS[chunks[ind_of+1]](0,chunks[ind_of+2])
                chunks.pop(ind_of+2)
            chunks[ind_of-1] = MATRIX_OPS[chunks[ind_of]](chunks[ind_of-1],chunks[ind_of+1])
            chunks.pop(ind_of)
            chunks.pop(ind_of)
    return chunks[0] #T T^-1 = I but T*T^-1 != I, this is intentional
    
def custom_Var(m):
    if m["var"] in IMMUTABLE_values:
        print(ERR_BAD_VAR_NAME + m["var"])
    else:
        match m["rest"]:
            case "ans":
                special_values[m["var"]] = special_values["ans"]
            case _ if m["rest"].isnumeric():
                special_values[m["var"]] = float(m["rest"])
            case _ if "," in m["rest"] or ";" in m["rest"]:
                rows = re.finditer(r".+?(?=;|$)",m["rest"])
                construct = []
                for row in rows: construct.append([*map(lambda x: float(x[0]),[*re.finditer(r"((\+|\-)?\d+(\.\d+)?)(?=,|$)",row[0])])])
                DEBUG(construct)
                while len(construct) == 1: construct = construct[0]
                special_values[m["var"]] = tc.tensor(construct)
            case _:
                special_values[m["var"]] = m["rest"]  
    DEBUG(special_values)

def printTensor(x: tc.Tensor):
    if x.ndim == 0: 
        print(f"ANSWER: {x.item()}")
        return
    s = "┌\n. "
    if x.ndim == 1:
        for i in range(x.size(0)):
            s += str(x[i].item())
            if i+1 != x.size(0): s += " "
    else:        
        for i in range(x.size(0)):
            for j in range(x.size(1)):
                s += str(x[i][j].item())
                if j+1 != x.size(1): s += " "
            if i+1 != x.size(0): s += "\n  "
    width = reduce(lambda a, b: a if a > b else b,[*map(lambda x: len(x),s.split("\n"))])
    s += "\n" + " "*width + " ┘"
    print(f"ANSWER: \n{s}")

def __main__():
    print("Welcome To The Text Based Calculator!")
    while True:
        uin = input("Input a cmd: ").lower()
        fresh = True
        cmd = CMD_REG.fullmatch(uin)
        if not cmd:
            make_var = CUSTOM_VAR_REG.fullmatch(uin)
            if not make_var:
                print(ERR_UNKNOWN_CMD + " - " + uin)
                print("type \"help\" or ask for an explaination of a mode by entering \"explain <MODE>\"")
                continue
            custom_Var(make_var)
            continue
        if cmd["exit"] or cmd["exit"] == "": break
        if cmd["help"]: 
            print(HELP_STRINGS[cmd["help"]])
            continue
        while cmd["normal"]:
            uin = input("Input a statement to evaluate: ").lower()
            if re.match(r"\s*$", uin): continue
            make_var = CUSTOM_VAR_REG.fullmatch(uin)
            if make_var:
                custom_Var(make_var)
                continue
            qstn = NORM_REG.search(uin)
            if not qstn: print(ERR_UNKNOWN_QSTN + " - " + uin)
            elif qstn["return"]: break
            elif qstn["third"] and qstn["second"]:
                operator_1 = evaluate(qstn["first"])
                if type(operator_1) == str: 
                    print(operator_1)
                    continue
                operator_2 = evaluate(qstn["third"])
                if type(operator_2) == str:
                    print(operator_2)
                    continue
                out = COMPARATORS[qstn["second"]](operator_1,operator_2)
                DEBUG(f"DEBUG: {operator_1} {operator_2} {out}")
                match qstn["second"]:
                    case _ if qstn["second"] in ["and","|","&"]:
                        match out:
                            case -1:
                                out = "less than"
                            case 0:
                                out = "equal to"
                            case 1:
                                out = "greater than"
                        print(f"{operator_1} is {out} {operator_2}")
                    case _:
                        if out: out = "is True"
                        else: out = "is False"
                        print(f"{operator_1} {qstn['second']} {operator_2} {out}")
            else:
                retval = evaluate(qstn["first"])
                if not retval: continue
                elif type(retval) != str:
                    special_values["ans"] = retval
                    print(f"ANSWER: {special_values['ans']}")
                else:
                    print(retval)
        while cmd["matrix"]:
            if fresh: print("Welcome to Matrix mode (to make a matrix use <VAR> = [] with , between elements of a row and ; between rows)")
            fresh = False
            uin = input("Input a statement to evaluate: ")
            if re.match(r"\s*$", uin): continue
            make_var = TO_TENSOR_REG.fullmatch(uin)
            if make_var:
                custom_Var(make_var)
                continue
            make_var = CUSTOM_VAR_REG.fullmatch(uin)
            if make_var:
                custom_Var(make_var)
                continue
            if re.fullmatch(r"b(ack\ out|o)|c(hange\ modes?|m)|\.", uin): break
            retval = matrix_ev(uin)
            if retval is None: continue
            elif type(retval) != str:
                special_values["ans"] = retval
                if tc.is_tensor(retval): printTensor(retval)
                else: print(f"ANSWER: {special_values['ans']}")
            else:
                print(retval)

        while cmd["lang"]:
            print("language is currently a w.i.p")
            break
        while cmd["equation"]:
            print("equation is currently a w.i.p")
            break


if __name__ == "__main__":
    __main__()