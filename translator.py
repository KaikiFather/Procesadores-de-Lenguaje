
import sys
from sly import Lexer, Parser

# --------- Lexer ---------
class CLexer(Lexer):
    tokens = {
        'FID', 'ID', 'CTEENT', 'ASSIGN', 'OR', 'AND', 'NOT',
        'EQ', 'NE', 'LE', 'GE', 'LT', 'GT', 'INT', 'RETURN',
        'VOID', 'STRING', 'AMP', 'IF', 'ELSE'
    }
    literals = {'+', '-', '*', '/', '(', ')', ';', '{', '}', ',', '[', ']'}
    ignore = ' \t'

    @_(r'//[^\n]*')
    def comment(self, t):
        pass

    @_(r'[a-zA-Z_][a-zA-Z0-9_]*(?=\s*\()')
    def FID(self, t):
        if t.value == 'int':
            t.type = 'INT'
        elif t.value == 'void':
            t.type = 'VOID'
        elif t.value == 'return':
            t.type = 'RETURN'
        elif t.value == 'if':
            t.type = 'IF'
        return t

    INT     = r'int'
    VOID    = r'void'
    RETURN  = r'return'
    IF      = r'if'
    ELSE    = r'else'

    ID      = r'[a-zA-Z_][a-zA-Z0-9_]*'
    CTEENT  = r'\d+'

    OR      = r'\|\|'
    AND     = r'&&'
    EQ      = r'=='
    NE      = r'!='
    LE      = r'<='
    GE      = r'>='
    LT      = r'<'
    GT      = r'>'
    ASSIGN  = r'='
    NOT     = r'!'
    AMP     = r'&'
    STRING  = r'"([^\\"]|\\.)*"'

    @_(r'\n+')
    def newline(self, t):
        self.lineno += t.value.count('\\n')

    def error(self, t):
        print(f"Illegal character '{t.value[0]}'", file=sys.stderr)
        self.index += 1

# --------- Parser builds an AST ---------
class CParser(Parser):
    tokens = CLexer.tokens
    precedence = (
        ('nonassoc', 'IFX'),
        ('nonassoc', 'ELSE'),
        ('left', 'OR'),
        ('left', 'AND'),
        ('left', 'EQ', 'NE', 'LT', 'GT', 'LE', 'GE'),
        ('left', '+', '-'),
        ('left', '*', '/'),
        ('right', 'NOT', 'AMP', 'UMINUS', 'DEREF'),
    )

    def __init__(self):
        self.module_items = []  # sequence of top-level declarations and function defs

    # ---- driver rules ----
    @_( 'lines line' )
    def lines(self, p): return None

    @_('')
    def lines(self, p): return None

    @_( 'stmt' )
    def line(self, p):
        # accept function definitions and ignore declarations handled by var_decl+; rule
        if p.stmt is not None:
            self.module_items.append(p.stmt)
        return None

    @_( 'var_decl ";"' )
    def line(self, p):
        self.module_items.append(('globals', p.var_decl))
        return None

    # ---------- declarations ----------
    @_( 'INT declarator_list' )
    def var_decl(self, p): return p.declarator_list

    @_( 'declarator' )
    def declarator_list(self, p): return [p.declarator]

    @_( 'declarator_list "," declarator' )
    def declarator_list(self, p): return p.declarator_list + [p.declarator]

    @_( 'ID arr_suffix opt_init' )
    def declarator(self, p): return ('array', p.ID, p.arr_suffix, p.opt_init)

    @_( '"*" ID opt_init' )
    def declarator(self, p): return ('ptr', p.ID, p.opt_init)

    @_( 'ID opt_init' )
    def declarator(self, p): return ('var', p.ID, p.opt_init)

    @_( 'ASSIGN t_expr' )
    def opt_init(self, p): return p.t_expr

    @_( 'ASSIGN "{" init_list_opt "}"' )
    def opt_init(self, p): return ('initlist', p.init_list_opt)

    @_( '' )
    def opt_init(self, p): return None

    @_( 'dim arr_suffix_opt' )
    def arr_suffix(self, p): return [p.dim] + p.arr_suffix_opt

    @_( 'arr_suffix' )
    def arr_suffix_opt(self, p): return p.arr_suffix

    @_( '' )
    def arr_suffix_opt(self, p): return []

    @_( '"[" CTEENT "]"' )
    def dim(self, p): return int(p.CTEENT)

    @_( '"[" "]"' )
    def dim(self, p): return None

    @_( 'init_list' )
    def init_list_opt(self, p): return p.init_list

    @_( '' )
    def init_list_opt(self, p): return []

    @_( 't_expr' )
    def init_list(self, p): return [p.t_expr]

    @_( 'init_list "," t_expr' )
    def init_list(self, p): return p.init_list + [p.t_expr]

    # ---------- funciones ----------
    @_( 'INT' )
    def type_spec(self, p): return 'int'

    @_( 'VOID' )
    def type_spec(self, p): return 'void'

    @_( 'type_spec FID "(" param_list_opt ")" compound_stmt' )
    def stmt(self, p):
        return ('fun', p.type_spec, p.FID, p.param_list_opt, p.compound_stmt)

    @_( 'type_spec FID "(" param_list_opt ")" ";"' )
    def stmt(self, p):
        return None  # prototype, ignore for codegen

    @_( 'param_list' )
    def param_list_opt(self, p): return p.param_list

    @_( '' )
    def param_list_opt(self, p): return []

    @_( 'param' )
    def param_list(self, p): return [p.param]

    @_( 'param_list "," param' )
    def param_list(self, p): return p.param_list + [p.param]

    @_( 'INT ID' )
    def param(self, p): return ('param', p.ID)

    @_( 'INT "*" ID' )
    def param(self, p): return ('paramptr', p.ID)

    # ---------- statements & blocks ----------
    @_( 'compound_stmt' )
    def stmt(self, p): return ('block', p.compound_stmt)

    @_( 't_expr ";"' )
    def stmt(self, p): return ('exprstmt', p.t_expr)

    @_( 'RETURN t_expr ";"' )
    def stmt(self, p): return ('return', p.t_expr)

    @_( 'RETURN ";"' )
    def stmt(self, p): return ('return', None)

    @_( '";"' )
    def stmt(self, p): return None

    @_( '"{" item_list_opt "}"' )
    def compound_stmt(self, p): return p.item_list_opt

    @_( 'item_list' )
    def item_list_opt(self, p): return p.item_list

    @_( '' )
    def item_list_opt(self, p): return []

    @_( 'item' )
    def item_list(self, p): return [p.item]

    @_( 'item_list item' )
    def item_list(self, p): return p.item_list + [p.item]

    @_( 'stmt' )
    def item(self, p):
        if isinstance(p.stmt, tuple) and p.stmt and p.stmt[0] == 'block':
            return p.stmt[1]
        return p.stmt

    @_( 'var_decl ";"' )
    def item(self, p): return ('decls', p.var_decl)

    # ---------- expresiones ----------
    @_( 'lvalue ASSIGN t_expr' )
    def t_expr(self, p): return ('assign_l', p.lvalue, p.t_expr)

    @_( 'or_expr' )
    def t_expr(self, p): return p.or_expr

    @_( '"*" unary %prec DEREF' )
    def lvalue(self, p): return ('deref', p.unary)

    @_( 'postfix' )
    def lvalue(self, p): return p.postfix

    @_( 'or_expr OR and_expr' )
    def or_expr(self, p): return ('or', p.or_expr, p.and_expr)

    @_( 'and_expr' )
    def or_expr(self, p): return p.and_expr

    @_( 'and_expr AND eq_expr' )
    def and_expr(self, p): return ('and', p.and_expr, p.eq_expr)

    @_( 'eq_expr' )
    def and_expr(self, p): return p.eq_expr

    @_( 'eq_expr EQ rel_expr' )
    def eq_expr(self, p): return ('==', p.eq_expr, p.rel_expr)

    @_( 'eq_expr NE rel_expr' )
    def eq_expr(self, p): return ('!=', p.eq_expr, p.rel_expr)

    @_( 'rel_expr' )
    def eq_expr(self, p): return p.rel_expr

    @_( 'rel_expr LT arith_expr' )
    def rel_expr(self, p): return ('<', p.rel_expr, p.arith_expr)

    @_( 'rel_expr GT arith_expr' )
    def rel_expr(self, p): return ('>', p.rel_expr, p.arith_expr)

    @_( 'rel_expr LE arith_expr' )
    def rel_expr(self, p): return ('<=', p.rel_expr, p.arith_expr)

    @_( 'rel_expr GE arith_expr' )
    def rel_expr(self, p): return ('>=', p.rel_expr, p.arith_expr)

    @_( 'arith_expr' )
    def rel_expr(self, p): return p.arith_expr

    @_( 'arith_expr "+" mul_expr' )
    def arith_expr(self, p): return ('+', p.arith_expr, p.mul_expr)

    @_( 'arith_expr "-" mul_expr' )
    def arith_expr(self, p): return ('-', p.arith_expr, p.mul_expr)

    @_( 'mul_expr' )
    def arith_expr(self, p): return p.mul_expr

    @_( 'mul_expr "*" unary' )
    def mul_expr(self, p): return ('*', p.mul_expr, p.unary)

    @_( 'mul_expr "/" unary' )
    def mul_expr(self, p): return ('/', p.mul_expr, p.unary)

    @_( 'unary' )
    def mul_expr(self, p): return p.unary

    @_( '"-" unary %prec UMINUS' )
    def unary(self, p): return ('neg', p.unary)

    @_( 'NOT unary' )
    def unary(self, p): return ('not', p.unary)

    @_( 'AMP unary' )
    def unary(self, p): return ('addr', p.unary)

    @_( '"*" unary %prec DEREF' )
    def unary(self, p): return ('deref', p.unary)

    @_( 'postfix' )
    def unary(self, p): return p.postfix

    @_( 'factor' )
    def postfix(self, p): return p.factor

    @_( 'postfix "[" t_expr "]"' )
    def postfix(self, p): return ('index', p.postfix, p.t_expr)

    @_( 'IF "(" t_expr ")" stmt ELSE stmt' )
    def stmt(self, p): return ('if', p.t_expr, p.stmt0, p.stmt1)

    @_( 'IF "(" t_expr ")" stmt %prec IFX' )
    def stmt(self, p): return ('if', p.t_expr, p.stmt, None)

    @_( 'FID "(" arg_list_opt ")"' )
    def factor(self, p): return ('call', p.FID, p.arg_list_opt)

    @_( 'ID' )
    def factor(self, p): return ('id', p.ID)

    @_( 'CTEENT' )
    def factor(self, p): return ('const', int(p.CTEENT))

    @_( 'STRING' )
    def factor(self, p): return ('string', p.STRING[1:-1])

    @_( '"(" t_expr ")"' )
    def factor(self, p): return p.t_expr

    @_( 't_expr' )
    def arg(self, p): return p.t_expr

    @_( 'arg' )
    def arg_list(self, p): return [p.arg]

    @_( 'arg_list "," arg' )
    def arg_list(self, p): return p.arg_list + [p.arg]

    @_( '' )
    def arg_list_opt(self, p): return []

    @_( 'arg_list' )
    def arg_list_opt(self, p): return p.arg_list

# --------- Code Generator ---------
class AsmEmitter:
    def __init__(self):
        self.text = []
        self.data = []
        self.strings = {}
    def emit(self, s): self.text.append(s)
    def emit_data(self, s): self.data.append(s)
    def label(self, name): self.emit(f'{name}:')
    def get_output(self):
        out = []
        if self.data or self.strings:
            out.append('.data')
            for k, v in self.strings.items():
                out.append(f'{k}: .asciz "{v}"')
            out.extend(self.data)
            out.append('')
        out.append('.text')
        out.extend(self.text)
        return '\n'.join(out) + '\n'
    def add_string(self, value):
        key = f's{len(self.strings)}'
        # Preserve existing escape sequences like \n while only protecting quotes
        esc = value.replace('"', '\\"')
        self.strings[key] = esc
        return key

class CodeGen:
    def __init__(self, module_items):
        self.module_items = module_items
        self.em = AsmEmitter()
        self.globals = {}  # name -> {'kind': 'global'}
    # ---- helpers ----
    def is_const_expr(self, node):
        if isinstance(node, tuple):
            op = node[0]
            if op == 'const':
                return True
            if op in ('+', '-', '*', '/', 'neg'):
                return all(self.is_const_expr(x) for x in node[1:])
            if op == 'assign_l':
                # Assignment expressions can be constant if the RHS is constant.
                return self.is_const_expr(node[2])
            return False
        return False

    def eval_const(self, node):
        # integer-only
        tag = node[0]
        if tag == 'const': return node[1]
        if tag == 'neg': return - self.eval_const(node[1])
        if tag == 'assign_l':
            # For constant contexts, propagate the value of the RHS.
            return self.eval_const(node[2])
        a = self.eval_const(node[1]); b = self.eval_const(node[2])
        if tag == '+': return a + b
        if tag == '-': return a - b
        if tag == '*': return a * b
        if tag == '/': return int(a // b)
        raise ValueError('non-const')

    def gen_expr(self, node, fn):
        """Generate code that leaves result in %eax."""
        if node is None:
            return
        tag = node[0] if isinstance(node, tuple) else None
        if tag == 'const':
            self.em.emit(f'    movl ${node[1]}, %eax')
            return
        if tag == 'id':
            info = self.lookup(fn, node[1])
            if info['kind'] == 'local':
                self.em.emit(f'    movl {info["offset"]}(%ebp), %eax')
            elif info['kind'] == 'param':
                self.em.emit(f'    movl {info["offset"]}(%ebp), %eax')
            else:
                self.em.emit(f'    movl {node[1]}, %eax')
            return
        if tag == 'assign_l':
            lv, rv = node[1], node[2]
            if isinstance(lv, tuple) and lv[0] == 'id':
                self.gen_expr(rv, fn)
                info = self.lookup(fn, lv[1])
                if info['kind'] in ('local', 'param'):
                    self.em.emit(f'    movl %eax, {info["offset"]}(%ebp)')
                else:
                    self.em.emit(f'    movl %eax, {lv[1]}')
            return
        if tag == 'neg':
            self.gen_expr(node[1], fn)
            self.em.emit('    negl %eax')
            return
        if tag == 'addr':
            # address-of a variable
            target = node[1]
            if isinstance(target, tuple) and target[0] == 'id':
                info = self.lookup(fn, target[1])
                if info['kind'] in ('local','param'):
                    self.em.emit(f'    leal {info["offset"]}(%ebp), %eax')
                else:
                    self.em.emit(f'    movl ${target[1]}, %eax')
            return
        if tag == 'string':
            # place string in .data and load address
            lbl = self.em.add_string(node[1])
            self.em.emit(f'    movl ${lbl}, %eax')
            return
        if tag == 'call':
            callee = node[1]
            args = node[2]
            bytes_pushed = 0
            for arg in reversed(args):
                self.push_arg(arg, fn)
                bytes_pushed += 4
            self.em.emit(f'    call {callee}')
            if bytes_pushed:
                self.em.emit(f'    addl ${bytes_pushed}, %esp')
            return
        if tag in ('+', '-', '*', '/'):
            # left -> push, right -> eax, merge
            left, right = node[1], node[2]
            self.gen_expr(left, fn)
            self.em.emit('    pushl %eax')
            self.gen_expr(right, fn)
            self.em.emit('    movl %eax, %ebx')
            self.em.emit('    popl %eax')
            if tag == '+':
                self.em.emit('    addl %ebx, %eax')
            elif tag == '-':
                self.em.emit('    subl %ebx, %eax')
            elif tag == '*':
                self.em.emit('    imull %ebx, %eax')
            else:  # '/'
                self.em.emit('    cdq')
                self.em.emit('    idivl %ebx')
            return
        # basic fallback: do nothing

    def push_arg(self, node, fn):
        """Push the value of an expression node onto the stack."""
        if isinstance(node, tuple):
            tag = node[0]
            if tag == 'const':
                self.em.emit(f'    pushl ${node[1]}')
                return
            if tag == 'id':
                info = self.lookup(fn, node[1])
                if info['kind'] in ('local', 'param'):
                    self.em.emit(f'    pushl {info["offset"]}(%ebp)')
                else:
                    self.em.emit(f'    pushl {node[1]}')
                return
            if tag == 'addr' and isinstance(node[1], tuple) and node[1][0] == 'id':
                target = node[1][1]
                info = self.lookup(fn, target)
                if info['kind'] in ('local', 'param'):
                    self.em.emit(f'    leal {info["offset"]}(%ebp), %eax')
                    self.em.emit('    pushl %eax')
                else:
                    self.em.emit(f'    pushl ${target}')
                return
            if tag == 'string':
                lbl = self.em.add_string(node[1])
                self.em.emit(f'    pushl ${lbl}')
                return
            if tag == 'call':
                self.gen_expr(node, fn)
                self.em.emit('    pushl %eax')
                return
        self.gen_expr(node, fn)
        self.em.emit('    pushl %eax')

    def lookup(self, fn, name):
        if fn is None:
            return {'kind': 'global'}
        # locals
        if name in fn['locals']:
            return {'kind':'local', 'offset': fn['locals'][name]}
        # params
        if name in fn['params']:
            return {'kind':'param', 'offset': fn['params'][name]}
        # else global
        return {'kind': 'global'}

    # ---- codegen for statements ----
    def gen_stmt(self, node, fn):
        if node is None:
            return
        tag = node[0]
        if tag == 'exprstmt':
            self.gen_expr(node[1], fn)
            return
        if tag == 'assign_l':
            lv, rv = node[1], node[2]
            # only ID lvalues for now
            if isinstance(lv, tuple) and lv[0] == 'id':
                self.gen_expr(rv, fn)
                info = self.lookup(fn, lv[1])
                if info['kind'] in ('local', 'param'):
                    self.em.emit(f'    movl %eax, {info["offset"]}(%ebp)')
                else:
                    self.em.emit(f'    movl %eax, {lv[1]}')
            return
        if tag == 'if':
            cond, then_s, else_s = node[1], node[2], node[3]
            self.gen_expr(cond, fn)
            lbl_else = f'.Lelse_{id(node)}'
            lbl_end  = f'.Lend_{id(node)}'
            self.em.emit('    cmpl $0, %eax')
            self.em.emit(f'    je {lbl_else}')
            self.gen_stmt(then_s, fn)
            self.em.emit(f'    jmp {lbl_end}')
            self.em.emit(f'{lbl_else}:')
            if else_s is not None:
                self.gen_stmt(else_s, fn)
            self.em.emit(f'{lbl_end}:')
            return
        if tag == 'block':
            for item in node[1]:
                self.gen_stmt(item, fn)
            return
        if tag == 'decls':
            # declarations already accounted for in prologue; handle initializers
            for decl in node[1]:
                kind = decl[0]
                if kind != 'var':
                    continue
                name, init = decl[1], decl[2]
                if init is not None:
                    self.gen_expr(init, fn)
                    off = fn['locals'][name]
                    self.em.emit(f'    movl %eax, {off}(%ebp)')
            return
        if tag == 'return':
            if node[1] is not None:
                self.gen_expr(node[1], fn)
            # epilogue
            self.em.emit('    movl %ebp, %esp')
            self.em.emit('    popl %ebp')
            self.em.emit('    ret')
            return
        # other tags ignored for this stage

    def count_locals(self, body):
        # returns (count, names) and map name -> offset
        locals_list = []
        def walk(items):
            for it in items:
                if isinstance(it, tuple) and it and it[0] == 'decls':
                    for d in it[1]:
                        if d[0] == 'var':
                            locals_list.append(d[1])
                elif isinstance(it, list):
                    walk(it)
                elif isinstance(it, tuple) and it and it[0] == 'block':
                    walk(it[1])
        walk(body)
        offsets = {}
        off = -4
        for name in locals_list:
            offsets[name] = off
            off -= 4
        return len(locals_list), offsets

    def build_params(self, params):
        # build name -> offset map, first param at 8(%ebp)
        m = {}
        cur = 8
        for p in params:
            if p[0] in ('param','paramptr'):
                m[p[1]] = cur
                cur += 4
        return m

    def gen_function(self, rettype, name, params, body):
        fn = {
            'name': name,
            'rettype': rettype,
            'params': self.build_params(params),
            'locals': {},
        }
        nlocals, locals_map = self.count_locals(body)
        fn['locals'] = locals_map

        # prologue
        self.em.emit(f'.globl {name}')
        self.em.emit(f'.type {name}, @function')
        self.em.label(name)
        self.em.emit('    pushl %ebp')
        self.em.emit('    movl %esp, %ebp')
        if nlocals > 0:
            self.em.emit(f'    subl ${4*nlocals}, %esp')

        # init code for declarations at top level of body will be emitted by walking body
        for item in body:
            self.gen_stmt(item, fn)

        # ensure function returns even if no explicit return by checking the
        # last real instruction that was emitted. If control flow already ends
        # in a `ret`, avoid emitting a duplicate epilogue.
        needs_epilogue = True
        for line in reversed(self.em.text):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.endswith(':'):
                # Labels do not affect whether we need the epilogue.
                continue
            needs_epilogue = stripped != 'ret'
            break
        if needs_epilogue:
            self.em.emit('    movl %ebp, %esp')
            self.em.emit('    popl %ebp')
            self.em.emit('    ret')

    def gen_globals(self, decls):
        for decl in decls:
            kind = decl[0]
            if kind != 'var':
                continue
            name, init = decl[1], decl[2]
            if init is None:
                self.em.emit_data(f'{name}: .long 0')
            else:
                if isinstance(init, tuple) and self.is_const_expr(init):
                    val = self.eval_const(init)
                    self.em.emit_data(f'{name}: .long {val}')
                else:
                    # non-constant initializers for globals are not supported yet; default 0
                    self.em.emit_data(f'{name}: .long 0')

    def generate(self):
        # walk module items
        # first collect all global var decls
        for item in self.module_items:
            if isinstance(item, tuple) and item and item[0] == 'globals':
                self.gen_globals(item[1])
        # then functions
        for item in self.module_items:
            if isinstance(item, tuple) and item and item[0] == 'fun':
                _, rettype, name, params, body = item
                self.gen_function(rettype, name, params, body)
        return self.em.get_output()

def parse_source(src):
    lexer = CLexer()
    parser = CParser()
    parser.parse(lexer.tokenize(src))  # fills parser.module_items
    gen = CodeGen(parser.module_items)
    asm = gen.generate()
    return asm

def main():
    if len(sys.argv) < 2:
        print("uso: python translator.py fuente.c", file=sys.stderr)
        sys.exit(1)
    inpath = sys.argv[1]
    with open(inpath, 'r') as f:
        code = f.read()
    asm = parse_source(code)
    with open('assembly.out', 'w') as f:
        f.write(asm)

if __name__ == '__main__':
    main()
