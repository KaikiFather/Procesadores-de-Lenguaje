
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
        self.globals = {}
        self.label_counter = 0
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

    def fresh_label(self, prefix):
        lbl = f'.L{prefix}_{self.label_counter}'
        self.label_counter += 1
        return lbl

    def gen_address(self, node, fn):
        """Leave address of *node* in %eax."""
        tag = node[0]
        if tag == 'id':
            info = self.lookup(fn, node[1])
            if info['kind'] in ('local', 'param'):
                self.em.emit(f'    leal {info["offset"]}(%ebp), %eax')
            else:
                self.em.emit(f'    movl ${node[1]}, %eax')
            return
        if tag == 'index':
            base, idx = node[1], node[2]
            const_info = self.const_index_offset(node, fn)
            if const_info is not None:
                name, info, offset = const_info
                kind = info.get('kind')
                if kind == 'local':
                    adj = info['offset'] - offset
                    self.em.emit(f'    leal {adj}(%ebp), %eax')
                    return
                if kind == 'global':
                    if offset:
                        self.em.emit(f'    movl ${name}+{offset}, %eax')
                    else:
                        self.em.emit(f'    movl ${name}, %eax')
                    return
            if isinstance(base, tuple) and base[0] == 'id' and self.is_const_expr(idx):
                stride = self.index_stride(base, fn)
                offset = self.eval_const(idx) * stride
                info = self.lookup(fn, base[1])
                kind = info.get('kind')
                if info.get('type') == 'array':
                    if kind == 'local':
                        adj = info['offset'] - offset
                        self.em.emit(f'    leal {adj}(%ebp), %eax')
                        return
                    if kind == 'global':
                        if offset:
                            self.em.emit(f'    movl ${base[1]}+{offset}, %eax')
                        else:
                            self.em.emit(f'    movl ${base[1]}, %eax')
                        return
            if isinstance(base, tuple) and base[0] in ('id', 'index'):
                self.gen_address(base, fn)
            else:
                self.gen_expr(base, fn)
            if self.is_const_expr(idx):
                stride = self.index_stride(base, fn)
                offset = self.eval_const(idx) * stride
                if isinstance(base, tuple) and base[0] == 'id':
                    info = self.lookup(fn, base[1])
                    if info.get('kind') == 'local' and info.get('type') == 'array':
                        if offset:
                            self.em.emit(f'    subl ${offset}, %eax')
                        return
                    if info.get('kind') == 'global' and info.get('type') == 'array':
                        if offset:
                            self.em.emit(f'    addl ${offset}, %eax')
                        return
                if offset:
                    self.em.emit(f'    addl ${offset}, %eax')
                return
            self.em.emit('    movl %eax, %ebx')
            self.gen_expr(idx, fn)
            stride = self.index_stride(base, fn)
            self.em.emit(f'    imull ${stride}, %eax')
            if isinstance(base, tuple) and base[0] == 'id':
                info = self.lookup(fn, base[1])
                if info.get('kind') == 'local' and info.get('type') == 'array':
                    self.em.emit('    subl %eax, %ebx')
                    self.em.emit('    movl %ebx, %eax')
                    return
            self.em.emit('    addl %eax, %ebx')
            self.em.emit('    movl %ebx, %eax')
            return
        if tag == 'deref':
            self.gen_expr(node[1], fn)
            return
        raise NotImplementedError(f'Unsupported lvalue {node}')

    def resolve_array_info(self, node, fn):
        if isinstance(node, tuple):
            if node[0] == 'id':
                info = self.lookup(fn, node[1])
                dims = info.get('dims', [])
                return node[1], list(dims), 0
            if node[0] == 'index':
                res = self.resolve_array_info(node[1], fn)
                if res is None:
                    return None
                name, dims, level = res
                return name, dims, level + 1
        return None

    def index_stride(self, base, fn):
        info = self.resolve_array_info(base, fn)
        if info is None:
            return 4
        _, dims, level = info
        start = level + 1 if dims else 0
        remaining = dims[start:] if start < len(dims) else []
        stride = 4
        for d in remaining:
            stride *= d if d else 1
        return stride

    def index_direct_operand(self, node, fn):
        res = self.const_index_offset(node, fn)
        if res is None:
            return None
        name, info, offset = res
        kind = info.get('kind')
        if kind == 'local':
            adj = info['offset'] - offset
            return f'{adj}(%ebp)'
        if kind == 'global':
            if offset:
                return f'{name}+{offset}'
            return name
        return None

    def const_index_offset(self, node, fn):
        total = 0
        cur = node
        while isinstance(cur, tuple) and cur and cur[0] == 'index':
            base, idx = cur[1], cur[2]
            if not self.is_const_expr(idx):
                return None
            stride = self.index_stride(base, fn)
            total += self.eval_const(idx) * stride
            cur = base
        if isinstance(cur, tuple) and cur and cur[0] == 'id':
            info = self.lookup(fn, cur[1])
            if info.get('type') == 'array':
                return cur[1], info, total
        return None

    def gen_expr(self, node, fn, want_result=True):
        """Generate code that leaves result in %eax."""
        if node is None:
            return
        tag = node[0] if isinstance(node, tuple) else None
        if tag == 'const':
            self.em.emit(f'    movl ${node[1]}, %eax')
            return
        if tag == 'id':
            info = self.lookup(fn, node[1])
            kind = info.get('kind')
            typ = info.get('type')
            if kind == 'local':
                if typ == 'array':
                    self.em.emit(f'    leal {info["offset"]}(%ebp), %eax')
                else:
                    self.em.emit(f'    movl {info["offset"]}(%ebp), %eax')
            elif kind == 'param':
                self.em.emit(f'    movl {info["offset"]}(%ebp), %eax')
            else:
                if typ == 'array':
                    self.em.emit(f'    movl ${node[1]}, %eax')
                else:
                    self.em.emit(f'    movl {node[1]}, %eax')
            return
        if tag == 'assign_l':
            lv, rv = node[1], node[2]
            self.gen_expr(rv, fn)
            self.em.emit('    movl %eax, %edx')
            if isinstance(lv, tuple) and lv[0] == 'id':
                info = self.lookup(fn, lv[1])
                if info['kind'] in ('local', 'param'):
                    self.em.emit(f'    movl %edx, {info["offset"]}(%ebp)')
                else:
                    self.em.emit(f'    movl %eax, {lv[1]}')
                return
            if isinstance(lv, tuple) and lv[0] == 'deref':
                ptr_operand = self.simple_operand(lv[1], fn)
                if ptr_operand is not None:
                    self.gen_expr(rv, fn)
                    self.em.emit(f'    movl {ptr_operand}, %edx')
                    self.em.emit('    movl %eax, (%edx)')
                    return
            # general lvalue: compute address first, then value
            self.gen_address(lv, fn)
            self.em.emit('    movl %eax, %edx')
            self.gen_expr(rv, fn)
            self.em.emit('    movl %eax, (%edx)')
            return
        if tag == 'neg':
            self.gen_expr(node[1], fn)
            self.em.emit('    negl %eax')
            return
        if tag == 'addr':
            self.gen_address(node[1], fn)
            return
        if tag == 'deref':
            self.gen_address(node, fn)
            self.em.emit('    movl (%eax), %eax')
            return
        if tag == 'index':
            direct = self.index_direct_operand(node, fn)
            if direct is not None:
                self.em.emit(f'    movl {direct}, %eax')
            else:
                self.gen_address(node, fn)
                self.em.emit('    movl (%eax), %eax')
            return
        if tag == 'string':
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
        if tag in ('==', '!=', '<', '>', '<=', '>='):
            left, right = node[1], node[2]
            self.gen_expr(left, fn)
            self.em.emit('    pushl %eax')
            self.gen_expr(right, fn)
            self.em.emit('    movl %eax, %ebx')
            self.em.emit('    popl %eax')
            self.em.emit('    cmpl %ebx, %eax')
            set_instr = {
                '==': 'sete',
                '!=': 'setne',
                '<': 'setl',
                '>': 'setg',
                '<=': 'setle',
                '>=': 'setge',
            }[tag]
            self.em.emit(f'    {set_instr} %al')
            self.em.emit('    movzbl %al, %eax')
            return
        if tag == 'and':
            lbl_false = self.fresh_label('and_false')
            lbl_end = self.fresh_label('and_end')
            self.gen_expr(node[1], fn)
            self.em.emit('    cmpl $0, %eax')
            self.em.emit(f'    je {lbl_false}')
            self.gen_expr(node[2], fn)
            self.em.emit('    cmpl $0, %eax')
            self.em.emit(f'    je {lbl_false}')
            self.em.emit('    movl $1, %eax')
            self.em.emit(f'    jmp {lbl_end}')
            self.em.emit(f'{lbl_false}:')
            self.em.emit('    movl $0, %eax')
            self.em.emit(f'{lbl_end}:')
            return
        if tag == 'or':
            lbl_true = self.fresh_label('or_true')
            lbl_end = self.fresh_label('or_end')
            self.gen_expr(node[1], fn)
            self.em.emit('    cmpl $0, %eax')
            self.em.emit(f'    jne {lbl_true}')
            self.gen_expr(node[2], fn)
            self.em.emit('    cmpl $0, %eax')
            self.em.emit(f'    jne {lbl_true}')
            self.em.emit('    movl $0, %eax')
            self.em.emit(f'    jmp {lbl_end}')
            self.em.emit(f'{lbl_true}:')
            self.em.emit('    movl $1, %eax')
            self.em.emit(f'{lbl_end}:')
            return
        if tag == 'not':
            self.gen_expr(node[1], fn)
            self.em.emit('    cmpl $0, %eax')
            lbl_true = self.fresh_label('not_true')
            lbl_end = self.fresh_label('not_end')
            self.em.emit(f'    je {lbl_true}')
            self.em.emit('    movl $0, %eax')
            self.em.emit(f'    jmp {lbl_end}')
            self.em.emit(f'{lbl_true}:')
            self.em.emit('    movl $1, %eax')
            self.em.emit(f'{lbl_end}:')
            return
        if tag in ('+', '-', '*', '/'):
            left, right = node[1], node[2]
            if tag in ('+', '-'):
                stride, pointer_side = self.pointer_arith_info(tag, left, right, fn)
                if stride is not None:
                    if pointer_side == 'left':
                        pointer_expr, other = left, right
                    else:
                        pointer_expr, other = right, left
                    sign = self.pointer_offset_sign(pointer_expr, fn)
                    if self.is_const_expr(other):
                        raw_offset = self.eval_const(other) * stride
                        if isinstance(pointer_expr, tuple) and pointer_expr[0] == 'id':
                            info = self.lookup(fn, pointer_expr[1])
                            kind = info.get('kind')
                            if info.get('type') == 'array':
                                if kind == 'local':
                                    adj = info['offset'] - raw_offset
                                    self.em.emit(f'    leal {adj}(%ebp), %eax')
                                    return
                                if kind == 'global':
                                    if raw_offset:
                                        self.em.emit(f'    movl ${pointer_expr[1]}+{raw_offset}, %eax')
                                    else:
                                        self.em.emit(f'    movl ${pointer_expr[1]}, %eax')
                                    return
                        offset = raw_offset * sign
                        self.gen_expr(pointer_expr, fn)
                        if offset:
                            if offset > 0:
                                op = 'addl'
                                val = offset
                            else:
                                op = 'subl'
                                val = -offset
                            if tag == '+' or pointer_side == 'right':
                                self.em.emit(f'    {op} ${val}, %eax')
                            else:
                                rev = 'subl' if op == 'addl' else 'addl'
                                self.em.emit(f'    {rev} ${val}, %eax')
                        return
                    if pointer_side == 'left':
                        self.gen_expr(left, fn)
                        self.em.emit('    pushl %eax')
                        self.gen_expr(right, fn)
                    else:
                        self.gen_expr(right, fn)
                        self.em.emit('    pushl %eax')
                        self.gen_expr(left, fn)
                    if stride != 1:
                        self.em.emit(f'    imull ${stride}, %eax')
                    self.em.emit('    movl %eax, %ebx')
                    self.em.emit('    popl %eax')
                    if pointer_side == 'left':
                        if tag == '+':
                            if sign > 0:
                                self.em.emit('    addl %ebx, %eax')
                            else:
                                self.em.emit('    subl %ebx, %eax')
                        else:
                            if sign > 0:
                                self.em.emit('    subl %ebx, %eax')
                            else:
                                self.em.emit('    addl %ebx, %eax')
                    else:
                        if sign > 0:
                            self.em.emit('    addl %ebx, %eax')
                        else:
                            self.em.emit('    subl %ebx, %eax')
                    return
            rhs_operand = self.simple_operand(right, fn)
            if rhs_operand is not None:
                self.gen_expr(left, fn)
                if tag == '+':
                    self.em.emit(f'    addl {rhs_operand}, %eax')
                elif tag == '-':
                    self.em.emit(f'    subl {rhs_operand}, %eax')
                elif tag == '*':
                    self.em.emit(f'    imull {rhs_operand}, %eax')
                else:
                    self.em.emit(f'    movl {rhs_operand}, %ebx')
                    self.em.emit('    cdq')
                    self.em.emit('    idivl %ebx')
                return
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
            else:
                self.em.emit('    cdq')
                self.em.emit('    idivl %ebx')
            return
        # basic fallback

    def push_arg(self, node, fn):
        """Push the value of an expression node onto the stack."""
        if isinstance(node, tuple):
            tag = node[0]
            if tag == 'const':
                self.em.emit(f'    pushl ${node[1]}')
                return
            if tag == 'id':
                info = self.lookup(fn, node[1])
                kind = info.get('kind')
                typ = info.get('type')
                if kind == 'local':
                    if typ == 'array':
                        self.em.emit(f'    leal {info["offset"]}(%ebp), %eax')
                        self.em.emit('    pushl %eax')
                    else:
                        self.em.emit(f'    pushl {info["offset"]}(%ebp)')
                elif kind == 'param':
                    self.em.emit(f'    pushl {info["offset"]}(%ebp)')
                else:
                    if typ == 'array':
                        self.em.emit(f'    pushl ${node[1]}')
                    else:
                        self.em.emit(f'    pushl {node[1]}')
                return
            if tag == 'addr':
                self.gen_address(node[1], fn)
                self.em.emit('    pushl %eax')
                return
            if tag == 'index':
                direct = self.index_direct_operand(node, fn)
                if direct is not None:
                    self.em.emit(f'    pushl {direct}')
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
        if fn is not None:
            if name in fn['locals']:
                info = dict(fn['locals'][name])
                info['kind'] = 'local'
                return info
            if name in fn['params']:
                info = dict(fn['params'][name])
                info['kind'] = 'param'
                return info
        info = self.globals.get(name, {}).copy()
        info.setdefault('kind', 'global')
        return info

    def pointer_arith_info(self, op, left, right, fn):
        """Return (stride, side) if pointer arithmetic is needed."""
        if op not in ('+', '-'):
            return (None, None)
        left_ptr = self.is_pointer_value(left, fn)
        right_ptr = self.is_pointer_value(right, fn)
        if left_ptr and not right_ptr:
            return (self.pointer_stride(left, fn), 'left')
        if right_ptr and not left_ptr and op == '+':
            return (self.pointer_stride(right, fn), 'right')
        return (None, None)

    def pointer_offset_sign(self, pointer_expr, fn):
        if isinstance(pointer_expr, tuple) and pointer_expr[0] == 'id':
            info = self.lookup(fn, pointer_expr[1])
            if info.get('kind') == 'local' and info.get('type') == 'array':
                return -1
        return 1

    def is_pointer_value(self, node, fn):
        if not isinstance(node, tuple):
            return False
        tag = node[0]
        if tag == 'id':
            info = self.lookup(fn, node[1])
            return info.get('type') in ('ptr', 'array')
        if tag == 'addr':
            return True
        if tag == 'assign_l':
            return self.is_pointer_value(node[2], fn)
        return False

    def pointer_stride(self, node, fn):
        info = self.resolve_array_info(node, fn)
        if info is None:
            return 4
        _, dims, level = info
        if not dims:
            return 4
        start = level + 1 if level + 1 <= len(dims) else len(dims)
        remaining = dims[start:]
        stride = 4
        for d in remaining:
            stride *= d if d else 1
        return stride

    def simple_operand(self, node, fn):
        """Return an AT&T operand string for simple expressions."""
        if not isinstance(node, tuple):
            return None
        tag = node[0]
        if tag == 'const':
            return f'${node[1]}'
        if tag == 'id':
            info = self.lookup(fn, node[1])
            kind = info.get('kind')
            typ = info.get('type')
            if typ == 'array':
                return None
            if kind == 'local':
                return f'{info["offset"]}(%ebp)'
            if kind == 'param':
                return f'{info["offset"]}(%ebp)'
            return node[1]
        if tag == 'index':
            return self.index_direct_operand(node, fn)
        return None

    def address_literal(self, node, fn=None):
        if not isinstance(node, tuple):
            return None
        tag = node[0]
        if tag == 'string':
            lbl = self.em.add_string(node[1])
            return lbl
        if tag == 'id':
            info = self.lookup(fn, node[1])
            if info.get('type') == 'array':
                return node[1]
            return None
        if tag == 'addr':
            target = node[1]
            if isinstance(target, tuple) and target[0] == 'id':
                return target[1]
            res = self.const_index_offset(target, fn)
            if res is not None:
                name, info, offset = res
                if info.get('kind') == 'global':
                    if offset:
                        return f'{name}+{offset}'
                    return name
        if tag == 'index':
            res = self.const_index_offset(node, fn)
            if res is not None:
                name, info, offset = res
                if info.get('kind') == 'global':
                    if offset:
                        return f'{name}+{offset}'
                    return name
        return None

    def emit_conditional_branch(self, jump_instr, true_lbl, false_lbl):
        self.em.emit(f'    {jump_instr} {true_lbl}')
        self.em.emit(f'    jmp {false_lbl}')

    def gen_condition(self, node, fn, true_lbl, false_lbl):
        if node is None:
            self.em.emit(f'    jmp {false_lbl}')
            return
        tag = node[0] if isinstance(node, tuple) else None
        if tag == 'or':
            rhs_lbl = self.fresh_label('or_rhs')
            self.gen_condition(node[1], fn, true_lbl, rhs_lbl)
            self.em.label(rhs_lbl)
            self.gen_condition(node[2], fn, true_lbl, false_lbl)
            return
        if tag == 'and':
            rhs_lbl = self.fresh_label('and_rhs')
            self.gen_condition(node[1], fn, rhs_lbl, false_lbl)
            self.em.label(rhs_lbl)
            self.gen_condition(node[2], fn, true_lbl, false_lbl)
            return
        if tag == 'not':
            self.gen_condition(node[1], fn, false_lbl, true_lbl)
            return
        if tag in ('==', '!=', '<', '>', '<=', '>='):
            left, right = node[1], node[2]
            self.gen_expr(left, fn)
            self.em.emit('    pushl %eax')
            self.gen_expr(right, fn)
            self.em.emit('    movl %eax, %ebx')
            self.em.emit('    popl %eax')
            self.em.emit('    cmpl %ebx, %eax')
            jump_map = {
                '==': 'je',
                '!=': 'jne',
                '<': 'jl',
                '<=': 'jle',
                '>': 'jg',
                '>=': 'jge',
            }
            self.emit_conditional_branch(jump_map[tag], true_lbl, false_lbl)
            return
        self.gen_expr(node, fn)
        self.em.emit('    cmpl $0, %eax')
        self.emit_conditional_branch('jne', true_lbl, false_lbl)

    # ---- codegen for statements ----
    def gen_stmt(self, node, fn):
        if node is None:
            return
        tag = node[0]
        if tag == 'exprstmt':
            self.gen_expr(node[1], fn, want_result=False)
            return
        if tag == 'assign_l':
            self.gen_expr(node, fn, want_result=False)
            return
        if tag == 'if':
            cond, then_s, else_s = node[1], node[2], node[3]
            if else_s is not None:
                lbl_then = self.fresh_label('if_then')
                lbl_else = self.fresh_label('if_else')
                lbl_end = self.fresh_label('if_end')
                self.gen_condition(cond, fn, lbl_then, lbl_else)
                self.em.label(lbl_then)
                self.gen_stmt(then_s, fn)
                self.em.emit(f'    jmp {lbl_end}')
                self.em.label(lbl_else)
                self.gen_stmt(else_s, fn)
                self.em.label(lbl_end)
            else:
                lbl_then = self.fresh_label('if_then')
                lbl_end = self.fresh_label('if_end')
                self.gen_condition(cond, fn, lbl_then, lbl_end)
                self.em.label(lbl_then)
                self.gen_stmt(then_s, fn)
                self.em.label(lbl_end)
            return
        if tag == 'block':
            for item in node[1]:
                self.gen_stmt(item, fn)
            return
        if tag == 'decls':
            for decl in node[1]:
                kind = decl[0]
                name = decl[1]
                init = decl[3] if kind == 'array' else decl[2]
                info = fn['locals'].get(name)
                if info is None:
                    continue
                if kind in ('var', 'ptr'):
                    if init is not None:
                        self.gen_expr(init, fn)
                        self.em.emit(f'    movl %eax, {info["offset"]}(%ebp)')
                elif kind == 'array' and init is not None and isinstance(init, tuple) and init[0] == 'initlist':
                    base = info['offset']
                    for idx, expr in enumerate(init[1]):
                        self.gen_expr(expr, fn)
                        self.em.emit(f'    movl %eax, {base + 4*idx}(%ebp)')
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

    def array_elems(self, dims, init):
        if not dims:
            return 1
        resolved = []
        for d in dims:
            resolved.append(d if d is not None else 0)
        if resolved[0] == 0 and isinstance(init, tuple) and init[0] == 'initlist':
            resolved[0] = len(init[1])
        total = 1
        for d in resolved:
            total *= d if d else 1
        return max(total, 1)

    def count_locals(self, body):
        locals_info = []

        def walk(items):
            for it in items:
                if isinstance(it, tuple) and it and it[0] == 'decls':
                    for d in it[1]:
                        kind = d[0]
                        name = d[1]
                        if kind == 'var':
                            locals_info.append({'name': name, 'size': 4, 'type': 'var'})
                        elif kind == 'ptr':
                            locals_info.append({'name': name, 'size': 4, 'type': 'ptr'})
                        elif kind == 'array':
                            dims = list(d[2])
                            init = d[3]
                            elems = self.array_elems(dims, init)
                            locals_info.append({'name': name, 'size': 4 * elems, 'type': 'array', 'dims': dims})
                elif isinstance(it, list):
                    walk(it)
                elif isinstance(it, tuple) and it and it[0] == 'block':
                    walk(it[1])

        walk(body)
        offsets = {}
        off = 0
        for entry in locals_info:
            off -= entry['size']
            info = {'offset': off, 'type': entry['type'], 'size': entry['size']}
            if 'dims' in entry:
                info['dims'] = entry['dims']
            offsets[entry['name']] = info
        return -off, offsets

    def build_params(self, params):
        m = {}
        cur = 8
        for p in params:
            if p[0] == 'param':
                m[p[1]] = {'offset': cur, 'type': 'int'}
                cur += 4
            elif p[0] == 'paramptr':
                m[p[1]] = {'offset': cur, 'type': 'ptr'}
                cur += 4
        return m

    def gen_function(self, rettype, name, params, body):
        fn = {
            'name': name,
            'rettype': rettype,
            'params': self.build_params(params),
            'locals': {},
        }
        stack_size, locals_map = self.count_locals(body)
        fn['locals'] = locals_map

        # prologue
        self.em.emit(f'.globl {name}')
        self.em.emit(f'.type {name}, @function')
        self.em.label(name)
        self.em.emit('    pushl %ebp')
        self.em.emit('    movl %esp, %ebp')
        if stack_size > 0:
            self.em.emit(f'    subl ${stack_size}, %esp')

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
            name = decl[1]
            if kind in ('var', 'ptr'):
                init = decl[2]
                literal = self.address_literal(init) if init is not None else None
                if literal is not None:
                    self.em.emit_data(f'{name}: .long {literal}')
                elif init is None:
                    self.em.emit_data(f'{name}: .long 0')
                elif isinstance(init, tuple) and self.is_const_expr(init):
                    val = self.eval_const(init)
                    self.em.emit_data(f'{name}: .long {val}')
                else:
                    self.em.emit_data(f'{name}: .long 0')
                self.globals[name] = {'type': kind, 'size': 4}
            elif kind == 'array':
                dims = list(decl[2])
                init = decl[3]
                elems = self.array_elems(dims, init)
                self.globals[name] = {'type': 'array', 'dims': dims, 'size': 4 * elems}
                if init is not None and isinstance(init, tuple) and init[0] == 'initlist':
                    self.em.emit_data(f'{name}:')
                    for expr in init[1]:
                        if isinstance(expr, tuple) and self.is_const_expr(expr):
                            val = self.eval_const(expr)
                        elif not isinstance(expr, tuple):
                            val = expr
                        else:
                            val = 0
                        self.em.emit_data(f'    .long {val}')
                    remaining = elems - len(init[1])
                    if remaining > 0:
                        self.em.emit_data(f'    .zero {remaining * 4}')
                else:
                    self.em.emit_data(f'{name}: .zero {elems * 4}')

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
