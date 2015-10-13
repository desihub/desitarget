"""
npyquery, Query mini-language for numpy arrays.

We reuse the python expression parser to construct the
AST. This is inspired by TinyDB, and may have been inspired
by other python database implementations as well.

Override Column.apply for customized access to objects (e.g.
    Pandas dataframe or imaginglss's ColumnStore )

Simple manipulation of the AST is allowed. For example, 
:code:`node.assume(a, b)` allows one to replace 'a' with 'b'
in the expression.

Examples
--------
>>> d = dtype([
        ('BlackholeMass', 'f4'), 
        ('Position', ('f4', 3))])

>>> data = numpy.zeros(4, dtype=d)
>>> data['BlackholeMass'][:] = numpy.arange(4)
>>> data['Position'][:] = numpy.arange(12).reshape(3, 1)
>>> query  = (Column('BlackholeMass') > 2.0)
>>> query &= (Column('BlackholeMass') > 4.0)
>>> query &= (Column('Position')[:, 2] > 1.0)
>>> print(query.apply(data))

>>> query = query.assume(Column('BlackholeMass'), 1.0)
>>> print(query.apply(data))

"""
import numpy
from numpy import ndarray
from copy import deepcopy
import operator

try:
    basestring
except NameError:
    basestring = str

class Node(object):
    """ A node in the query expression.

        Parameters
        ----------
        array : ndarray or alike
            applying the query node to the array and returns items
            satisfying the array.

    """
    def __init__(self):
        self.children = []

    def __invert__(self):
        return Expr("~", operator.__invert__, [self])
    def __neg__(self):
        return Expr("-", operator.__neg__, [self])
    def __le__(self, other):
        return Expr("<=", operator.__le__, [self, other])
    def __lt__(self, other):
        return Expr("<", operator.__lt__, [self, other])
    def __eq__(self, other):
        return Expr("==", operator.__eq__, [self, other])
    def __ne__(self, other):
        return Expr("!=", operator.__ne__, [self, other])
    def __gt__(self, other):
        return Expr(">", operator.__gt__, [self, other])
    def __ge__(self, other):
        return Expr(">=", operator.__ge__, [self, other])
    def __and__(self, other):
        return Expr("&", numpy.bitwise_and, [self, other])
    def __or__(self, other):
        return Expr("|", numpy.bitwise_or, [self, other])
    def __xor__(self, other):
        return Expr("^", numpy.bitwise_xor, [self, other])
    def __pow__(self, other):
        return Expr("**", numpy.power, [self, other])
    def __rpow__(self, other):
        return Expr("**", numpy.power, [other, self])
    def __mul__(self, other):
        return Expr("*", numpy.multiply, [self, other])
    def __rmul__(self, other):
        return Expr("*", numpy.multiply, [other, self])
    def __div__(self, other):
        return Expr("/", numpy.divide, [self, other])
    def __rdiv__(self, other):
        return Expr("/", numpy.multiply, [other, self])
    def __add__(self, other):
        return Expr("+", numpy.add, [self, other])
    def __radd__(self, other):
        return Expr("+", numpy.add, [other, self])
    def __sub__(self, other):
        return Expr("-", numpy.subtract, [self, other])
    def __rsub__(self, other):
        return Expr("-", numpy.subtract, [other, self])
    def __mod__(self, other):
        return Expr("%", numpy.remainder, [self, other])
    def __rmod__(self, other):
        return Expr("%", numpy.remainder, [other, self])
    def __getitem__(self, index):
        return GetItem(self, index)
    def sin(self):
        return Expr("sin", numpy.sin, [self])
    def cos(self):
        return Expr("cos", numpy.cos, [self])
    def tan(self):
        return Expr("tan", numpy.tan, [self])
    def log(self):
        return Expr("log", numpy.log, [self])
    def log10(self):
        return Expr("log10", numpy.log10, [self])
    def max(self):
        f = lambda x: numpy.max(x, axis=tuple(range(1, len(x.shape))))
        return Expr("max", f, [self])
    def min(self):
        f = lambda x: numpy.min(x, axis=tuple(range(1, len(x.shape))))
        return Expr("min", f, [self])

    @property
    def T(self):
        return Transpose(self)

    def __call__(self, array):
        mask = self.apply(array)
        if isinstance(array, dict):
            d = {}
            for key in array.keys():
                d[key] = array[key][mask]
            return d
        else:
            return array[mask]

    def apply(self, array):
        chunksize=1024 * 128
        if isinstance(array, dict):
            length = len(array[array.keys()[0]])
        else:
            length = len(array)
        result = numpy.empty(length, dtype='?')
        for i in range(0, length, chunksize):
            s = slice(i, i + chunksize)
            v = QueryVisitor(array, s)
            result[s] = v.visit(self)
        return result

    def assume(self, node, literal):
        """ Replace all subexpression 'node' with 'literal'.
        """
        av = AssumptionVisitor(node, literal)
        rt = deepcopy(self)
        av.visit(rt)
        return rt

    def __deepcopy__(self, memo):
        return NotImplemented

    @property
    def names(self):
        """ returns a list of column names used in this node.
            
            This function is recursive
        """
        r = [] 
        for c in self.children:
            r.extend(c.names)
        return list(set(r))

    def equals(self, other):
        return repr(self) == repr(other)

def repr_slice(s):
    if not isinstance(s, slice):
        return repr(s)
    if s.start is None and s.stop is None and s.step is None:
        return ':'
    start = "" if s.start is None else repr(s.start)
    stop = "" if s.stop is None else repr(s.stop)
    if s.step is None:
        return "%s:%s" % (start, stop)
    else:
        step = repr(s.step)
    return "%s:%s:%s" % (start, stop, step)

class GetItem(Node):
    """ 
        Represents accesing an item in a vector column. 
        
        Astonishing that pandas can't do this.
    """
    def __init__(self, obj, index):
        self.children = [obj]
        self.index = index

    @property
    def obj(self):
        return self.children[0]

    def __repr__(self):
        if isinstance(self.index, tuple):
            rslice = ','.join([repr_slice(o) for o in self.index])
        else:
            rslice = repr_slice(self.index)
        return "%s[%s]" % (repr(self.obj), rslice)

    def __deepcopy__(self, memo):
        return GetItem(deepcopy(self.obj, memo), self.index)

class Literal(Node):
    """ 
        Represents a literal constant.
    """
    def __init__(self, value):
        self.value = value
        self.children = []

    def __repr__(self):
        return repr(self.value)

    def __deepcopy__(self, memo):
        return Literal(self.value)

class Column(Node):
    """ 
        Represents accessing a column from the data array 
    """
    def __init__(self, name):
        self.name = name
        self.children = []

    def __repr__(self):
        return "%s" % self.name

    @property
    def names(self):
        return [self.name]

    def __deepcopy__(self, memo):
        return Column(self.name)

    def apply(self, array, s):
        return array[self.name][s]

class Transpose(Node):
    """
        Represents a transpose. (.T attribute)
    """
    def __init__(self, node):
        self.children = [node]

    @property
    def obj(self):
        return self.children[0]

    def __repr__(self):
        return "%s.T" % str(self.obj)
    def __deepcopy__(self, memo):
        return Transpose(deepcopy(self.obj, memo))

class Expr(Node):
    """ 
        Represents an expression. 

        e.g. comparing a column with a number.

        An expression can have multiple operands.

        >>> for o in expr.operands:
        >>>   .....
        >>> print expr[0], expr[1]

    """
    def __init__(self, operator, function, operands):
        self.operator = operator
        self.function = function
        operands = [
            a if isinstance(a, Node)
            else Literal(a)
            for a in operands
        ]

        self.children = self.flatten(operands)

    @property
    def operands(self):
        return self.children

    def is_associative(self):
        """ Is the operator associative?

            We test this by see if the ufunc has a value identity property.
        """
        if not isinstance(self.function, numpy.ufunc):
            return False
        if self.function.identity is not None:
            return True
        return False

    def __iter__(self):
        return iter(self.operands)

    def __getitem__(self, index):
        return self.operands[index]

    def flatten(self, operands):
        """ Flattens operands of associative operators.

            e.g. (a + b) + (c + d) becomes a + b + c + d

        """
        if not self.is_associative(): return operands
        o = []
        for a in operands:
            if not isinstance(a, Expr):
                o.append(a)
                continue
            if a.function != self.function:
                o.append(a)
                continue
            else:
                o.extend(a.operands)
        return o

    def __repr__(self):
        if self.operator == 'max': 
            return "max(%s)" % (', ').join([repr(a) for a in self.operands])
        if self.operator == 'min': 
            return "min(%s)" % (', ').join([repr(a) for a in self.operands])
        if len(self.operands) >= 2:
            return "(%s)" % (' ' + self.operator + ' ').join([repr(a) for a in self.operands])
        elif len(self.operands) == 1:
            [a] = self.operands
            return "(%s %s)" % (self.operator, repr(a))
        else:
            raise ValueError

    def __deepcopy__(self, memo):
        return Expr(self.operator, self.function, 
                    [deepcopy(c, memo) for c in self.operands])

def Max(*args):
    f = lambda *args: numpy.max(args, axis=0)
    return Expr('max', f, args)

def Min(*args):
    f = lambda *args: numpy.min(args, axis=0)
    return Expr('min', f, args)

class Visitor(object):
    def __init__(self):
        self.m = {}

        self.reg('visit_getitem', GetItem)
        self.reg('visit_literal', Literal)
        self.reg('visit_column', Column)
        self.reg('visit_transpose', Transpose)
        self.reg('visit_expr', Expr)
        self.reg('visit_node', Node)

    def reg(self, attr, type):
        if hasattr(self, attr):
            self.m[type] = getattr(self, attr)

    def visit(self, node):
        t = type(node)
        while t not in self.m:
            t = t.__base__
        return self.m[t](node)

    def visit_node(self, node):
        return NotImplemented

class AssumptionVisitor(Visitor):
    def __init__(self, node, literal):
        Visitor.__init__(self)
        self.node = node
        if not isinstance(literal, Node):
            literal = Literal(literal)
        self.literal = literal

    def visit_node(self, node):
        newchildren = []
        for c in node.children:
            if c.equals(self.node):
                c = self.literal
            else:
                self.visit(c)
            newchildren.append(c)
        node.children = newchildren

class QueryVisitor(Visitor):
    def __init__(self, array, s):
        Visitor.__init__(self)
        self.array = array
        self.s = s

    def visit_getitem(self, node):
        return self.visit(node.obj)[node.index]

    def visit_literal(self, node):
        if isinstance(node.value, basestring):
            v = numpy.array(node.value, 'S')
        else:
            v = node.value
        return v

    def visit_column(self, node):
        return node.apply(self.array, self.s)

    def visit_transpose(self, node):
        return self.visit(node.obj).T

    def visit_expr(self, node):
        ops = [self.visit(a) for a in node.operands]
        if node.is_associative():
            # do not use ufunc reduction because ops can 
            # be of different shape
            r = ops[0]
            for a, s in zip(ops[1:], node.operands[1:]):
                r = node.function(r, a)
        else:
            r = node.function(*ops)
        return r

def test():    
    d = numpy.dtype([
        ('BlackholeMass', 'f4'), 
        ('PhaseOfMoon', 'f4'), 
        ('Name', 'S4'), 
        ('Position', ('f4', 3)),
])

    data = numpy.zeros(5, dtype=d)
    data['BlackholeMass'][:] = numpy.arange(len(data))
    data['PhaseOfMoon'][:] = numpy.linspace(0, 1, len(data), endpoint=True)
    data['Position'][:] = numpy.arange(data['Position'].size).reshape(len(data), -1)
    data['Name'][0] = 'N1'
    data['Name'][1] = 'N2'
    data['Name'][2] = 'N3'
    data['Name'][3] = 'N4'

    query1 = (Column('BlackholeMass') > 0.0)
    assert query1.apply(data).sum() == 4
    query2 = (Column('Name') == 'N1')
    assert query2.apply(data).sum() == 1
    query3 = (Column('BlackholeMass') < 5.0)
    assert query3.apply(data).sum() == 5
    query4 = (Column('Position')[:, 2] > 0.0) | (Column('Position')[:, 1] < 0.0)
    assert query4.apply(data).sum() == 5
    query5 = (numpy.sin(Column('PhaseOfMoon') * (2 * numpy.pi)) < 0.1)
    assert query5.apply(data).sum() == 4
    query6 = Max(Column('BlackholeMass'), Column('PhaseOfMoon'), Column('Position').max()) > 0
    assert query6.apply(data).sum() == 5

    assert query6.equals(deepcopy(query6))
    assert not query6.equals(deepcopy(query5))
    query7 = query4.assume(Column('Position')[:, 1], 1.0) \
                .assume(Column('Position')[:, 2], -1.0)
    assert query7.apply(data).sum() == 0

if __name__ == '__main__':
    test()
