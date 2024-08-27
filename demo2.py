import functools

class Expr():
    def __add__(self, other): return Add(self, other)
    def __and__(self, other): return And(self, other)
    def __or__(self, other): return Or(self, other)
    def __xor__(self, other): return Xor(self, other)
    def __invert__(self): return Invert(self)
    def rotr(self, amount): return Rotr(self, amount)

class SoloExpr(Expr):
    def child(self): return self.e

class MultiExpr(Expr):
    def __init__(self, *args): self.xs = [*args]
    def children(self): return self.xs
    def __repr__(self): return f'({self.name} {" ".join([repr(x) for x in self.children()])})'

class Invert(SoloExpr):
    op = lambda x: ~x

    def __init__(self, e):
        self.e = e

    def __repr__(self): return f'(not {self.e})'

class Rotr(SoloExpr):
    op = lambda x,n: x

    def __init__(self, e, amount):
        self.e = e
        self.amount = amount

    def __repr__(self): return f'(rotr {self.e} {self.amount})'

class Add(MultiExpr):
    name = '+'
    op = lambda x,y: x+y

class And(MultiExpr):
    name = 'and'
    op = lambda x,y: x&y

class Or(MultiExpr):
    name = 'or'
    op = lambda x,y: x|y

class Xor(MultiExpr):
    name = 'xor'
    op = lambda x,y: x^y

class Var(Expr):
    def __init__(self, name):
        self.name = name

    def __repr__(self): return self.name

class Const(Expr):
    def __init__(self, val):
        self.val = val

    def __repr__(self): return str(self.val)

def compress(e):
    if isinstance(e, (Var, Const)):
        return e
    elif isinstance(e, Invert):
        return Invert(compress(e.e))
    elif isinstance(e, Rotr):
        return Rotr(compress(e.e), e.amount)
    elif isinstance(e, MultiExpr):
        T = type(e)
        xs = [compress(x) for x in e.children()]
        ys = []
        for x in xs:
            if type(x) == T:
                ys += x.children()
            else:
                ys.append(x)

        return T(*ys)
    else:
        raise Exception('What is this thing', e)

def partition(xs, pred):
    yes, no = [], []
    for x in xs:
        if pred(x):
            yes.append(x)
        else:
            no.append(x)

    return yes, no

def constfold(e):
    if isinstance(e, (Var, Const)):
        return e
    elif isinstance(e, Invert):
        x = constfold(e.e)
        if isinstance(x, Const):
            return Const(Invert.op(x.val))
        return Invert(x)
    elif isinstance(e, Rotr):
        x = constfold(e.e)
        if isinstance(x, Const):
            return Const(Rotr.op(x.val, e.amount))
        return Rotr(x, e.amount)
    elif isinstance(e, MultiExpr):
        xs = [constfold(x) for x in e.children()]
        consts, vs = partition(xs, lambda x: isinstance(x, Const))

        T = type(e)

        if not consts:
            return T(*vs)

        c = Const(functools.reduce(T.op, [x.val for x in consts]))

        if not vs:
            return c

        vs.append(c)

        return T(*vs)
    else:
        raise Exception('What is this thing')

SHA256_SEED = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ]

ROUND_CONST = [
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
    ]

def intialize_m_from_payload(p, n):
    block = [0] * 64

    for i in range(len(p)):
        block[i] = ord(p[i])

    for j in range(len(p), n):
        block[j] = ord('x')

    block[n] = 0x80

    n *= 8
    block[62] = n // 256
    block[63] = n % 256

    m = []
    for i in range(16):
        x = block[i*4] << 24 | block[i*4+1] << 16 | block[i*4+2] << 8 | block[i*4+3]
        m.append(Const(x))

    return m

# Cost: 3 ROTR, 2 XOR
def ep0(x): return x.rotr(2) ^ x.rotr(13) ^ x.rotr(22)

# Cost: 3 ROTR, 2 XOR
def ep1(x): return x.rotr(6) ^ x.rotr(11) ^ x.rotr(25)

# Cost: 2 AND, 1 XOR, 1 NOT
def ch(x,y,z): return (x & y) ^ (~x & z)

# Cost: 3 AND, 2 XOR
def maj(x,y,z): return (x & y) ^ (x & z) ^ (y & z)

variable_index = list(range(6, 13))

for vi in variable_index:
    l = 4 * (vi + 1)
    a,b,c,d,e,f,g,h = [Const(x) for x in SHA256_SEED]
    k = [Const(x) for x in ROUND_CONST]
    m = intialize_m_from_payload("toteload/davidbos+dot+me/", l)
    m[vi-1], m[vi] = Var('m0'), Var('m1')

    print(f'Length: {l}')
    print(m)

    for i in range(16):
        # Cost: 4 ADD, 3 ROTR, 3 XOR, 2 AND, 1 NOT
        t1 = constfold(compress(h + ep1(e) + ch(e,f,g) + k[i] + m[i]))

        # Cost: 3 ROTR, 4 XOR, 2 AND
        t2 = constfold(compress((ep0(a) + maj(a,b,c))))

        h = g
        g = f
        f = e
        e = constfold(compress(d + t1))
        d = c
        c = b
        b = a
        a = constfold(compress(t1 + t2))

        regs = a,b,c,d,e,f,g,h

        print(f'ROUND {i}')
        for j, r in enumerate(regs):
            if not isinstance(r, Const):
                print('abcdefgh'[j] + ' is no longer const in round ' + str(i))

        if not any([isinstance(r, Const) for r in regs]):
            break
