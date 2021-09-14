#・エラトステネス　bool
n=100000
p=[0]*2+[1]*(n-1)
for i in range(2,n+1):
  if p[i]:
    for j in range(i*2,n+1,i):
      p[j]=0
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
n = 100000
prime = [False] * 2 + [True] * (n-2)
for i in range(2,n):
  if prime[i]:
    for j in range(i*2,n,i):
      prime[j] = False
#リストに変換
s = [i for i in range(2,n) if prime[i]]

#・部分文字列全列挙
def getCharNGram(n,s):
    charGram = [''.join(s[i:i+n]) for i in range(len(s)-n+1)]
    return charGram
for i in range(1,4):
    print(getCharNGram(i,"abc"))

#・素因数分解(下に単体用あり)
def fact(n):
  arr=[]
  temp=n
  for i in range(2,int(-(-n**0.5//1))+1):
    if temp%i==0:
      cnt=0
      while temp%i==0:
        cnt+=1
        temp//=i
      arr.append([i,cnt])
  if temp!=1:
    arr.append([temp,1])
  if arr==[]:
    arr.append([n,1])
  return arr

#・高速素因数分解（osa_k）
ln=10**6+10
sieve=[i for i in range(ln)]
p=2
while p**2<=ln:
  if sieve[p]==p:
    for q in range(p*2,ln,p):
      if sieve[q]==q:
        sieve[q]=p
  p+=1
import collections
def primef(n):
  prime_count=collections.Counter()    
  while n>1:
    prime_count[sieve[n]]+=1
    n//=sieve[n]   
  return prime_count

#・bit全探査
from itertools import product
for bits in product([0, 1], repeat=n):

#・約数全列挙
def make_divisors(n):
  divisors=[]
  for i in range(1,int(n**0.5)+1):
    if n%i==0:
      divisors.append(i)
      if i!=n//i:
        divisors.append(n//i)
  # divisors.sort()
  return divisors

#・グラフの受け取り 隣接リスト表現
ki=[[] for _ in range(n)]
for i in range(n-1):
  a,b=map(int,input().split())
  ki[a-1].append(b-1)
  ki[b-1].append(a-1)


#・にぶたん（ソートしたリストのどこに指定した要素が入るか探索）
import bisect as bi
bi.bisect_right(lst, 3)
bi.bisect_left(lst, 3)

bi.insort_right(lst, 3)
bi.insort_left(lst, 3)
lst.insert(bi.bisect_left(lst, 8), 8)

#・レーベンシュタイン編集距離
#（一文字の挿入、削除、置換で文字列を目的の文字列に変換する最小コスト）
def levenshtein(s1, s2):
    # s1とs2が同じであれば、0であることは明らか
    if s1 == s2:
        return 0

    # 文字列の長さを取得
    s1_len = len(s1)
    s2_len = len(s2)

    # 空文字列からs2への文字挿入回数 = s2の長さ
    if s1_len == 0:
        return s2_len
    # 空文字列からs1への文字挿入回数 = s1の長さ
    if s2_len == 0:
        return s1_len

    # 行列の初期化
    m = [[j for j in range(s2_len+1)] if i==0 else [i if j==0 else 0 for j in range(s2_len+1)] for i in range(s1_len+1)]

    # m[i-1][j]、m[i][j-1]、m[i-1][j-1]を使ってm[i][j]を求める
    for i in range(1, s1_len+1):
        for j in range(1, s2_len+1):
            # m[i-1][j]から求める
            c1 = m[i-1][j] + 1
            # m[i][j-1]から求める
            c2 = m[i][j-1] + 1
            # m[i-1][j-1]から求める
            c3 = m[i-1][j-1] + (0 if s1[i-1]==s2[j-1] else 1)
            # 最小値を採用
            m[i][j] = min(c1, c2, c3)

    # 行列の中身を表示（コメントを外す）
    # print(m)

    # 行列の右下の値がs1とs2のレーベンシュタイン距離
    return m[s1_len][s2_len]

#・最大公約数いっぱい
from fractions import gcd
#3.7以降はfractionsではなくmath
from functools import reduce
#listはソートしたほうがいい
def gcdl(list):
  return reduce(gcd,list)

#・lcm
from fractions import gcd
def lcm(x,y):
  return (x*y)//gcd(x,y)

from functools import reduce
def lcm_list(l):
  return reduce(lcm,l,1)

#・幅優先
from collections import deque
d=deque()
d.append(0)
visited=[False]*n
visited[0]=True
while d:
  g=d.popleft()
  for i in ki[g]:
    if visited[i]:
      continue
    d.append(i)
    visited[i]=True

#・深さ優先探索（ルートの全列挙が必要、文字列の辞書順、状態遷移の分岐）
'''
上のpopleftをpopにするだけ
(visitedにかかるのは、逆走しようとした時だけ)
'''

#・dfs(再帰)
n = int(input()) # 点の総数
path = list() # path[p] : 点pから行ける点のリスト
path.append([])
for _ in range(n):
   tmp = list(map(int, input().split())) # tmp[0]から行ける点はtmp[1]個あり、それはtmp[2:]
   path.append(tmp[2:])

d = [0]*(n+1) # 最初に発見した時刻
f = [0]*(n+1) # 隣接リストを調べ終えた時刻

TIME = 0

def dfs(p, d, f):
   global TIME

   # ここに来訪時の処理を書く
   TIME += 1
   d[p] = TIME # 時刻の記録
   
   for nxt in path[p]: #繋がってる点の内 
       if d[nxt] == 0: # 未探索の(発見時刻が初期値のままの場合)
           dfs(nxt, d, f) # 先に進む
   
   # ここに帰る時の処理を書く
   TIME += 1
   f[p] = TIME # 繋がってる全ての点を探索し終えたらその点でやることは終わり
   
   return     
   
for start in range(1, n+1):
   if d[start] == 0: # 未探索の点があれば
       dfs(start, d, f) # dfs開始

for i in range(1, n+1):
   print(i, d[i], f[i])


#・連続要素数の取得（aabbbcaabb→[a,2],[b,3],[c,1],[a,2],[b,2]）
from itertools import groupby
g=groupby(s)
a=[]
for i,j in g:
  a.append([i,len(list(j))])

#・素因数分解
pf={}
m=341555136
for i in range(2,int(m**0.5)+1):
    while m%i==0:
        pf[i]=pf.get(i,0)+1
        m//=i
if m>1:pf[m]=1
print(pf)

#・決め打ちにぶたん(最小の最大化、ＯＫが左)
def is_ok(i):
  if i<0:
    return True
  if i>=l:
    return False
  return #適切な判断

ok=0
ng=l+1
while ng-ok>1:
  mid=(ok+ng)//2
  if is_ok(mid):
    ok=mid
  else:
    ng=mid
print(ok)

#・最長共通部分列
import numpy as np
def LCS(s,t,rec=False):
  S=np.array(list(s))
  T=np.array(list(t))
  LS,LT=len(S),len(T)
  dp=np.zeros((LS+1,LT+1))
  for n in range(1,LS+1):
    dp[n,1:]=dp[n-1,:-1]+(S[n-1]==T)
    np.maximum(dp[n],dp[n-1],out=dp[n])
    np.maximum.accumulate(dp[n],out=dp[n])
  if rec:
    tmp=[]
    while LS>0 and LT>0:
      if S[LS-1]==T[LT-1]:
        LS,LT=LS-1,LT-1
        tmp.append(S[LS])
      elif dp[LS,LT]==dp[LS-1,LT]:
        LS-=1
      else:
        LT-=1
    return ''.join(reversed(tmp))
  else:
    return dp[LS][LT]

#・C
from math import factorial as f
def c(n,r):
  if n<r:
    return 0
  else:
    return f(n)//(f(n-r)*f(r))

#・Counter
from collections import Counter as co
d=co(list)

#・modとってC（最速）
mod=10**9+7
def modinv(a):
    return pow(a, mod-2, mod)
def comb(n, r):
    r = min(r, n-r)
    res = 1
    for i in range(r):
        res = res * (n - i) * modinv(i+1) % mod
    return res

#・nC0~nCnのリスト（modinv必要）
def combination_list(n, mod=10**9+7):
    lst = [1]
    for i in range(1, n+1):
        lst.append(lst[-1] * (n+1-i) % mod * modinv(i, mod) % mod)
    return lst


#・modのC　構築O(N+logP) nCkがO(1)
mod = 10**9+7
MAX_N = 10**6

fact = [1]
fact_inv = [0]*(MAX_N+4)
for i in range(MAX_N+3):
    fact.append(fact[-1]*(i+1)%mod)

fact_inv[-1] = pow(fact[-1],mod-2,mod)
for i in range(MAX_N+2,-1,-1):
    fact_inv[i] = fact_inv[i+1]*(i+1)%mod

def mod_comb_k(n,k,mod):
    return fact[n] * fact_inv[k] % mod * fact_inv[n-k] %mod

#・modのC（高速）
def pf(x,y,p):
  if y==0: return 1
  if y%2==0:
    d=pf(x,y//2,p)
    return d*d%p
  if y%2==1:
    return (x*pf(x,y-1,p))%p

facl=[1]
for i in range(1,n+2):
  facl.append(facl[i-1]*i%mod)
invl=[1]*(n+2)
for i in range(1,n+2):
  invl[i]=pf(facl[i],mod-2,mod)

def comb(x,k,p):
  if x<0 or k<0 or x<k: return 0
  if x==0 or k==0: return 1
  return (facl[x]*invl[k]*invl[x-k])%mod

#・クラスカル法の導入
class Kruskal_UnionFind():
    # 無向グラフであるという前提に注意
    def __init__(self, N):
        self.edges = []
        self.rank = [0] * N
        self.par = [i for i in range(N)]
        self.counter = [1] * N

    def add(self, u, v, d):
        """
        u = from, v = to, d = cost
        """
        self.edges.append([u, v, d])

    def find(self, x):
        if self.par[x] == x:
            return x
        else:
            self.par[x] = self.find(self.par[x])
            return self.par[x]

    def unite(self, x, y):
        x = self.find(x)
        y = self.find(y)
        if x != y:
            z = self.counter[x] + self.counter[y]
            self.counter[x], self.counter[y] = z, z
        if self.rank[x] < self.rank[y]:
            self.par[x] = y
        else:
            self.par[y] = x
            if self.rank[x] == self.rank[y]:
                self.rank[x] += 1

    def size(self, x):
        x = self.find(x)
        return self.counter[x]

    def same(self, x, y):
        return self.find(x) == self.find(y)

    def Kruskal(self):
        """
        return: 最小全域木のコストの和
        """
        edges = sorted(self.edges, key=lambda x: x[2])  # costでself.edgesをソートする
        res = 0
        for e in edges:
            if not self.same(e[0], e[1]):
                self.unite(e[0], e[1])
                res += e[2]
        return res

#・クラスカル使用
N = int(input())
XY = [[i] + list(map(int, input().split())) for i in range(N)]

graph = Kruskal_UnionFind(N)
XY = sorted(XY, key=lambda x: x[1])
X_costs = [[XY[i-1][0], XY[i][0], abs(XY[i-1][1] - XY[i][1])] for i in range(1, N)]
XY = sorted(XY, key=lambda x: x[2])
Y_costs = [[XY[i-1][0], XY[i][0], abs(XY[i-1][2] - XY[i][2])] for i in range(1, N)]

for i in range(N-1):
    x0, x1, d = X_costs[i]
    graph.add(x0, x1, d)
    y0, y1, d = Y_costs[i]
    graph.add(y0, y1, d)

print(graph.Kruskal())

・座圧
def comp(l,reverse=False):
  zipp={}
  unzipp={}
  for i, xi in enumerate(sorted(set(l),reverse=reverse)):
    zipp[xi]=i
    unzipp[i]=xi
  return zipp, unzipp

#・プリム

AdjacentVertex = collections.namedtuple("AdjacentVertex", "vertex cost")
INF = 2 ** 31 - 1
NO_VERTEX = -1
 
 
# Prim法で頂点0からの最小全域木を求める
def compute_mst_prim(max_v, adj_list):
    # pred[u]は頂点uの「ひとつ前」の頂点を表す
    pred = collections.defaultdict(lambda: NO_VERTEX)
    # uとpred[u]を結ぶ辺の重みがkey[u]に入る
    key = collections.defaultdict(lambda: INF)
    key[0] = 0
    # 二分ヒープを優先度付きキューとして用いる
    pq = [(key[v], v) for v in range(max_v)]
    heapq.heapify(pq)
    # 優先度付きキューに頂点が入っているかを示す配列
    in_pq = array.array("B", (True for _ in range(max_v)))
    while pq:
        _, u = heapq.heappop(pq)
        in_pq[u] = False
        for v, v_cost in adj_list[u]:
            if in_pq[v]:
                weight = v_cost
                if weight < key[v]:
                    pred[v] = u
                    key[v] = weight
                    heapq.heappush(pq, (weight, v))
                    in_pq[v] = True
    return (pred, key)
 
 
def main():
    max_v, max_e = map(int, input().split())
    adjacency_list = collections.defaultdict(set)
    for _ in range(max_e):
        s, t, w = map(int, input().split())
        adjacency_list[s].add(AdjacentVertex(t, w))
        adjacency_list[t].add(AdjacentVertex(s, w))
    (_, key) = compute_mst_prim(max_v, adjacency_list)
    print(sum(key.values()))
 
 
if __name__ == '__main__':
    main()
 
#・unionfind（クラス）
class UnionFind():
  def __init__(self, n):
    self.n = n
    self.parents = [-1] * n

  def find(self, x):
    if self.parents[x] < 0:
      return x
    else:
      self.parents[x] = self.find(self.parents[x])
      return self.parents[x]

  def union(self, x, y):
    x = self.find(x)
    y = self.find(y)

    if x == y:
      return

    if self.parents[x] > self.parents[y]:
      x, y = y, x

    self.parents[x] += self.parents[y]
    self.parents[y] = x

  def size(self, x):
    return -self.parents[self.find(x)]

  def same(self, x, y):
    return self.find(x) == self.find(y)

  def members(self, x):
    root = self.find(x)
    return [i for i in range(self.n) if self.find(i) == root]

  def roots(self):
    return [i for i, x in enumerate(self.parents) if x < 0]

  def group_count(self):
    return len(self.roots())

  def all_group_members(self):
    return {r: self.members(r) for r in self.roots()}

  def __str__(self):
    return '\n'.join('{}: {}'.format(r, self.members(r)) for r in self.roots())

#・unionfind（非クラス）
xs=[tuple(map(int,input().split())) for _ in range(m)]
def root(n):
  if union[n]==n:
    return n
  union[n]=root(union[n])
  return union[n]
union={x:x for x in range(1,n+1)}
for x,y in xs:
  union[root(x)]=root(y) 

#・最長増加部分列
from bisect import bisect_left
def lis(A):
  L = [A[0]]
  for a in A[1:]:
    if a > L[-1]:
            # Lの末尾よりaが大きければ増加部分列を延長できる
      L.append(a)
    else:
            # そうでなければ、「aより小さい最大要素の次」をaにする
            # 該当位置は、二分探索で特定できる
      L[bisect_left(L, a)] = a
  return len(L)

#・最長共通部分列
def lcs(a: str, b: str):
    L = []
    for bk in b:
        bgn_idx = 0  # 検索開始位置
        for i, cur_idx in enumerate(L):
            # ※1
            chr_idx = a.find(bk, bgn_idx) + 1
            if not chr_idx:
                break
            L[i] = min(cur_idx, chr_idx)
            bgn_idx = cur_idx
        else:
            # ※2
            chr_idx = a.find(bk, bgn_idx) + 1
            if chr_idx:
                L.append(chr_idx)
    return len(L)


#・セグ木
class SegmentTree:
    def __init__(self, value=[], N=0, comp=lambda x,y: x<=y, reverse=False):
        M = max(len(value),N)
        N = 2**(len(bin(M))-3)
        if N < M: N *= 2
        self.N = N
        self.node = [0] * (2*N-1)
        for i in range(N):
            self.node[N-1+i] = i
        self.value = [None] * N
        for i, v in enumerate(value):
            self.value[i] = v
        self.comp = lambda x, y: True if y is None else False if x is None else comp(x,y)^reverse
        for i in range(N-2,-1,-1):
            left_i, right_i = self.node[2*i+1], self.node[2*i+2]
            left_v, right_v = self.value[left_i], self.value[right_i]
            self.node[i] = left_i if self.comp(left_v, right_v) else right_i

    def __setitem__(self, n, v):
        self.update(n,v)

    def __getitem__(self, n):
        #getitem,setitemは特殊なやつ、defすると[]でアクセス可能
        return self.at(n)

    def update(self, n, v):
        self.value[n] = v
        i = (self.N-1) + n
        while i > 0:
            i = (i-1)//2
            left_i, right_i = self.node[2*i+1], self.node[2*i+2]
            left_v, right_v = self.value[left_i], self.value[right_i]
            new_i = left_i if self.comp(left_v, right_v) else right_i
            if new_i == self.node[i] and new_i != n:
                break
            else:
                self.node[i] = new_i

    def at(self, n):
        if n is None:
            return None
        else:
            return self.value[n]

    def query(self, l=0, r=-1):
        return self.at(self.query_index(l,r))

    def query_index(self, l=0, r=-1):
        if r < 0: r = self.N
        L = l + self.N; R = r + self.N
        s = None
        while L < R:
            if R & 1:
                R -= 1
                if self.comp(self.at(self.node[R-1]), self.at(s)):
                    s = self.node[R-1]
            if L & 1:
                if self.comp(self.at(self.node[L-1]), self.at(s)):
                    s = self.node[L-1]
                L += 1
            L >>= 1; R >>= 1
        return s

#・セグ木２
def segfunc(x, y):
    return x^y
 
ide_ele = 0
 
class SegTree:
    """
    init(init_val, ide_ele): 配列init_valで初期化 O(N)
    update(k, x): k番目の値をxに更新 O(logN)
    query(l, r): 区間[l, r)をsegfuncしたものを返す O(logN)
    """
    def __init__(self, init_val, segfunc, ide_ele):
        """
        init_val: 配列の初期値
        segfunc: 区間にしたい操作
        ide_ele: 単位元
        n: 要素数
        num: n以上の最小の2のべき乗
        tree: セグメント木(1-index)
        """
        n = len(init_val)
        self.segfunc = segfunc
        self.ide_ele = ide_ele
        self.num = 1 << (n - 1).bit_length()
        self.tree = [ide_ele] * 2 * self.num
        # 配列の値を葉にセット
        for i in range(n):
            self.tree[self.num + i] = init_val[i]
        # 構築していく
        for i in range(self.num - 1, 0, -1):
            self.tree[i] = self.segfunc(self.tree[2 * i], self.tree[2 * i + 1])
 
    def update(self, k, x):
        """
        k番目の値をxに更新
        k: index(0-index)
        x: update value
        """
        k += self.num
        self.tree[k] = x
        while k > 1:
            self.tree[k >> 1] = self.segfunc(self.tree[k], self.tree[k ^ 1])
            k >>= 1
 
    def query(self, l, r):
        """
        [l, r)のsegfuncしたものを得る
        l: index(0-index)
        r: index(0-index)
        """
        res = self.ide_ele
 
        l += self.num
        r += self.num
        while l < r:
            if l & 1:
                res = self.segfunc(res, self.tree[l])
                l += 1
            if r & 1:
                res = self.segfunc(res, self.tree[r - 1])
            l >>= 1
            r >>= 1
        return res
      
n,q=map(int,input().split())
a=list(map(int,input().split()))
seg=SegTree(a,segfunc,ide_ele)
for _ in range(q):
  t,x,y=map(int,input().split())
  if t==1:
    a[x-1]^=y
    seg.update(x-1,a[x-1])
  else:
    print(seg.query(x-1,y))

#・グリッドから要素が存在する部分だけ切り抜く
def clip(x,h,w):
  u=h
  d=0
  l=w
  r=0
  for i in range(h):
    for j in range(w):
      if x[i][j]=='#':
        u=min(u,i)
        l=min(l,j)
        d=max(d,i)
        r=max(r,j)
    
  a,b=d-u+1,r-l+1
  y=[]
  for i in range(u,d+1):
    y.append(x[i][l:r+1])
  return a,b,y

#・グリッドの座標(1index)→通し番号(0index)
def convert(x,y): 
    return w*(y-1)+x-1