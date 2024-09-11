import sys
import os
from graph import Graph, read_graph, weight_tf
from struct import pack, unpack, calcsize
import time


class DeltaStepping:
    def __init__(self, G: Graph, s: int, delta: float) -> None:
        self.G = G
        self.s = s
        self.delta = delta

        vertices = set(range(self.G.n))
        vertices.remove(self.s)      
        self.d = [-1.0] * self.G.n
        self.d[self.s] = 0.0
        self.B = {
            0: set([self.s]),
            -1: vertices
        }

    def run(self):
        index = 0
        while True:
            self.process_bucket(index)
            try:
                index = min(
                    k for k, v in self.B.items()
                    if len(v) > 0 and k > index
                )
            except:
                break

    def process_bucket(self, index: int):
        A = self.B[index].copy()
        while len(A):
            updated_vertices = set()
            for u in A:
                for v_index in range(self.G.rows_indices[u], self.G.rows_indices[u + 1]):
                    v = self.G.end_v[v_index]
                    self.relax(u, v, v_index, updated_vertices)
            A = self.B[index].intersection(updated_vertices)
            if len(self.B[index]) == 0:
                del self.B[index]
            
    def relax(self, u: int, v: int, v_index: int, updated_vertices: set):
        is_updated = False
        if self.d[v] == -1.0:
            self.d[v] = self.d[u] + self.G.weights[v_index]
            j = int(self.d[v] / self.delta)
            if j in self.B:
                self.B[j].add(v)
                self.B[-1].remove(v)
            else:
                self.B[j] = set([v])
                self.B[-1].remove(v)
            is_updated = True
        else:
            i = int(self.d[v] / self.delta)
            if self.d[v] > self.d[u] + self.G.weights[v_index]:
                self.d[v] = self.d[u] + self.G.weights[v_index]
                is_updated = True
            j = int(self.d[v] / self.delta)
            if j < i:
                if j in self.B:
                    self.B[j].add(v)
                    self.B[i].remove(v)
                else:
                    self.B[j] = set([v])
                    self.B[i].remove(v)
        if is_updated:
            updated_vertices.add(v)

    def write_distance(self, filename: str) -> None:
        print("write_distance started")
        with open(filename, "wb") as out_file:
            for i in range(self.G.n):
                out_file.write(pack(weight_tf, self.d[i]))
        print("write_distance finished")

    def read_distance(self, filename: str) -> None:
        print("read_distance started")
        d = []
        with open(filename, "rb") as in_file:
            for i in range(self.G.n):
                d.append(
                    unpack(weight_tf, in_file.read(calcsize(weight_tf)))[0]
                )
        print(d)
        print("read_distance finished")
    

def main():
    G = Graph()
    filepath = sys.argv[1]
    _, filename = os.path.split(filepath)

    read_graph(G, filepath)

    delta = float(G.n) / G.m
    delta = 3
    DS = DeltaStepping(G, G.roots[0], delta)
    start_time = time.time()
    DS.run()
    finish_time = time.time()
    DS.write_distance(filename + "_ds_ans")

    # DS.read_distance(filepath + "_ds_ans")
    elapsed_time = finish_time - start_time
    print(f"Elapsed time: {elapsed_time} s", flush=True)
    pass


if __name__ == '__main__':
    main()








