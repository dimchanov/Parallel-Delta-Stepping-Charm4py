from struct import pack, unpack, calcsize
from typing import Text, List
import sys


vertex_id_tf = "<I"
edge_id_tf = "<Q"
uint8_tf = "<B"
bool_tf = "?"
weight_tf = "<d"


class Graph:

    def __init__(self, name: str = "0") -> None:
        self.name = name
        self.scale: int = 0
        self.n: int = 0
        self.m: int = 0
        self.avg_vertex_degree: int = 0
        self.directed: bool = False

        self.a, self.b, self.c = 0.0, 0.0, 0.0

        self.permute_vertices: bool = False
        self.rows_indices: List[int] = []
        self.end_v: List[int] = []
        self.weights: List[float] = []
        self.min_weight, self.max_weight = 0.0, 1.0

        self.n_roots: int = 1
        self.roots: List[int] = [0]
        self.num_traversed_edges: List[int] = [0]

        self.local_n: int = None
        self.local_m: int = None
        self.first_vertex: int = 0

    def __str__(self) -> str:
        desc = f"\n*****     Graph {self.name}     *****\n"
        desc += f"Vertices count = {self.n}; Edges count = {self.m}\n"
        desc += f"Vertices local count = {self.local_n}; Edges local count = {self.local_m}\n"
        for v_index in range(self.local_n):
            v = self.first_vertex + v_index
            desc += f"Vertex = {v:3d}\nEdges = {{\n"
            for u_index in range(self.rows_indices[v_index], self.rows_indices[v_index + 1]):
                u = self.end_v[u_index]
                desc += f"    ({v:3d}, {u:3d}) weight = {self.weights[u_index]:.9f},\n"
            desc += "}\n"
        return desc

    def params_str(self) -> str:
        desc = f"{{\n\
            n={self.n},\n\
            local_n={self.local_n},\n\
            count end_v={len(self.end_v)},\n\
            }}"
        return desc


def read_graph(G: Graph, filename: str, is_mpi: bool = False) -> None:
    print("read_graph started from ", filename, flush=True)
    with open(filename, "rb") as in_file:
        n = G.n = unpack(vertex_id_tf, in_file.read(calcsize(vertex_id_tf)))[0]
        arity = unpack(edge_id_tf, in_file.read(calcsize(edge_id_tf)))[0]
        m = G.m = arity * G.n

        if is_mpi:
            n = G.local_n = unpack(vertex_id_tf, in_file.read(calcsize(vertex_id_tf)))[0]
            m = G.local_m = unpack(edge_id_tf, in_file.read(calcsize(edge_id_tf)))[0]

        G.directed = unpack(bool_tf, in_file.read(calcsize(bool_tf)))[0]
        G.align = unpack(uint8_tf, in_file.read(calcsize(uint8_tf)))[0]
        G.rows_indices = [
            unpack(edge_id_tf, in_file.read(calcsize(edge_id_tf)))[0] 
            for _ in range(n + 1)
        ]
        G.end_v = [
            unpack(vertex_id_tf, in_file.read(calcsize(vertex_id_tf)))[0] 
            for _ in range(G.rows_indices[-1])
        ]
        G.n_roots = unpack(vertex_id_tf, in_file.read(calcsize(vertex_id_tf)))[0]
        G.roots = [
            unpack(vertex_id_tf, in_file.read(calcsize(vertex_id_tf)))[0] 
            for _ in range(G.n_roots)
        ]
        G.num_traversed_edges = [
            unpack(edge_id_tf, in_file.read(calcsize(edge_id_tf)))[0] 
            for _ in range(G.n_roots)
        ]
        G.weights = [
            unpack(weight_tf, in_file.read(calcsize(weight_tf)))[0] 
            for _ in range(m)
        ]
        G.min_weight = min(G.weights)
        G.max_weight = max(G.weights)

        if G.local_n is None:
            G.local_n = G.n
        if G.local_m is None:
            G.local_m = G.m
    print("read_graph finished", flush=True)


def write_graph(G: Graph, filename: str, is_mpi: bool = False) -> None:
    print("write_graph started to", filename, flush=True)
    with open(filename, "wb") as out_file:
        n = G.n
        m = G.m
        out_file.write(pack(vertex_id_tf, G.n))
        arity = G.m // G.n
        out_file.write(pack(edge_id_tf, arity))

        if is_mpi:
            n = G.local_n
            out_file.write(pack(vertex_id_tf, G.local_n))
            m = G.local_m
            out_file.write(pack(edge_id_tf, G.local_m))

        out_file.write(pack(bool_tf, G.directed))
        align = 0
        out_file.write(pack(uint8_tf, align))

        for i in range(n + 1):
            out_file.write(pack(edge_id_tf, G.rows_indices[i]))
        for i in range(G.rows_indices[-1]):
            out_file.write(pack(vertex_id_tf, G.end_v[i]))

        out_file.write(pack(vertex_id_tf, G.n_roots))
        for i in range(G.n_roots):
            out_file.write(pack(vertex_id_tf, G.roots[i]))
        for i in range(G.n_roots):
            out_file.write(pack(edge_id_tf, G.num_traversed_edges[i]))
        for i in range(m):
            out_file.write(pack(weight_tf, G.weights[i]))
    print("write_graph finished", flush=True)
    

def gen_test_graph():
    G = Graph()
    G.n = 8
    G.m = 16
    G.rows_indices = [0, 4, 6, 8, 10, 12, 13, 15, 16]
    G.end_v = [1, 3, 4, 6, 0, 2, 1, 3, 0, 2, 0, 5, 4, 0, 7, 6]
    G.weights = [3, 5, 3, 3, 3, 3, 3, 1, 5, 1, 3, 5, 5, 3, 1, 1]
    write_graph(G, "test_graph", flush=True)



def main():
    G = Graph()
    read_graph(G, sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else False)
    # print(G)
    write_graph(G, sys.argv[1] + "_serial")
    pass


if __name__ == '__main__':
    main()
    # gen_test_graph()

