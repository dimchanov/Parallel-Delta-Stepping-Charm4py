import os
import sys
import time

import graph
from graph import *
from struct import pack, unpack, calcsize
from typing import List

from charm4py import charm, Chare, Group, coro, Channel, Reducer


class MPIGraph(Chare):

    def set_process_vertices(self):
        local_n = self.G.n // charm.numPes()
        offset = self.G.rows_indices[local_n * self.thisIndex]
        local_row_indecies = self.G.rows_indices[local_n * self.thisIndex: local_n * (self.thisIndex + 1) + 1]
        local_end_v = self.G.end_v[local_row_indecies[0]: local_row_indecies[-1]]
        local_weights = self.G.weights[local_row_indecies[0]: local_row_indecies[-1]]
        for i in range(len(local_row_indecies)):
            local_row_indecies[i] -= offset
        
        self.G.first_vertex = local_n * self.thisIndex
        self.G.local_n = local_n
        self.G.local_m = len(local_end_v)
        self.G.rows_indices = local_row_indecies
        self.G.end_v = local_end_v
        self.G.weights = local_weights

    def read_graph(self, filename: str, is_mpi: bool = False):
        self.G = Graph()
        self.G.name = str(self.thisIndex)
        if is_mpi:
            filename = f"{filename}.{self.thisIndex}"
        graph.read_graph(self.G, filename, is_mpi)
        self.G.n_roots = 10
        self.G.roots = list(range(10))
        self.G.num_traversed_edges = [0] * 10
        if not is_mpi:
            self.set_process_vertices()

        self.is_mpi = is_mpi

    def get_rows_indices(self) -> List[int]:
        return self.G.rows_indices
    
    def get_end_v(self) -> List[int]:
        return self.G.end_v
    
    def get_weights(self) -> List[int]:
        return self.G.weights

    @coro
    def write_mpi_graph(self, filename: str):
        print("write_mpi_graph started to", filename)
        G = self.G
        with open(filename, "wb") as out_file:
            # n
            out_file.write(pack(vertex_id_tf, G.n))
            # arity
            arity = G.m // G.n
            out_file.write(pack(edge_id_tf, arity))
            # directed
            out_file.write(pack(bool_tf, G.directed))
            # align
            align = 0
            out_file.write(pack(uint8_tf, align))
            # rows_indices
            start_rows_index = 0
            for rank in range(charm.numPes()):
                if rank == self.thisIndex:
                    rows_indices = self.G.rows_indices
                else:
                    rows_indices = self.thisProxy[rank].get_rows_indices(ret=True).get()
                last_rows_index = rows_indices.pop(-1)
                for rows_index in rows_indices:
                    out_file.write(pack(edge_id_tf, start_rows_index + rows_index))
                start_rows_index += last_rows_index
            out_file.write(pack(edge_id_tf, start_rows_index))
            # end_v
            for rank in range(charm.numPes()):
                if rank == self.thisIndex:
                    end_v = self.G.end_v
                else:
                    end_v = self.thisProxy[rank].get_end_v(ret=True).get()
                for end_vertex in end_v:
                    out_file.write(pack(vertex_id_tf, end_vertex))
            # n_root
            out_file.write(pack(vertex_id_tf, G.n_roots))
            # roots
            for i in range(G.n_roots):
                out_file.write(pack(vertex_id_tf, G.roots[i]))
            # num_traversed_edges
            for i in range(G.n_roots):
                out_file.write(pack(edge_id_tf, G.num_traversed_edges[i]))
            # weights
            for rank in range(charm.numPes()):
                if rank == self.thisIndex:
                    weights = self.G.weights
                else:
                    weights = self.thisProxy[rank].get_weights(ret=True).get()
                for weight in weights:
                    out_file.write(pack(weight_tf, weight))
        print("write_mpi_graph finished")


def main(args):
    print('\nRunning MPIGraph on', charm.numPes(), 'process')
    filename = sys.argv[1]
    is_mpi = False
    if len(sys.argv) > 2:
        is_mpi = sys.argv[2] == "True"
    print(f"is_mpi={is_mpi}")
    group_proxy = Group(MPIGraph)

    if is_mpi:
        # Читаем граф из mpi файлов на разные процессы
        futures = [
            group_proxy[rank].read_graph(filename, is_mpi, awaitable=True)
            for rank in range(charm.numPes())
        ]
        for future in futures:
            future.get()
    else:
        # Читаем граф из одного файла на разные процессы (на каждом одинаковая часть)
        for rank in range(charm.numPes()):
            group_proxy[rank].read_graph(filename, is_mpi, awaitable=True).get()


    # Записать граф на 0 процессе
    group_proxy[0].write_mpi_graph(f"{filename}_mpi_graph2serial", awaitable=True).get()
    exit()


charm.start(main)
