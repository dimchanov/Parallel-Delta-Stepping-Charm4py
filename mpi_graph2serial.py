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

    exit()


charm.start(main)
