import os
import sys
import time

import graph
from graph import Graph
from struct import pack, unpack, calcsize

from charm4py import charm, Chare, Group, coro, Channel, Reducer


class Process(Chare):
    MAX_NUMBER = 10**9

    def set_process_vertices(self):
        local_n = self.G.n // charm.numPes()
        offset = self.G.rows_indices[local_n * self.thisIndex]
        local_row_indecies = self.G.rows_indices[local_n * self.thisIndex: local_n * (self.thisIndex + 1) + 1]
        local_end_v = self.G.end_v[local_row_indecies[0]: local_row_indecies[-1]]
        local_weights = self.G.weights[local_row_indecies[0]: local_row_indecies[-1]]
        for i in range(len(local_row_indecies)):
            local_row_indecies[i] -= offset
        
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
        if not is_mpi:
            self.set_process_vertices()
        self.G.n_roots = 10
        self.G.roots = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.G.num_traversed_edges = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.G.first_vertex = self.G.local_n * self.thisIndex

        self.channels = [Channel(self, remote=self.thisProxy[rank]) for rank in range(charm.numPes())]
        self.channels.pop(self.thisIndex)


    @coro
    def sssp(self, delta: float = None):
        if delta is None:
            delta = (self.G.n / self.G.m)
        self.delta = delta

        # print(f"Process {self.thisIndex}: {self.G.params_str()} \nfirst_vertex: {self.G.first_vertex}\ndelta: {delta}")

        self.distances_map = {
            vertex: self.MAX_NUMBER for vertex in self.G.end_v
        }
        self.distances_map.update(
            (vertex + self.G.first_vertex, self.MAX_NUMBER) for vertex in range(self.G.local_n)
        )

        self.buckets = {self.MAX_NUMBER: set(vertex + self.G.first_vertex for vertex in range(self.G.local_n))}
        self.bucket_index = 0

        if self.thisIndex == 0:
            self.distances_map[0] = 0
            self.buckets[self.MAX_NUMBER].remove(0)
            self.buckets[self.bucket_index] = set([0])        

        start_time = time.time()
        while True:
            # Выбирается минимальный bucket_index всех процессов
            self.bucket_index = self.allreduce(self.bucket_index, Reducer.min).get()
            if self.bucket_index == self.MAX_NUMBER:
                break

            self.buckets.setdefault(self.bucket_index, set())
            self.process_bucket()

            # Выбирается минимальный bucket_index данного процесса
            min_bucket_index = self.MAX_NUMBER
            for bucket_index in list(self.buckets):
                if bucket_index <= self.bucket_index or len(self.buckets[bucket_index]) == 0:
                    self.buckets.pop(bucket_index)
                else:
                    min_bucket_index = bucket_index if bucket_index < min_bucket_index else min_bucket_index
            self.bucket_index = min_bucket_index

        finish_time = time.time()
        elapsed_time = finish_time - start_time
        # print(f"Process {self.thisIndex} Elapsed time: {elapsed_time} s", flush=True)

    def update_work_bucket(self, vertices_distance: dict):
        self.work_bucket.update(vertices_distance)
        self.buckets[self.bucket_index].update(vertices_distance)
        for vertex, new_distance in vertices_distance.items():
            if new_distance < self.distances_map.get(vertex, self.MAX_NUMBER):
                bucket_index = int(self.distances_map.get(vertex, self.MAX_NUMBER) / self.delta)
                new_bucket_index = int(new_distance / self.delta)

                if new_bucket_index < bucket_index:
                    if new_bucket_index in self.buckets:
                        self.buckets[new_bucket_index].add(vertex)
                    else:
                        self.buckets[new_bucket_index] = set([vertex])
                    if bucket_index in self.buckets:
                        self.buckets[bucket_index].discard(vertex)
                
                self.distances_map[vertex] = new_distance

    def process_bucket(self):
        self.work_bucket = self.buckets[self.bucket_index].copy()

        while True:
            if self.allreduce(len(self.work_bucket), Reducer.max).get() == 0:
                break

            # Синхронизировать buckets and distances_map
            vertices_distance_by_rank = {}
            for vertex in self.work_bucket:
                vertices_distance_by_rank[vertex] = self.distances_map[vertex]

            futures = []
            for rank in range(charm.numPes()):
                if rank != self.thisIndex and len(vertices_distance_by_rank) > 0:
                              futures.append(
                        self.thisProxy[rank].update_work_bucket(
                            vertices_distance_by_rank, awaitable=True)
                    ) 
            for future in futures:
                future.get()


            if self.allreduce(len(self.work_bucket), Reducer.max).get() == 0:
                break
            del vertices_distance_by_rank

            updated_vertices = set()
            for sv in self.work_bucket:
                sv_index = sv - self.G.first_vertex
                if not (0 <= sv_index < self.G.local_n):
                    continue
                for ev_index_in_end_v in range(self.G.rows_indices[sv_index], self.G.rows_indices[sv_index + 1]):
    
                    self.relax(sv, ev_index_in_end_v, updated_vertices)
            self.work_bucket = self.buckets[self.bucket_index].intersection(updated_vertices)

    def relax(self, sv: int, ev_index_in_end_v: int, updated_vertices: set):
        ev = self.G.end_v[ev_index_in_end_v]

        dist_to_ev = self.distances_map[ev]
        dist_to_ev_from_sv = self.distances_map[sv] + self.G.weights[ev_index_in_end_v]

        ev_bucket_index = int(dist_to_ev / self.delta)
        ev_from_sv_bucket_index = int(dist_to_ev_from_sv / self.delta)

        if dist_to_ev_from_sv < dist_to_ev:
            self.distances_map[ev] = dist_to_ev_from_sv
            if ev_from_sv_bucket_index == self.bucket_index:
                updated_vertices.add(ev)

        if ev_from_sv_bucket_index < ev_bucket_index:
            if ev_from_sv_bucket_index in self.buckets:
                self.buckets[ev_from_sv_bucket_index].add(ev)
            else:
                self.buckets[ev_from_sv_bucket_index] = set([ev])
            if ev_bucket_index in self.buckets:
                self.buckets[ev_bucket_index].discard(ev)

    def get_self_distances(self):
        return [self.distances_map[vertex] for vertex in range(self.G.first_vertex, self.G.first_vertex + self.G.local_n)]
    
    @coro
    def write_distance(self, filename: str) -> None:
        if self.thisIndex == 0:
            print("write_distance started")
            with open(filename, "wb") as out_file:
                for rank in range(charm.numPes()):
                    for distance in self.thisProxy[rank].get_self_distances(ret=True).get():
                        if distance == self.MAX_NUMBER:
                            distance = -1
                        out_file.write(pack(graph.weight_tf, distance))
            print("write_distance finished")
    

def main(args):
    print('\nRunning sssp Graph on', charm.numPes(), 'processors')
    filename = sys.argv[1]
    group_proxy = Group(Process)

    is_mpi = False
    if len(sys.argv) > 2:
        is_mpi = sys.argv[2] == "True"
    print(f"is_mpi={is_mpi}")

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


    start_time = time.time()
    # Запускаем алгоритм delta stepping на распределенной памяти
    futures = []
    for rank in range(charm.numPes()):
        futures.append(group_proxy[rank].sssp(awaitable=True))
    for future in futures:
        future.get()
    finish_time = time.time()

    # Записываем результирующие расстояния в один файл
    futures = []
    for rank in range(charm.numPes()):
        futures.append(group_proxy[rank].write_distance(filename + "_mpi_ans", awaitable=True))
    for future in futures:
        future.get()

    elapsed_time = finish_time - start_time
    print(f"Elapsed time: {elapsed_time} s", flush=True)
    exit()


charm.start(main)