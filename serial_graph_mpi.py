import sys
import time

import graph
from graph import Graph
from struct import pack, unpack, calcsize

from charm4py import charm, Chare, Group, coro, Channel, Reducer


class Process(Chare):
    MAX_NUMBER = 10**15

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


    def read_graph(self, base_filename: str):
        self.G = Graph()
        self.G.name = str(self.thisIndex)
        filename = base_filename # filename = f"base_filename_{self.thisIndex}"
        graph.read_graph(self.G, filename)
        self.set_process_vertices()

    @coro
    def sssp(self, delta: float = None):
        if delta is None:
            delta = self.G.n / self.G.m
        self.delta = delta
        # Массив вершин (свой на каждом процессе)
        self.vertices = set(range(self.G.first_vertex, self.G.first_vertex + self.G.local_n))
        # Массив расстояний. По умолчанию все значения -1, кроме корня, там 0
        self.self_distances = [-1.0] * self.G.n
        if self.thisIndex == 0:
            self.self_distances[0] = 0
        # расстояния до вершин других процессов
        self.distances_map = {nb_rank: {} for nb_rank in range(charm.numPes())}
        self.distances_map[self.thisIndex] = {vi: self.MAX_NUMBER for vi in range(self.G.local_n)}
        if self.thisIndex == 0:
            self.vertices = set(range(self.G.first_vertex + 1, self.G.first_vertex + self.G.local_n))
            self.buckets = {self.MAX_NUMBER: self.vertices}
            self.buckets[0] = set([0])
            self.distances_map[0][0] = 0
        else:
            self.buckets = {self.MAX_NUMBER: self.vertices}
        # Каналы к другим процессам
        self.channels = [Channel(self, remote=self.thisProxy[rank]) for rank in range(charm.numPes())]
        self.channels.pop(self.thisIndex)

        self.bucket_index = self.thisIndex

        while True:
            # Выбирается минимальный bucket_index всех процессов
            self.bucket_index = self.allreduce(self.bucket_index, Reducer.min).get()
            if self.bucket_index == self.MAX_NUMBER:
                break

            self.buckets.setdefault(self.bucket_index, set())
            self.process_bucket()

            min_bucket_index = self.MAX_NUMBER
            for bucket_index in list(self.buckets):
                bucket = self.buckets[bucket_index]
                if bucket_index <= self.bucket_index or len(bucket) == 0:
                    self.buckets.pop(bucket_index)
                else:
                    min_bucket_index = bucket_index if bucket_index < min_bucket_index else min_bucket_index
            self.bucket_index = min_bucket_index
        return

    def process_bucket(self, index: int = 0):
        work_bucket = self.buckets[self.bucket_index].copy()
        while True:
            # Синхронизировать buckets и distances_map
            rank = 0
            for channel in self.channels:
                if rank == self.thisIndex:
                    rank += 1
                channel.send(work_bucket, self.distances_map[rank])
                rank += 1
            v_distances = self.distances_map[self.thisIndex]
            for channel in charm.iwait(self.channels):
                cur_work_bucket, cur_distances_map = channel.recv()
                work_bucket.update(cur_work_bucket)
                for v_index, v_distance in cur_distances_map.items():
                    if v_distance < v_distances[v_index]:
                        v_distances[v_index] = v_distance
            self.buckets[self.bucket_index].update(work_bucket)

            if len(work_bucket) == 0:
                break
            updated_vertices = set()
            for sv in work_bucket:
                sv_index = sv - self.G.first_vertex
                # Вершина не этого процесса
                if sv_index not in self.distances_map[self.thisIndex]:
                    continue
                for dv_index_in_end_v in range(self.G.rows_indices[sv_index], self.G.rows_indices[sv_index + 1]):
                    self.relax(sv_index, dv_index_in_end_v, updated_vertices)
            work_bucket = self.buckets[self.bucket_index].intersection(updated_vertices)
            
    def relax(self, sv_index: int, dv_index_in_end_v: int, updated_vertices: set):
        dv = self.G.end_v[dv_index_in_end_v]
        is_updated = False

        dv_process_rank = dv // self.G.local_n
        dv_index = dv % self.G.local_n
        sv_to_dv = self.distances_map[dv_process_rank].get(dv_index, self.MAX_NUMBER)
        sv_to_dv_updated = self.distances_map[self.thisIndex][sv_index] + self.G.weights[dv_index_in_end_v]
        if sv_to_dv_updated < sv_to_dv:
            self.distances_map[dv_process_rank][dv_index] = sv_to_dv_updated
            updated_vertices.add(dv)
        dv_bucket_index = int(sv_to_dv / self.delta)
        dv_bucket_index_updated = int(sv_to_dv_updated / self.delta)
        if dv_bucket_index_updated < dv_bucket_index:
            # Переместить между buckets
            if dv_bucket_index_updated in self.buckets:
                self.buckets[dv_bucket_index_updated].add(dv)
            else:
                self.buckets[dv_bucket_index_updated] = set([dv])
            if dv_bucket_index in self.buckets:
                self.buckets[dv_bucket_index].discard(dv)

    @coro
    def write_distance(self, filename: str) -> None:
        if self.thisIndex == 0:
            print("write_distance started")
            distances_map = {0: self.distances_map[self.thisIndex]}
            with open(filename, "wb") as out_file:
                for channel in charm.iwait(self.channels):
                    rank, distances = channel.recv()
                    distances_map[rank] = distances
                for rank, distances in sorted(distances_map.items()):
                    for v_index, distance in sorted(distances.items()):
                        out_file.write(pack(graph.weight_tf, distance))
            print("write_distance finished")
        else:
            self.channels[0].send(self.thisIndex, self.distances_map[self.thisIndex])


    


def main(args):
    print('\nRunning sssp Graph on', charm.numPes(), 'processors')
    filename = sys.argv[1]
    group_proxy = Group(Process)

    # Читаем граф из одного файла на разные процессы (на каждом одинаковая часть)
    for rank in range(charm.numPes()):
        group_proxy[rank].read_graph(filename, awaitable=True).get()
    # charm.wait([group_proxy[rank].read_graph(filename, awaitable=True) for rank in range(charm.numPes())])

    start_time = time.time()
    # Запускаем алгоритм delta stepping на распределенной памяти
    futures = []
    for rank in range(charm.numPes()):
        futures.append(group_proxy[rank].sssp(awaitable=True))
    for future in futures:
        future.get()
    # charm.wait([group_proxy[rank].sssp(awaitable=True) for rank in range(charm.numPes())])
    finish_time = time.time()

    # Записываем результирующие расстояния в один файл
    futures = []
    for rank in range(charm.numPes()):
        futures.append(group_proxy[rank].write_distance(filename + "_mpi_ans", awaitable=True))
    for future in futures:
        future.get()
    # charm.wait([group_proxy[rank].write_distance(filename + "_mpi_ans", awaitable=True) for rank in range(charm.numPes())])

    elapsed_time = finish_time - start_time
    print(f"Elapsed time: {elapsed_time} s", flush=True)
    exit()


charm.start(main)