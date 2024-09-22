import os
import sys
import time
import numpy as np
# sys.path.append(os.path.join(os.path.dirname(__file__), "charm4py"))


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
        # print(self.G)

        self.channels = [Channel(self, remote=self.thisProxy[rank]) for rank in range(charm.numPes())]
        self.channels.pop(self.thisIndex)


    @coro
    def sssp(self, delta: float = None):
        start_time = time.time()
        if delta is None:
            delta = self.G.n / self.G.m
        self.delta = delta
        # Массив вершин (свой на каждом процессе)
        self.vertices = set(range(self.G.first_vertex, self.G.first_vertex + self.G.local_n))
        # Массив расстояний. По умолчанию все значения -1 (max float, но жалко память), кроме корня, там 0
        # self.self_distances = [-1.0] * self.G.n
        # if self.thisIndex == 0:
        #     self.self_distances[0] = 0
        # расстояния до вершин других процессов
        self.distances_map = {nb_rank: {} for nb_rank in range(charm.numPes())}
        self.distances_map[self.thisIndex] = {vi: self.MAX_NUMBER for vi in range(self.G.local_n)}
        if self.thisIndex == 0:
            # self.vertices = set(range(self.G.first_vertex + 1, self.G.first_vertex + self.G.local_n))
            self.buckets = {self.MAX_NUMBER: self.vertices}
            self.buckets[0] = set([0])
            self.distances_map[0][0] = 0
        else:
            self.buckets = {self.MAX_NUMBER: self.vertices}
        # Каналы к другим процессам
        # self.channels = [Channel(self, remote=self.thisProxy[rank]) for rank in range(charm.numPes())]
        # self.channels.pop(self.thisIndex)

        self.bucket_index = 0

        # self.updated_distances_map_indexes = [0] * charm.numPes()

        # print("VVVVVV", self.thisIndex, self.vertices)

        iter = 1 # for debug
        while True and iter > 0:
            # iter -= 1
            # Выбирается минимальный bucket_index всех процессов
            self.bucket_index = self.allreduce(self.bucket_index, Reducer.min).get()
            # print(f"Process {self.thisIndex} bucket_index: {self.bucket_index}", flush=True)
            if self.bucket_index == self.MAX_NUMBER:
                break

            self.buckets.setdefault(self.bucket_index, set())
            self.process_bucket()
            # print(f"Process {self.thisIndex} buckets count={len(self.buckets)}")

            min_bucket_index = self.MAX_NUMBER
            for bucket_index in list(self.buckets):
                bucket = self.buckets[bucket_index]
                if bucket_index <= self.bucket_index or len(bucket) == 0:
                    self.buckets.pop(bucket_index)
                else:
                    min_bucket_index = bucket_index if bucket_index < min_bucket_index else min_bucket_index
            self.bucket_index = min_bucket_index

            # self.bucket_index = min(
            #     k for k, v in self.buckets.items()
            #     if len(v) > 0 and k > self.bucket_index
            # )

        # self.allreduce(self.bucket_index, Reducer.min).get()

        # print(f"ANS Process {self.thisIndex}: {self.distances_map[self.thisIndex]}", flush=True)
        finish_time = time.time()
        elapsed_time = finish_time - start_time
        print(f"Process {self.thisIndex} Elapsed time: {elapsed_time} s", flush=True)
        return
    
    def update_bucket(self, vertices: np.array, distances: np.array) -> None:
        # self.work_bucket.update(bucket)
        # for sv in bucket:
        #     if sv // self.G.local_n == self.thisIndex:
        self.work_bucket.update(vertices)
        v_distances = self.distances_map[self.thisIndex]
        for v_index in range(len(vertices)):
            vertex = vertices[v_index]
            if distances[v_index] < v_distances[vertex % self.G.local_n]:
                v_distances[vertex % self.G.local_n] = distances[v_index]

    def process_bucket(self):
        self.work_bucket = self.buckets.get(self.bucket_index, set()).copy()

        # self.updated_vertices = [[] for _ in range(self.G.local_n)]
        # for v_index, v_distance in self.distances_map[self.thisIndex].items():
        #     self.updated_vertices[v_index] = [v_distance]

        tmp = 0
        while True and tmp < 10:
            # tmp += 1

            if self.allreduce(len(self.work_bucket), Reducer.max).get() == 0:
                break
            # Синхронизировать buckets
            futures = []
            # is_empty_work_bucket = len(self.work_bucket) == 0

            send_vertices = {rank: ([], []) for rank in range(charm.numPes())}
            for sv in self.work_bucket:
                rank = sv // self.G.local_n

                send_vertices[rank][0].append(sv)
                send_vertices[rank][1].append(self.distances_map[rank][sv % self.G.local_n])

            for rank in range(charm.numPes()):
                # print(f"Process {self.thisIndex}; Rank {rank}; {len(self.work_bucket)} {len(send_vertices[rank][0])}")
                if rank != self.thisIndex and len(send_vertices[rank][0]) > 0:
                    # send_vertices = {
                    #     sv: self.distances_map[rank][sv % self.G.local_n]
                    #     for sv in self.work_bucket if sv // self.G.local_n == rank
                    # }
                    # if len(send_vertices) > 0:
                    futures.append(
                        self.thisProxy[rank].update_bucket(
                            np.array(send_vertices[rank][0], dtype=int), np.array(send_vertices[rank][1], dtype=float), awaitable=True)
                    ) 
            
            # print(self.thisIndex, len(futures))
            for future in futures:
                future.get()
            if self.allreduce(len(self.work_bucket), Reducer.max).get() == 0:
                break


            # rank = 0
            # for channel in self.channels:
            #     if rank == self.thisIndex:
            #         rank += 1
            #     channel.send(np.array(send_vertices[rank][0], dtype=int), np.array(send_vertices[rank][1], dtype=float))
            #     rank += 1
            
            # v_distances = self.distances_map[self.thisIndex]
            # for channel in charm.iwait(self.channels):
            #     cur_vertices, cur_distances = channel.recv()
            #     self.work_bucket.update(cur_vertices)
            #     for v_index in range(len(cur_vertices)):
            #         vertex = cur_vertices[v_index]
            #         if cur_distances[v_index] < v_distances[vertex % self.G.local_n]:
            #             v_distances[vertex % self.G.local_n] = cur_distances[v_index]
            # print(self.thisIndex, len(futures))


            self.buckets[self.bucket_index].update(self.work_bucket)
            del send_vertices
            

            updated_vertices = set()
            for sv in self.work_bucket:
                sv_index = sv - self.G.first_vertex
                # Вершина не этого процесса
                if sv not in self.vertices:
                    continue
                # if sv_index not in self.distances_map[self.thisIndex]:
                #     continue
                # print(f"Process {self.thisIndex}", "VERTEX:", sv, list(range(self.G.rows_indices[sv_index], self.G.rows_indices[sv_index + 1])), flush=True)
                for dv_index_in_end_v in range(self.G.rows_indices[sv_index], self.G.rows_indices[sv_index + 1]):
                    self.relax(sv_index, dv_index_in_end_v, updated_vertices)
            # print(f"Process {self.thisIndex}", "NEW self.buckets:", self.buckets, self.distances_map, flush=True)
            self.work_bucket = self.buckets[self.bucket_index].intersection(updated_vertices)
            
    def relax(self, sv_index: int, dv_index_in_end_v: int, updated_vertices: set):
        # print("relax", end=' ')
        dv = self.G.end_v[dv_index_in_end_v]
        is_updated = False

        dv_process_rank = dv // self.G.local_n
        dv_index = dv % self.G.local_n
        # print("START", sv_index, dv_index_in_end_v, updated_vertices)
        sv_to_dv = self.distances_map[dv_process_rank].get(dv_index, self.MAX_NUMBER)
        sv_to_dv_updated = self.distances_map[self.thisIndex][sv_index] + self.G.weights[dv_index_in_end_v]

        dv_bucket_index = int(sv_to_dv / self.delta)
        dv_bucket_index_updated = int(sv_to_dv_updated / self.delta)

        if sv_to_dv_updated < sv_to_dv:
            self.distances_map[dv_process_rank][dv_index] = sv_to_dv_updated
            # self.updated_distances_map_indexes[dv_process_rank] = 1
            if dv_bucket_index_updated == self.bucket_index:
                updated_vertices.add(dv)
        # print(sv_index, dv_index_in_end_v, updated_vertices, dv_bucket_index, dv_bucket_index_updated)
        if dv_bucket_index_updated < dv_bucket_index:
            # Переместить между buckets
            if dv_bucket_index_updated in self.buckets:
                self.buckets[dv_bucket_index_updated].add(dv)
            else:
                self.buckets[dv_bucket_index_updated] = set([dv])
            if dv_bucket_index in self.buckets:
                self.buckets[dv_bucket_index].discard(dv)

            # if dv_bucket_index_updated == self.bucket_index:
            #     self.updated_distances_map_indexes[dv_process_rank] = 1

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
            # print([distance for rank, distances in sorted(distances_map.items()) for v_index, distance in sorted(distances.items())])
        else:
            self.channels[0].send(self.thisIndex, self.distances_map[self.thisIndex])


    


def main(args):
    print('\nRunning sssp Graph on', charm.numPes(), 'processors')
    filename = sys.argv[1]
    group_proxy = Group(Process)
    # charm.options.profiling = True

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