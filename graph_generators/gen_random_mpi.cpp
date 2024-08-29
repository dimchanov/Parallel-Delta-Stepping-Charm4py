#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "defs.h"
#include "mpi.h"
#include <assert.h>
#include <math.h>
#include <error.h>
#include <iostream>

using namespace std;

#define FNAME_LEN   256
char outFilename[FNAME_LEN];

/* function returns size of edges block for current process */
static edge_id_t get_local_m(graph_t *G)
{
    return G->local_n * G->avg_vertex_degree;
}

static unsigned long long my_next = 1;

static double my_rand()
{
    my_next = my_next * 1365781351523LL + 12345;
    return (double)((my_next / (1LL << 32)) % (1LL << 31)) / (1LL << 31);
}

static void my_srand(unsigned seed)
{
    my_next = seed;
}

void usage(int argc, char **argv) 
{
	printf("Usage:\n");
	printf("    %s -s [options]\n", argv[0]);
    printf("Options:\n");
    printf("   -s <scale>, number of vertices is 2^<scale>\n");
    printf("   -k <half the average vertex degree>, default value is 16\n");
    printf("   -nRoots <value> -- number of search root vertices. Default value is 10\n");
    MPI_Finalize();
    exit(1);
}

/* distributed random graph generator */
void gen_random_graph_MPI(int argc, char** argv, graph_t *G) {
    MPI_Comm_size(MPI_COMM_WORLD, &G->nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &G->rank);

    /* init */
    G->scale = -1;
    G->directed = false;
    G->permute_vertices = true;
    G->min_weight = 0;
    G->max_weight = 1;
    /* default value */
    G->nRoots = 10;
    G->avg_vertex_degree = DEFAULT_ARITY;

    for (int i = 1; i < argc; ++i) {
		if (!strcmp(argv[i], "-s")) {
			G->scale = (int) atoi(argv[++i]);
		}
		if (!strcmp(argv[i], "-k")) {
			G->avg_vertex_degree = (int) atoi(argv[++i]);
        }
        if (!strcmp(argv[i], "-nRoots")) {
            G->nRoots = (uint32_t) atoi(argv[++i]);
        }
    }

	if (G->scale == -1) usage(argc, argv);
    sprintf(outFilename, "random-%d.%d", G->scale, G->rank);

    vertex_id_t n, local_n;
    edge_id_t local_m, num_dir_edges;
    edge_id_t offset;
    vertex_id_t *permV, tmpVal;
    bool permute_vertices;
    vertex_id_t u, v;
    int seed;
    vertex_id_t *edges;
    vertex_id_t *degree;
    int size, rank, lgsize;
    G->n = (vertex_id_t)1 << G->scale;
    G->m = G->n * (edge_id_t)G->avg_vertex_degree;
    permute_vertices = G->permute_vertices;
    n = G->n;
    unsigned TotVertices;
    TotVertices = G->n;

    weight_t *weight;
    weight_t min_weight, max_weight;

    MPI_Datatype MPI_VERTEX_ID_T;
    MPI_Type_contiguous(sizeof(vertex_id_t), MPI_BYTE, &MPI_VERTEX_ID_T);
    MPI_Type_commit(&MPI_VERTEX_ID_T);
    
    rank = G->rank;
    size = G->nproc;
    for (lgsize = 0; lgsize < size; ++lgsize) {
        if ((1 << lgsize) == size) {
            break;
        }
    }
    /* get size of vertices and edges blocks for current processes */
    local_n = n/size;
    G->local_n = local_n;
    local_m = get_local_m(G);
    num_dir_edges = local_m;
    local_m = 2 * local_m;
    G->local_m = local_m;

    weight = (weight_t *) malloc(num_dir_edges * sizeof(weight_t));
    assert(weight != NULL);

    edges = new vertex_id_t[2 * num_dir_edges];
    assert(edges != NULL);
    degree = new vertex_id_t[local_n];
    assert(degree != NULL);
    memset(degree, 0, sizeof(vertex_id_t) * local_n);

    seed = 2387 + rank;
    srand(seed);
    /* generate edges */
    for (edge_id_t i = 0; i < num_dir_edges; i++) {
        vertex_id_t u = rand() % n;
        vertex_id_t v = rand() % n;
        edges[2 * i + 0] = u;
        edges[2 * i + 1] = v;
    }

    /* reshuffle */
    if (permute_vertices) {
        permV = new vertex_id_t[n];
        assert(permV != NULL);
        
        my_srand(4791);
        for (vertex_id_t i = 0; i < n; i++) {
            permV[i] = i;
        }
        
        for (vertex_id_t i = 0; i < n; i++) {
            vertex_id_t j = n * my_rand();
            tmpVal = permV[i];
            permV[i] = permV[j];
            permV[j] = tmpVal;
        }
        
        for (edge_id_t i = 0; i < num_dir_edges; i++) {
            edges[2 * i + 0]  = permV[edges[2 * i + 0]];
            edges[2 * i + 1] = permV[edges[2 * i + 1]];
        }
        
        delete[] permV;
    }
    
    min_weight = G->min_weight;
    max_weight = G->max_weight;
    /* Generate edges weights */
    for (edge_id_t i=0; i<num_dir_edges; i++) {
        weight[i]  = min_weight + (max_weight-min_weight)*drand48();
    }

    weight_t *send_weight = (weight_t*) malloc (local_m * sizeof(weight_t));
    assert(send_weight != NULL);
    weight_t *recv_weight;
    vertex_id_t *send_edges = new vertex_id_t[2 * local_m];
    assert(send_edges != NULL);
    vertex_id_t *recv_edges;
    int *send_counts = new int[size];
    assert(send_counts != NULL);
    memset(send_counts, 0, sizeof(int) * size);
    int *recv_counts = new int[size];
    assert(recv_counts != NULL);
    memset(recv_counts, 0, sizeof(int) * size);
    edge_id_t *send_offsets_edge = new edge_id_t[size];
    assert(send_offsets_edge != NULL);
    memset(send_offsets_edge, 0, sizeof(edge_id_t) * size);
    edge_id_t *recv_offsets_edge = new edge_id_t[size];
    assert(recv_offsets_edge != NULL);
    memset(recv_offsets_edge, 0, sizeof(edge_id_t) * size);
    edge_id_t* send_offsets_weight = (edge_id_t*) calloc (size, sizeof(edge_id_t));
    assert(send_offsets_weight != NULL);
    edge_id_t* recv_offsets_weight = (edge_id_t*) calloc (size, sizeof(edge_id_t));
    assert(recv_offsets_weight != NULL);

    /* calc count of data in each process */
    for (edge_id_t i = 0; i < num_dir_edges; i++) {
        int proc_id = VERTEX_OWNER(edges[2 * i + 0], TotVertices, size);
        send_counts[proc_id]++;
        proc_id = VERTEX_OWNER(edges[2 * i + 1], TotVertices, size);
        send_counts[proc_id]++;
    }
    
    /* calc offsets */
    for (int i = 1; i < size; i++) {
        send_offsets_edge[i] = send_offsets_edge[i - 1] + 2 * send_counts[i - 1];
        send_offsets_weight[i] = send_offsets_weight[i-1] + send_counts[i-1];
    }
    
    /* clear send_counts for next using */
    for (int i = 0; i < size; i++) {
        send_counts[i] = 0;
    }
    
    /* copy edges to send_data */
    for (edge_id_t i = 0; i < num_dir_edges; i++) {
        int proc_id = VERTEX_OWNER(edges[2 * i + 0], TotVertices, size);
        offset = send_offsets_edge[proc_id] + 2 * send_counts[proc_id];
        send_edges[offset + 0] = edges[2 * i + 0];
        send_edges[offset + 1] = edges[2 * i + 1];
        offset = send_offsets_weight[proc_id] + send_counts[proc_id];
        send_weight[offset] = weight[i];
        send_counts[proc_id]++;
        proc_id = VERTEX_OWNER(edges[2 * i + 1], TotVertices, size);
        offset = send_offsets_edge[proc_id] + 2 * send_counts[proc_id];
        send_edges[offset + 0] = edges[2 * i + 1];
        send_edges[offset + 1] = edges[2 * i + 0];
        offset = send_offsets_weight[proc_id] + send_counts[proc_id];
        send_weight[offset] = weight[i];
        send_counts[proc_id]++;
    }
    
    delete[] edges;
    delete[] weight;
    MPI_Request request[size];
    MPI_Status status[size];
    /* report counts to each process */
    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);
    for (int i=1; i<size; i++) {
        recv_offsets_weight[i] = recv_offsets_weight[i-1] + recv_counts[i-1];
    }
    edge_id_t counts = 0;
    for (int i = 0; i < size; i++) {
        counts += recv_counts[i];
    }
    
    recv_weight = (weight_t*) malloc (counts * sizeof(weight_t));
    assert(recv_weight != NULL);

    /* send weights to each process */ 
    for (int i = 0; i < size; i++) {
        MPI_Irecv(&recv_weight[recv_offsets_weight[i]], recv_counts[i], MPI_DOUBLE, i, G->rank, MPI_COMM_WORLD, &request[i]);
    }   
    for (int i = 0; i < size; i++) {
        MPI_Send(&send_weight[send_offsets_weight[i]], send_counts[i], MPI_DOUBLE, i, i, MPI_COMM_WORLD);
    }
    MPI_Waitall(size, request, status);   

    free(send_weight);
    free(send_offsets_weight);
    free(recv_offsets_weight);

    /* calc offsets and number of elements for the next MPI_Send */
    for (int i = 0; i < size; i++) {
        recv_counts[i] = 2 * recv_counts[i];
        send_counts[i] = 2 * send_counts[i]; 
    }
    
    for (int i = 1; i < size; i++) {
        recv_offsets_edge[i] = recv_offsets_edge[i - 1] + recv_counts[i - 1];
    }
    
    recv_edges = new vertex_id_t[2 * counts];
    assert(recv_edges != NULL);
    
    /* send edges to each process */
    for (int i = 0; i < size; i++) {
        MPI_Irecv(&recv_edges[recv_offsets_edge[i]], recv_counts[i], MPI_VERTEX_ID_T, i, G->rank, MPI_COMM_WORLD, &request[i]);
    }
    
    for (int i = 0; i < size; i++) {
        MPI_Send(&send_edges[send_offsets_edge[i]], send_counts[i], MPI_VERTEX_ID_T, i, i, MPI_COMM_WORLD);
    }
    
    MPI_Waitall(size, request, status);
    /* saving new value for local_m */
    local_m = counts;
    G->local_m = local_m;
    /* undirected graph, each edge is stored twice; if edge is (u, v), then it's
     *stored at the vertex u and at the vertex v */
    G->m *= 2;
    delete[] send_edges;
    delete[] recv_offsets_edge;
    delete[] send_offsets_edge;
    delete[] recv_counts;
    delete[] send_counts;
    
    for (edge_id_t i = 0; i < 2 * G->local_m; i += 2) {
        degree[VERTEX_LOCAL(recv_edges[i], TotVertices, size, rank)]++;
    }
    
    /* update graph data structure */
    G->endV = new vertex_id_t[G->local_m];
    assert(G->endV != NULL);
    memset(G->endV, 0, sizeof(vertex_id_t) * G->local_m);
    G->rowsIndices = new edge_id_t[local_n + 1];
    assert(G->rowsIndices != NULL);

    G->weights = (weight_t *) malloc(G->local_m * sizeof(weight_t));       
    assert(G->weights != NULL);

    G->rowsIndices[0] = 0; 
    for (vertex_id_t i = 1; i <= G->local_n; i++) {
        G->rowsIndices[i] = G->rowsIndices[i - 1] + degree[i - 1];
    }
    
    for (edge_id_t i = 0; i < 2 * G->local_m; i += 2) {
        u = VERTEX_LOCAL(recv_edges[i + 0], TotVertices, size, rank);
        v = recv_edges[i + 1];
        offset = degree[u]--;
        G->endV[G->rowsIndices[u] + offset - 1] = v;
        G->weights[G->rowsIndices[u]+offset-1] = recv_weight[i/2];
    }
    
    delete[] recv_edges;
    delete[] degree;
    delete[] recv_weight;

}

/* write graph to file */
void writeGraph(graph_t *G, char *filename) {
    MPI_Comm_size(MPI_COMM_WORLD, &G->nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &G->rank);

    FILE *F = fopen(filename, "wb");
    if (!F) error(EXIT_FAILURE, 0, "Error in opening file %s", filename);
	size_t objects_written = 0;

    objects_written = fwrite(&G->n, sizeof(vertex_id_t), 1, F);
    assert(objects_written == 1);
    edge_id_t arity = G->m / G->n;
    objects_written = fwrite(&arity, sizeof(edge_id_t), 1, F);
    assert(objects_written ==  1);

    objects_written = fwrite(&G->local_n, sizeof(vertex_id_t), 1, F);
    assert(objects_written == 1);
    objects_written = fwrite(&G->local_m, sizeof(edge_id_t), 1, F);
    assert(objects_written == 1);

    objects_written = fwrite(&G->directed, sizeof(bool), 1, F);
    assert(objects_written == 1);
    uint8_t align = 0;
    objects_written = fwrite(&align, sizeof(uint8_t), 1, F);
    assert(objects_written == 1);

    objects_written = fwrite(G->rowsIndices, sizeof(edge_id_t), G->local_n+1, F);
    assert(objects_written == G->local_n+1);
    printf("last rowsIndex: %ld\n", G->rowsIndices[G->local_n]);
    
    objects_written = fwrite(G->endV, sizeof(vertex_id_t), G->rowsIndices[G->local_n], F);
    assert(objects_written == G->rowsIndices[G->local_n]);

    objects_written = fwrite(&G->nRoots, sizeof(uint32_t), 1, F);
    assert(objects_written == 1);
    objects_written = fwrite(G->roots, sizeof(vertex_id_t), G->nRoots, F);
    assert(objects_written == G->nRoots);
    objects_written = fwrite(G->numTraversedEdges, sizeof(edge_id_t), G->nRoots, F);
    assert(objects_written == G->nRoots);

    objects_written = fwrite(G->weights, sizeof(weight_t), G->local_m, F);
    assert(objects_written == G->local_m);
    fclose(F);
}

void printGraph(graph_t *G)
{
	int i,j;
	for (i = 0; i < (int)G->local_n; ++i) {
		printf("%d:", i);
		for (j=G->rowsIndices[i]; j < (int)G->rowsIndices[i+1]; ++j)
			printf("%d (%f), ", G->endV[j], G->weights[j]);
		printf("\n");
	}
}

int main(int argc, char **argv) {
    MPI_Init (&argc, &argv);
    graph_t g;

    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    gen_random_graph_MPI(argc, argv, &g);

    for(int i = 0; i < size; ++i) {
        if(rank == i) {
            //printGraph(&g);
            writeGraph(&g, outFilename);

        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
