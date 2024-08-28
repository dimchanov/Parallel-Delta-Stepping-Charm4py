#ifndef __GRAPH_HPC_DEFS_H
#define __GRAPH_HPC_DEFS_H

#include <stdint.h>
#include <stdbool.h>

#define DEFAULT_ARITY 16
#define SMALL_COMPONENT_EDGES_THRESHOLD   2
#define FNAME_LEN   256

typedef uint32_t vertex_id_t;
typedef uint64_t edge_id_t;
typedef double weight_t;

/* The graph data structure*/
typedef struct
{
    /***
     The minimal graph repesentation consists of:
     n        -- the number of vertices
     m        -- the number of edges
     endV     -- an array of size 2*m (m for directed graphs) that stores the 
                 destination ID of an edge <src->dest>.
     rowsIndices -- an array of size n+1 that stores the degree 
                 (out-degree in case of directed graphs) and pointers to
                 the endV array. The degree of vertex i is given by 
                 rowsIndices[i+1]-rowsIndices[i], and the edges out of i are
                 stored in the contiguous block endV[rowsIndices[i] .. rowsIndices[i+1]-1].
     Vertices are ordered from 0 in our internal representation
     ***/
    int scale; /* log2 of vertices number */
    vertex_id_t n;
    edge_id_t m;
    int avg_vertex_degree; /* relation m / n */ /* generation only */
    bool directed; 

    double a, b, c; /* RMAT graph parameters*/ /* generation only */

    bool permute_vertices; /* generation only */

    edge_id_t* rowsIndices;
    vertex_id_t* endV;
    
    /* Edge weights */
    weight_t* weights;

    weight_t min_weight, max_weight; /* generation only */

    /* Search parameters */
    uint32_t nRoots;
    vertex_id_t *roots;
    edge_id_t *numTraversedEdges;

    /* Distributed version parameters */
    int nproc, rank;
    vertex_id_t local_n; /* local vertices number */
    edge_id_t local_m; /* local edges number */

} graph_t;

/* write graph to file */
void writeGraph(graph_t *G, char *filename);

/* read graph from file */
void readGraph(graph_t *G, char *filename);

/* free graph memory */
void freeGraph(graph_t *G);

#define MAX_NLEVELS 500 // max number of levels in bfs for validation

#ifdef __cplusplus
/* Single Source Shortest Path */
extern "C" void sssp(vertex_id_t root, graph_t *G, weight_t *dist, uint64_t *traversed_edges);
/* initialize algorithm memory */
extern "C" void init_sssp(graph_t *G);
extern "C" void finalize_sssp();
#else
void sssp(vertex_id_t root, graph_t *G, weight_t *dist, uint64_t *traversed_edges);
void init_sssp(graph_t *G);
void finalize_sssp();
#endif

//void sssp_mpi(vertex_id_t root, graph_t *G, weight_t *dist);

extern int lgsize; /* log2 (number of processes) */
#define MOD_SIZE(v) ((v) & ((1 << lgsize) - 1))
#define DIV_SIZE(v) ((v) >> lgsize)
#define MUL_SIZE(x) ((x) << lgsize)

/* macroses for obtaining vertices distribution between nodes */
#define VERTEX_OWNER(v, t, s) ((int)(MOD_SIZE(v)))
#define VERTEX_LOCAL(v, t, s, r) ((vertex_id_t)(DIV_SIZE(v)))
#define VERTEX_TO_GLOBAL(r, i) ((vertex_id_t)(MUL_SIZE((vertex_id_t)i) + (int)(r)))


#endif
