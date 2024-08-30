#include <stdio.h>
#include <stdlib.h>
#include <error.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <float.h>
#include "defs.h"

char* inFilename;
char* outFilename;
uint32_t rootNumberToValidate;
int nIters;
#if defined(CLOCK_MONOTONIC)
#define CLOCK CLOCK_MONOTONIC
#elif defined(CLOCK_REALTIME)
#define CLOCK CLOCK_REALTIME
#else
#error "Failed to find a timing clock."
#endif

void usage(int argc, char **argv)
{
    printf("Usage:\n");
    printf("    %s -in <input> [options]\n", argv[0]);
    printf("Options:\n");
    printf("    -in <input> -- input graph filename\n");
    printf("    -out <output> -- output filename. Default output is '<input>.v'\n");
    printf("    -root <root> -- root number for validation. Default is the first vertex\n");
    printf("    -nIters <nIters> -- number of iterations\n");
    exit(1);
}

void init (int argc, char** argv, graph_t* G)
{
    inFilename = outFilename = NULL;
    nIters = -1;
    rootNumberToValidate = 0;
    for (int i = 1; i < argc; ++i) {
   		if (!strcmp(argv[i], "-in")) {
            inFilename = argv[++i];
        }
   		if (!strcmp(argv[i], "-out")) {
            outFilename = argv[++i];
        }
		if (!strcmp(argv[i], "-root")) {
			rootNumberToValidate = (int) atoi(argv[++i]);
        }
		if (!strcmp(argv[i], "-nIters")) {
			nIters = (int) atoi(argv[++i]);
        }
    }
    if (!inFilename) usage(argc, argv);
    if (!outFilename) {
        outFilename = (char *)malloc((strlen(inFilename) + 3) * sizeof(char));
        sprintf(outFilename, "%s.v", inFilename);
    }
}

void readGraph(graph_t *G, char *filename)
{
    edge_id_t arity;
    uint8_t align;
    printf("readGraph...\n"); fflush(NULL);
    FILE *F = fopen(filename, "rb");
    if (!F) error(EXIT_FAILURE, 0, "Error in opening file %s", filename);

    size_t objects_read = 0;

    objects_read =  fread(&G->n, sizeof(vertex_id_t), 1, F);
    assert(objects_read == 1);
    G->scale = log(G->n) / log (2);   
    objects_read =  fread(&arity, sizeof(edge_id_t), 1, F);
    assert(objects_read == 1);
    objects_read =  fread(&G->directed, sizeof(bool), 1, F);
    assert(objects_read == 1);
    objects_read =  fread(&align, sizeof(uint8_t), 1, F);
    assert(objects_read == 1);

    G->m = G->n * arity;
    printf("m: %ld;\n",  G->m);

	G->rowsIndices = (edge_id_t *)malloc((G->n+1) * sizeof(edge_id_t));
    assert(G->rowsIndices);

	objects_read =  fread(G->rowsIndices, sizeof(edge_id_t), G->n+1, F);
    assert(objects_read == (G->n+1));
    printf("last rowsIndex: %ld\n", G->rowsIndices[G->n]);
    
    G->endV = (vertex_id_t *)malloc(G->rowsIndices[G->n] * sizeof(vertex_id_t));
    assert(G->endV);

    objects_read =  fread(G->endV, sizeof(vertex_id_t), G->rowsIndices[G->n], F);
    assert(objects_read == G->rowsIndices[G->n]);

    objects_read =  fread(&G->nRoots, sizeof(uint32_t), 1, F);
    assert(objects_read == 1);
    
    G->roots = (vertex_id_t *)malloc(G->nRoots * sizeof(vertex_id_t));
    assert(G->roots);
    G->numTraversedEdges = (edge_id_t *)malloc(G->nRoots * sizeof(edge_id_t));
    assert(G->numTraversedEdges);

    objects_read =  fread(G->roots, sizeof(vertex_id_t), G->nRoots, F);
    assert(objects_read == G->nRoots);
    objects_read =  fread(G->numTraversedEdges, sizeof(edge_id_t), G->nRoots, F);
    assert(objects_read == G->nRoots);

    if (nIters == -1) nIters = G->nRoots;

    G->weights = (weight_t *)malloc(G->m * sizeof(weight_t));
    assert(G->weights);

    objects_read =  fread(G->weights, sizeof(weight_t), G->m, F);
    assert(objects_read == G->m);

    fclose(F);
    printf("readGraph finished.\n"); fflush(NULL);
}

/* write distances from root vertex to each others to output file. -1 = infinity */
void writeDistance(char* filename, weight_t *dist, vertex_id_t n)
{
    FILE *F = fopen(filename, "wb");
    assert(fwrite(dist, sizeof(weight_t), n, F) == n);
    fclose(F);
}
/* write number of vertices at each level */
void writeLevels (char* filename, vertex_id_t *validateNLevels, int validateNLevelsLength)
{
    FILE *F = fopen(filename, "w"); 
    fprintf(F, "%d\n", validateNLevelsLength);
    for (int i = 0; i < validateNLevelsLength; ++i) 
        fprintf(F, "%d\n", validateNLevels[i]);
}


void freeGraph(graph_t *G)
{
    free(G->rowsIndices);
    free(G->endV);
    free(G->weights);
    free(G->roots);
    free(G->numTraversedEdges);
}

int main(int argc, char **argv) 
{
    weight_t* dist = NULL;
    graph_t g;
    struct timespec start_ts, finish_ts;
    double *perf;

    /* initializing and reading the graph */
    init(argc, argv, &g); 
    readGraph(&g, inFilename);
    
    init_sssp(&g);
    dist = (weight_t *)malloc(g.n * sizeof(weight_t));
    
    perf = (double *)malloc(g.nRoots * sizeof(double));
 
    nIters = (nIters < g.nRoots) ? nIters : g.nRoots; 
    
    printf("start algorithm iterations...\n");
    for (uint32_t i = 0; i < nIters; ++i) {
        /* initializing, -1 == infinity */
        for (vertex_id_t i = 0; i < g.n; i++) {
            dist[i] = -1;
        }
       
        uint64_t nedges;
        clock_gettime(CLOCK, &start_ts);
        sssp(g.roots[i], &g, dist, &nedges);
        clock_gettime(CLOCK, &finish_ts);
        double time =
            (finish_ts.tv_nsec - (double)start_ts.tv_nsec) * 1.0e-9 +
            (finish_ts.tv_sec - (double)start_ts.tv_sec);
        
        g.numTraversedEdges[i] = static_cast<edge_id_t>(nedges);
        perf[i] = g.numTraversedEdges[i] / (1000000 * time);

        if (rootNumberToValidate == i) {
            /* writing for validation */
            writeDistance(outFilename, dist, g.n);
        } 
    }
    printf("algorithm iterations finished.\n");

    /* final print */
    double min_perf, max_perf, avg_perf;
    int nNonZeroRoots = 0;
    max_perf = avg_perf = 0;
    min_perf = DBL_MAX;
    for (uint32_t i = 0; i < nIters; ++i) {
        /* we consider search only with large number of traversed edges. Random root can point to connected component with very small number of edges */
        if (g.numTraversedEdges[i] > SMALL_COMPONENT_EDGES_THRESHOLD) {
            ++nNonZeroRoots;
    	    avg_perf += perf[i];
            if (perf[i] < min_perf) min_perf = perf[i];
            if (perf[i] > max_perf) max_perf = perf[i];
        }
    }
    if (nNonZeroRoots != 0) avg_perf /= nNonZeroRoots;
    else error(EXIT_FAILURE, 0, "Number of roots with large traversed edges number is zero");

    printf(
        "%s: nIters = %d SSSP performance min = %.4f avg = %.4f max = %.4f MTEPS\n",
        inFilename, nNonZeroRoots, min_perf, avg_perf, max_perf
    );

    freeGraph(&g);
    finalize_sssp();

    return 0;
}
