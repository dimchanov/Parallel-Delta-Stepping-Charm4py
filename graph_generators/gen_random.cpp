#include "defs.h"
#include <stdio.h>
#include <cstdlib>
#include <assert.h>
#include <string.h>
#include <error.h>

using namespace std;

char outFilename[FNAME_LEN];

/* helper */
void usage(int argc, char **argv)
{
    printf("Random graph generator\n");
    printf("Usage:\n");
    printf("%s -s <scale> [other options]\n", argv[0]);
    printf("Options:\n");
    printf("   -s <scale>, number of vertices is 2^<scale>\n");
    printf("   -k <half the average vertex degree>, default value is 16\n");
    printf("   -nRoots <value> -- number of search root vertices. Default value is 10\n");
    printf("   -out <output filename>, file for the graph storage\n");
    exit(1);
}

/* initialization */
void init(int argc, char **argv, graph_t *G)
{
	bool no_out_filename = true;
    G->scale = -1;
    G->directed = false;
    G->permute_vertices = true;
    G->min_weight = 0;
    G->max_weight = 1;
    /* default value */
    G->nRoots = 10;
    G->avg_vertex_degree = DEFAULT_ARITY;
    if (argc == 1) {
        usage(argc, argv);
    }
    
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-s")) {
            G->scale = (int)atoi(argv[++i]);
        }
        if (!strcmp(argv[i], "-k")) {
            G->avg_vertex_degree = (int)atoi(argv[++i]);
        }
        if (!strcmp(argv[i], "-nRoots")) {
            G->nRoots = (uint32_t) atoi(argv[++i]);
        }
		if (!strcmp(argv[i], "-out")) {
            int l = strlen(argv[i]);
            strncpy(outFilename, argv[++i], (l > FNAME_LEN-1 ? FNAME_LEN-1 : l) );
            no_out_filename = false;
        }
    }
    
	if (no_out_filename) {
    	sprintf(outFilename, "random-%d", G->scale);
	}
	
    if (G->scale == -1) {
        usage(argc, argv);
    }
    
    G->n = (vertex_id_t)1 << G->scale;
    G->m = G->n * G->avg_vertex_degree;

    G->roots = (vertex_id_t *)malloc(G->nRoots * sizeof(vertex_id_t));
    assert(G->roots);
    G->numTraversedEdges = (edge_id_t *)malloc(G->nRoots * sizeof(edge_id_t));
    assert(G->numTraversedEdges);
    for (int i = 0; i < G->nRoots; ++i) {
        G->roots[i] = i; /* can be any index, but let it be i */
        G->numTraversedEdges[i] = 0; /* filled by sssp */
    }
}

/* random graph generator */
void gen_random_graph(graph_t *G)
{
    /* init */
    vertex_id_t n;
    edge_id_t m;
    edge_id_t offset;
    bool permute_vertices;
    vertex_id_t *permV, tmpVal;
    vertex_id_t u, v;
    vertex_id_t *src;
    vertex_id_t *dest;
    unsigned *degree;
    permute_vertices = G->permute_vertices;
    double *dbl_weight;
    double min_weight, max_weight;
    n = G->n;
    m = G->m;
    src = new vertex_id_t[m];
    assert(src != NULL);
    dest = new vertex_id_t[m];
    assert(dest != NULL);
    degree = new unsigned[n];
    assert(degree != NULL);
    memset(degree, 0, sizeof(unsigned) * n);
    
    dbl_weight = (double *) malloc(m * sizeof(double));
    assert(dbl_weight != NULL);

    srand48(2387);
    
    /* generate edges */
    for (edge_id_t i = 0; i < m; i++) {
        vertex_id_t u = rand() % n;
        vertex_id_t v = rand() % n;
        src[i] = u;
        dest[i] = v;
    }
    
    /* reshuffle */
    if (permute_vertices) {
        srand48(4791);
        permV = new vertex_id_t[n];
        assert(permV != NULL);
        
        for (vertex_id_t i = 0; i < n; i++) {
            permV[i] = i;
        }
        
        for (vertex_id_t i = 0; i < n; i++) {
            vertex_id_t j = n * drand48();
            tmpVal = permV[i];
            permV[i] = permV[j];
            permV[j] = tmpVal;
        }
        
        for (edge_id_t i = 0; i < m; i++) {
            src[i] = permV[src[i]];
            dest[i] = permV[dest[i]];
        }
        
        delete[] permV;
    }
    
    min_weight = G->min_weight;
    max_weight = G->max_weight;

    /* Generate edge weights */
    for (edge_id_t i=0; i<m; i++) {
        dbl_weight[i]  = min_weight + (max_weight-min_weight)*drand48();
    }

    /* update graph data structure */
    for (edge_id_t i = 0; i < m; i++) {
        degree[src[i]]++;
        degree[dest[i]]++;
    }
    
    G->endV = new vertex_id_t[2 * m];
    assert(G->endV != NULL);
    
    G->rowsIndices = new edge_id_t[n + 1];
    assert(G->rowsIndices != NULL);
    
    G->n = n;
    /* undirected graph, each edge is stored twice; if edge is (u, v), then it's
     * stored at the vertex u and at the vertex v */
    G->m = 2 * m;
    G->directed = false;
    
    G->weights = (double *) malloc(G->m * sizeof(double));       
    assert(G->weights != NULL);

    G->rowsIndices[0] = 0;
    for (vertex_id_t i = 1; i <= G->n; i++) {
        G->rowsIndices[i] = G->rowsIndices[i - 1] + degree[i - 1];
    }
    
    for (edge_id_t i = 0; i < m; i++) {
        u = src[i];
        v = dest[i];
        offset = degree[u]--;
        G->endV[G->rowsIndices[u] + offset - 1] = v;
        G->weights[G->rowsIndices[u]+offset-1] = dbl_weight[i];
        offset = degree[v]--;
        G->endV[G->rowsIndices[v] + offset - 1] = u;
        G->weights[G->rowsIndices[v]+offset-1] = dbl_weight[i];

    }
    
    delete[] src;
    delete[] dest;
    delete[] degree;
}

/* write graph to file */
void writeGraph(graph_t *G, char *filename)
{
    FILE *F = fopen(filename, "wb");
    if (!F) error(EXIT_FAILURE, 0, "Error in opening file %s", filename);
	size_t objects_written = 0;

    objects_written = fwrite(&G->n, sizeof(vertex_id_t), 1, F);
    assert(objects_written ==  1);
    
    edge_id_t arity = G->m / G->n;
    objects_written = fwrite(&arity, sizeof(edge_id_t), 1, F);
    assert(objects_written ==  1);
    objects_written = fwrite(&G->directed, sizeof(bool), 1, F);
    assert(objects_written ==  1);
    uint8_t align = 0;
    objects_written = fwrite(&align, sizeof(uint8_t), 1, F);
    assert(objects_written ==  1);

    objects_written = fwrite(G->rowsIndices, sizeof(edge_id_t), G->n+1, F);
    assert(objects_written ==  G->n+1);
    objects_written = fwrite(G->endV, sizeof(vertex_id_t), G->rowsIndices[G->n], F);
    assert(objects_written ==  G->rowsIndices[G->n]);
    
    objects_written = fwrite(&G->nRoots, sizeof(uint32_t), 1, F);
    assert(objects_written ==  1);
    objects_written = fwrite(G->roots, sizeof(vertex_id_t), G->nRoots, F);
    assert(objects_written ==  G->nRoots);
    objects_written = fwrite(G->numTraversedEdges, sizeof(edge_id_t), G->nRoots, F);
    assert(objects_written ==  G->nRoots);

    objects_written = fwrite(G->weights, sizeof(weight_t), G->m, F);
    assert(objects_written ==  G->m);
    fclose(F);
}

/* print graph */
void printGraph(graph_t *G)
{
	int i,j;
	for (i = 0; i < (int)G->n; ++i) {
		printf("%d:", i);
		for (j=G->rowsIndices[i]; j < (int)G->rowsIndices[i+1]; ++j)
			printf("%d (%f), ", G->endV[j], G->weights[j]);
		printf("\n");
	}
}

int main(int argc, char **argv) {
    graph_t g;
    init(argc, argv, &g);
    gen_random_graph(&g);
    //printGraph(&g);
    writeGraph(&g, outFilename);
    
    return 0;
}
