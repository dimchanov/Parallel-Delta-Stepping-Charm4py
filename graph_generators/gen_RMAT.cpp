#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <error.h>
#include "defs.h"

char outFilename[FNAME_LEN];

void usage(int argc, char **argv) 
{
	printf("Usage:\n");
	printf("    %s -s <scale> [options]\n", argv[0]);
    printf("Options:\n");
    printf("   -s <scale>, number of vertices is 2^<scale>\n");
    printf("   -k <half the average vertex degree>, default value is 16\n");
    printf("   -nRoots <value> -- number of search root vertices. Default value is 10\n");
    printf("   -out <filename>, rmat-<s> is default value\n");
    exit(1);
}

void init (int argc, char** argv, graph_t* G)
{
	bool no_out_filename = true;
    G->scale = -1;
    G->directed = false;
    G->a = 0.45;
    G->b = 0.25;
    G->c = 0.15;
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
   		if (!strcmp(argv[i], "-out")) {
            int l = strlen(argv[i]);
            strncpy(outFilename, argv[++i], (l > FNAME_LEN-1 ? FNAME_LEN-1 : l) );
            no_out_filename = false;
        }
    }

    if (no_out_filename) {
        sprintf(outFilename, "rmat-%d", G->scale);
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

/* function is adapted from SNAP
 * http://snap-graph.sourceforge.net/ */
void gen_RMAT_graph(graph_t* G) 
{
    edge_id_t i, j;
    bool undirected;
    vertex_id_t n;
    edge_id_t m;
    edge_id_t offset;
    double a, b, c, d;
    double av, bv, cv, dv, S, p;
    int SCALE;
    double var;
    vertex_id_t step;
    double *dbl_weight;
    double min_weight, max_weight;
    bool permute_vertices;
    vertex_id_t* permV, tmpVal;
    vertex_id_t u, v;
    int seed;
    vertex_id_t *src;
    vertex_id_t *dest;
    uint32_t *degree;

    a = G->a;
    b = G->b;
    c = G->c;
    assert(a+b+c < 1);
    d = 1  - (a+b+c);

    permute_vertices = G->permute_vertices;

    undirected = !G->directed;
    n = G->n;
    m = G->m;

    src = (vertex_id_t *) malloc (m * sizeof(vertex_id_t));
    dest = (vertex_id_t *) malloc(m * sizeof(vertex_id_t));
    degree = (uint32_t *) calloc(n, sizeof(uint32_t));

    assert(src != NULL);
    assert(dest != NULL);
    assert(degree != NULL);

    dbl_weight = (double *) malloc(m * sizeof(double));
    assert(dbl_weight != NULL);

    /* Initialize RNG stream */
    seed = 2387;
    srand48(seed);
    SCALE = G->scale;

    /* Generate edges */
    for (i=0; i<m; i++) {

        u = 1;
        v = 1;
        step = n/2;

        av = a;
        bv = b;
        cv = c;
        dv = d;

        p = drand48();
        if (p < av) {
            /* Do nothing */
        } else if ((p >= av) && (p < av+bv)) {
            v += step;
        } else if ((p >= av+bv) && (p < av+bv+cv)) {
            u += step;
        } else {
            u += step;
            v += step;
        }

        for (j=1; j<(unsigned)SCALE; j++) {
            step = step/2;

            /* Vary a,b,c,d by up to 10% */
            var = 0.1;
            av *= 0.95 + var * drand48();
            bv *= 0.95 + var * drand48();
            cv *= 0.95 + var * drand48();
            dv *= 0.95 + var * drand48();

            S = av + bv + cv + dv;
            av = av/S;
            bv = bv/S;
            cv = cv/S;
            dv = dv/S;

            /* Choose partition */
            p = drand48();
            if (p < av) {
                /* Do nothing */
            } else if ((p >= av) && (p < av+bv)) {
                v += step;
            } else if ((p >= av+bv) && (p < av+bv+cv)) {
                u += step;
            } else {
                u += step;
                v += step;
            }
        }

        src[i] = u-1;
        dest[i] = v-1;
    }

    if (permute_vertices) {
        permV = (vertex_id_t *) malloc(n*sizeof(vertex_id_t));
        assert(permV != NULL);

        for (i=0; i<n; i++) {
            permV[i] = i;
        }

        for (i=0; i<n; i++) {
            j = n * drand48();
            tmpVal = permV[i];
            permV[i] = permV[j];
            permV[j] = tmpVal;
        }

        for (i=0; i<m; i++) {
            src[i]  = permV[src[i]];
            dest[i] = permV[dest[i]];
        }
        free(permV);
    }

    for (i=0; i<m; i++) {
        degree[src[i]]++;
        if (undirected)
            degree[dest[i]]++;
    }

    min_weight = G->min_weight;
    max_weight = G->max_weight;

    /* Generate edge weights */
    for (i=0; i<m; i++) {
        dbl_weight[i]  = min_weight + (max_weight-min_weight)*drand48();
    }

    /* Update graph data structure */
    if (undirected) {
        G->endV = (vertex_id_t *) malloc(2*m * sizeof(vertex_id_t));
    } else { 
        G->endV = (vertex_id_t *) malloc(m * sizeof(vertex_id_t));
    }

    assert(G->endV != NULL);

    G->rowsIndices = (edge_id_t *) malloc((n+1)*sizeof(edge_id_t));
    assert(G->rowsIndices != NULL);

    G->n = n;
    if (undirected)
        G->m = 2*m;
    else
        G->m = m;

    G->weights = (double *) malloc(G->m * sizeof(double));       
    assert(G->weights != NULL);

    G->rowsIndices[0] = 0; 
    for (i=1; i<=G->n; i++) {
        G->rowsIndices[i] = G->rowsIndices[i-1] + degree[i-1];
    }

    for (i=0; i<m; i++) {
        u = src[i];
        v = dest[i];
        offset = degree[u]--;
        G->endV[G->rowsIndices[u]+offset-1] = v;
        G->weights[G->rowsIndices[u]+offset-1] = dbl_weight[i];

        if (undirected) {
            offset = degree[v]--;
            G->endV[G->rowsIndices[v]+offset-1] = u;
            G->weights[G->rowsIndices[v]+offset-1] = dbl_weight[i];
        }
    } 

    free(src);
    free(dest);
    free(degree);

    free(dbl_weight);
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

int main (int argc, char** argv)
{
    graph_t g;
    init(argc, argv, &g);
    gen_RMAT_graph(&g);
    //printGraph(&g);
    writeGraph(&g, outFilename);
    return 0;
}
