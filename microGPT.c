/*
 * The most atomic way to train and run inference for a GPT in pure C.
 * Port of @karpathy's Python script. Compile with:
 *   gcc -O2 -lm -o gpt gpt.c
 * @karpathy (Python) -> C port
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* -------------------------------------------------------------------------
 * Config
 * ------------------------------------------------------------------------- */
#define N_LAYER     1
#define N_EMBD      16
#define BLOCK_SIZE  16
#define N_HEAD      4
#define HEAD_DIM    (N_EMBD / N_HEAD)
#define NUM_STEPS   1000
#define MAX_DOCS    50000
#define MAX_DOC_LEN 64
#define MAX_VOCAB   128   /* ASCII chars + 1 BOS */
#define MAX_PARAMS  131072

/* -------------------------------------------------------------------------
 * Simple LCG RNG (seeded deterministically, like random.seed(42))
 * ------------------------------------------------------------------------- */
static unsigned long long rng_state = 42;
static double rng_uniform(void) {
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)((rng_state >> 33) & 0x7FFFFFFF) / (double)0x7FFFFFFF;
}
/* Box-Muller for Gaussian */
static double rng_gauss(double std) {
    double u = rng_uniform() + 1e-12;
    double v = rng_uniform();
    return std * sqrt(-2.0 * log(u)) * cos(2.0 * 3.14159265358979323846 * v);
}

/* -------------------------------------------------------------------------
 * Autograd Value
 * ------------------------------------------------------------------------- */
typedef struct Value Value;
struct Value {
    double data;
    double grad;
    /* up to 2 children */
    Value *children[2];
    double local_grads[2];
    int n_children;
};

/* We use a flat pool allocator to avoid malloc overhead per node */
#define POOL_SIZE (1 << 24)  /* 16M nodes should be enough for one step */
static Value pool[POOL_SIZE];
static int pool_top = 0;

static Value *new_val(double data, Value *c0, Value *c1, double lg0, double lg1, int nc) {
    if (pool_top >= POOL_SIZE) { fprintf(stderr, "pool overflow\n"); exit(1); }
    Value *v = &pool[pool_top++];
    v->data = data;
    v->grad = 0.0;
    v->n_children = nc;
    if (nc >= 1) { v->children[0] = c0; v->local_grads[0] = lg0; }
    if (nc >= 2) { v->children[1] = c1; v->local_grads[1] = lg1; }
    return v;
}

static Value *val_add(Value *a, Value *b) {
    return new_val(a->data + b->data, a, b, 1.0, 1.0, 2);
}
static Value *val_mul(Value *a, Value *b) {
    return new_val(a->data * b->data, a, b, b->data, a->data, 2);
}
static Value *val_mul_scalar(Value *a, double s) {
    /* treat scalar as leaf with no grad */
    Value *tmp = new_val(s, NULL, NULL, 0, 0, 0);
    return val_mul(a, tmp);
}
static Value *val_pow(Value *a, double exp) {
    return new_val(pow(a->data, exp), a, NULL, exp * pow(a->data, exp - 1.0), 0, 1);
}
static Value *val_log(Value *a) {
    return new_val(log(a->data), a, NULL, 1.0 / a->data, 0, 1);
}
static Value *val_exp(Value *a) {
    double e = exp(a->data);
    return new_val(e, a, NULL, e, 0, 1);
}
static Value *val_relu(Value *a) {
    return new_val(a->data > 0 ? a->data : 0.0, a, NULL, a->data > 0 ? 1.0 : 0.0, 0, 1);
}
static Value *val_sub(Value *a, Value *b) {
    Value *nb = val_mul_scalar(b, -1.0);
    return val_add(a, nb);
}
static Value *val_div(Value *a, Value *b) {
    Value *inv = val_pow(b, -1.0);
    return val_mul(a, inv);
}

/* Topological sort + backward */
/* We do iterative DFS to avoid stack overflow */
#define TOPO_SIZE POOL_SIZE
static Value **topo;
static int topo_n;
static char *visited_flags; /* byte array indexed by pool index */

static void build_topo(Value *root) {
    /* iterative post-order DFS */
    topo_n = 0;
    /* use a stack */
    static Value *stack[POOL_SIZE];
    static char on_stack[POOL_SIZE];
    int sp = 0;
    stack[sp++] = root;
    while (sp > 0) {
        Value *v = stack[sp - 1];
        int idx = (int)(v - pool);
        if (visited_flags[idx]) { sp--; continue; }
        /* push children first if not visited */
        int pushed = 0;
        for (int i = 0; i < v->n_children; i++) {
            Value *c = v->children[i];
            if (c && !visited_flags[(int)(c - pool)]) {
                stack[sp++] = c;
                pushed = 1;
            }
        }
        if (!pushed) {
            visited_flags[idx] = 1;
            topo[topo_n++] = v;
            sp--;
        }
    }
}

static void backward(Value *loss) {
    memset(visited_flags, 0, pool_top);
    build_topo(loss);
    loss->grad = 1.0;
    for (int i = topo_n - 1; i >= 0; i--) {
        Value *v = topo[i];
        for (int j = 0; j < v->n_children; j++) {
            if (v->children[j])
                v->children[j]->grad += v->local_grads[j] * v->grad;
        }
    }
}

/* -------------------------------------------------------------------------
 * State dict: flat parameter matrices
 * We store pointers to Value* for each matrix
 * ------------------------------------------------------------------------- */

/* Matrix: nout x nin */
typedef struct {
    Value **data; /* row-major: data[r*nin + c] */
    int nout, nin;
} Matrix;

static Matrix make_matrix(int nout, int nin, double std) {
    Matrix m;
    m.nout = nout; m.nin = nin;
    m.data = (Value **)malloc(nout * nin * sizeof(Value *));
    for (int i = 0; i < nout * nin; i++) {
        m.data[i] = new_val(rng_gauss(std), NULL, NULL, 0, 0, 0);
    }
    return m;
}

/* GPT parameters */
static Matrix wte, wpe, lm_head;
static Matrix attn_wq[N_LAYER], attn_wk[N_LAYER], attn_wv[N_LAYER], attn_wo[N_LAYER];
static Matrix mlp_fc1[N_LAYER], mlp_fc2[N_LAYER];

/* Flat params list for optimizer */
static Value **params_list;
static int num_params;

static void collect_matrix_params(Matrix *m) {
    for (int i = 0; i < m->nout * m->nin; i++)
        params_list[num_params++] = m->data[i];
}

/* -------------------------------------------------------------------------
 * Model forward helpers (operate on Value* arrays allocated from pool)
 * We allocate temporary arrays on the C stack or as static locals.
 * For variable-length arrays we use small fixed sizes since dims are known.
 * ------------------------------------------------------------------------- */

static void linear(Value **x, int xlen, Matrix *w, Value **out) {
    /* out[i] = sum_j w[i][j] * x[j] */
    for (int i = 0; i < w->nout; i++) {
        Value *acc = new_val(0.0, NULL, NULL, 0, 0, 0);
        for (int j = 0; j < w->nin; j++) {
            acc = val_add(acc, val_mul(w->data[i * w->nin + j], x[j]));
        }
        out[i] = acc;
    }
}

static void softmax(Value **logits, int n, Value **out) {
    double max_val = logits[0]->data;
    for (int i = 1; i < n; i++) if (logits[i]->data > max_val) max_val = logits[i]->data;
    Value *max_v = new_val(max_val, NULL, NULL, 0, 0, 0);
    Value **exps = (Value **)alloca(n * sizeof(Value *));
    Value *total = new_val(0.0, NULL, NULL, 0, 0, 0);
    for (int i = 0; i < n; i++) {
        exps[i] = val_exp(val_sub(logits[i], max_v));
        total = val_add(total, exps[i]);
    }
    for (int i = 0; i < n; i++) out[i] = val_div(exps[i], total);
}

static void rmsnorm(Value **x, int n, Value **out) {
    Value *ms = new_val(0.0, NULL, NULL, 0, 0, 0);
    for (int i = 0; i < n; i++) ms = val_add(ms, val_mul(x[i], x[i]));
    /* ms / n */
    ms = val_mul_scalar(ms, 1.0 / n);
    /* scale = (ms + 1e-5)^-0.5 */
    Value *ms_eps = val_add(ms, new_val(1e-5, NULL, NULL, 0, 0, 0));
    Value *scale = val_pow(ms_eps, -0.5);
    for (int i = 0; i < n; i++) out[i] = val_mul(x[i], scale);
}

/* KV cache per layer */
/* Each position appends k/v vectors. We store pointers. */
#define MAX_SEQ BLOCK_SIZE
static Value *keys_cache[N_LAYER][MAX_SEQ][N_EMBD];
static Value *vals_cache[N_LAYER][MAX_SEQ][N_EMBD];
static int kv_len[N_LAYER];

static void gpt_forward(int token_id, int pos_id, Value **logits_out) {
    /* token + position embedding */
    Value *x[N_EMBD];
    for (int i = 0; i < N_EMBD; i++)
        x[i] = val_add(wte.data[token_id * N_EMBD + i], wpe.data[pos_id * N_EMBD + i]);

    /* initial rmsnorm */
    Value *xn[N_EMBD];
    rmsnorm(x, N_EMBD, xn);
    memcpy(x, xn, sizeof(xn));

    for (int li = 0; li < N_LAYER; li++) {
        /* --- Attention --- */
        Value *x_res[N_EMBD];
        memcpy(x_res, x, sizeof(x));
        Value *xrn[N_EMBD];
        rmsnorm(x, N_EMBD, xrn);

        Value *q[N_EMBD], *k[N_EMBD], *v[N_EMBD];
        linear(xrn, N_EMBD, &attn_wq[li], q);
        linear(xrn, N_EMBD, &attn_wk[li], k);
        linear(xrn, N_EMBD, &attn_wv[li], v);

        /* store in kv cache */
        int t = kv_len[li];
        for (int i = 0; i < N_EMBD; i++) {
            keys_cache[li][t][i] = k[i];
            vals_cache[li][t][i] = v[i];
        }
        kv_len[li]++;
        int seq_len = kv_len[li];

        /* multi-head attention */
        Value *x_attn[N_EMBD];
        for (int h = 0; h < N_HEAD; h++) {
            int hs = h * HEAD_DIM;
            /* attn logits */
            Value *attn_logits[MAX_SEQ];
            double scale = 1.0 / sqrt((double)HEAD_DIM);
            for (int tt = 0; tt < seq_len; tt++) {
                Value *dot = new_val(0.0, NULL, NULL, 0, 0, 0);
                for (int j = 0; j < HEAD_DIM; j++) {
                    dot = val_add(dot, val_mul(q[hs + j], keys_cache[li][tt][hs + j]));
                }
                attn_logits[tt] = val_mul_scalar(dot, scale);
            }
            Value *attn_w[MAX_SEQ];
            softmax(attn_logits, seq_len, attn_w);
            /* weighted sum over values */
            for (int j = 0; j < HEAD_DIM; j++) {
                Value *acc = new_val(0.0, NULL, NULL, 0, 0, 0);
                for (int tt = 0; tt < seq_len; tt++) {
                    acc = val_add(acc, val_mul(attn_w[tt], vals_cache[li][tt][hs + j]));
                }
                x_attn[hs + j] = acc;
            }
        }

        Value *x_proj[N_EMBD];
        linear(x_attn, N_EMBD, &attn_wo[li], x_proj);
        for (int i = 0; i < N_EMBD; i++) x[i] = val_add(x_proj[i], x_res[i]);

        /* --- MLP --- */
        memcpy(x_res, x, sizeof(x));
        Value *xrn2[N_EMBD];
        rmsnorm(x, N_EMBD, xrn2);

        Value *fc1_out[4 * N_EMBD];
        linear(xrn2, N_EMBD, &mlp_fc1[li], fc1_out);
        for (int i = 0; i < 4 * N_EMBD; i++) fc1_out[i] = val_relu(fc1_out[i]);

        Value *fc2_out[N_EMBD];
        linear(fc1_out, 4 * N_EMBD, &mlp_fc2[li], fc2_out);
        for (int i = 0; i < N_EMBD; i++) x[i] = val_add(fc2_out[i], x_res[i]);
    }

    /* lm_head */
    linear(x, N_EMBD, &lm_head, logits_out);
}

/* -------------------------------------------------------------------------
 * Dataset loading
 * ------------------------------------------------------------------------- */
static char docs[MAX_DOCS][MAX_DOC_LEN];
static int num_docs = 0;
static char uchars[MAX_VOCAB];
static int vocab_size;
static int BOS;

static int char_to_id(char c) {
    for (int i = 0; i < vocab_size - 1; i++)
        if (uchars[i] == c) return i;
    return -1;
}

/* Fisher-Yates shuffle of docs */
static void shuffle_docs(void) {
    for (int i = num_docs - 1; i > 0; i--) {
        int j = (int)(rng_uniform() * (i + 1));
        char tmp[MAX_DOC_LEN];
        memcpy(tmp, docs[i], MAX_DOC_LEN);
        memcpy(docs[i], docs[j], MAX_DOC_LEN);
        memcpy(docs[j], tmp, MAX_DOC_LEN);
    }
}

/* -------------------------------------------------------------------------
 * Main
 * ------------------------------------------------------------------------- */
int main(void) {
    /* --- Load dataset --- */
    FILE *f = fopen("input.txt", "r");
    if (!f) {
        fprintf(stderr, "input.txt not found. Download it first:\n");
        fprintf(stderr, "  curl -o input.txt https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt\n");
        return 1;
    }
    char line[MAX_DOC_LEN];
    while (fgets(line, sizeof(line), f) && num_docs < MAX_DOCS) {
        int len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = '\0';
        if (len > 0) {
            strncpy(docs[num_docs++], line, MAX_DOC_LEN - 1);
        }
    }
    fclose(f);
    printf("num docs: %d\n", num_docs);

    /* Build vocabulary */
    char seen[256] = {0};
    int nuc = 0;
    for (int d = 0; d < num_docs; d++)
        for (int i = 0; docs[d][i]; i++)
            seen[(unsigned char)docs[d][i]] = 1;
    for (int c = 0; c < 256; c++)
        if (seen[c]) uchars[nuc++] = (char)c;
    /* uchars is already sorted since we iterate c=0..255 */
    BOS = nuc;
    vocab_size = nuc + 1;
    printf("vocab size: %d\n", vocab_size);

    shuffle_docs();

    /* --- Allocate topo/visited arrays --- */
    topo = (Value **)malloc(POOL_SIZE * sizeof(Value *));
    visited_flags = (char *)malloc(POOL_SIZE);

    /* --- Initialize parameters --- */
    /* We must initialize params before pool is used for computations.
     * Parameters are permanent nodes; we'll reset pool_top after each step
     * but keep params at the beginning. */

    /* Reserve space for params at pool bottom */
    wte  = make_matrix(vocab_size, N_EMBD, 0.08);
    wpe  = make_matrix(BLOCK_SIZE, N_EMBD, 0.08);
    lm_head = make_matrix(vocab_size, N_EMBD, 0.08);
    for (int li = 0; li < N_LAYER; li++) {
        attn_wq[li] = make_matrix(N_EMBD, N_EMBD, 0.08);
        attn_wk[li] = make_matrix(N_EMBD, N_EMBD, 0.08);
        attn_wv[li] = make_matrix(N_EMBD, N_EMBD, 0.08);
        attn_wo[li] = make_matrix(N_EMBD, N_EMBD, 0.08);
        mlp_fc1[li] = make_matrix(4 * N_EMBD, N_EMBD, 0.08);
        mlp_fc2[li] = make_matrix(N_EMBD, 4 * N_EMBD, 0.08);
    }

    /* Collect params */
    params_list = (Value **)malloc(MAX_PARAMS * sizeof(Value *));
    num_params = 0;
    collect_matrix_params(&wte);
    collect_matrix_params(&wpe);
    collect_matrix_params(&lm_head);
    for (int li = 0; li < N_LAYER; li++) {
        collect_matrix_params(&attn_wq[li]);
        collect_matrix_params(&attn_wk[li]);
        collect_matrix_params(&attn_wv[li]);
        collect_matrix_params(&attn_wo[li]);
        collect_matrix_params(&mlp_fc1[li]);
        collect_matrix_params(&mlp_fc2[li]);
    }
    printf("num params: %d\n", num_params);

    int params_pool_end = pool_top; /* everything below this is a param node */

    /* --- Adam buffers --- */
    double *m_buf = (double *)calloc(num_params, sizeof(double));
    double *v_buf = (double *)calloc(num_params, sizeof(double));
    double lr = 0.01, beta1 = 0.85, beta2 = 0.99, eps_adam = 1e-8;

    /* --- Training loop --- */
    for (int step = 0; step < NUM_STEPS; step++) {
        /* Reset computation graph (keep param nodes) */
        pool_top = params_pool_end;
        memset(visited_flags, 0, pool_top); /* clear visited flags for param range */

        /* Reset param grads */
        for (int i = 0; i < num_params; i++) params_list[i]->grad = 0.0;

        /* Get document */
        const char *doc = docs[step % num_docs];
        int doc_len = strlen(doc);
        /* tokens: BOS + chars + BOS */
        int tokens[MAX_DOC_LEN + 2];
        int tlen = 0;
        tokens[tlen++] = BOS;
        for (int i = 0; i < doc_len; i++) tokens[tlen++] = char_to_id(doc[i]);
        tokens[tlen++] = BOS;
        int n = tlen - 1;
        if (n > BLOCK_SIZE) n = BLOCK_SIZE;

        /* Reset KV cache */
        memset(kv_len, 0, sizeof(kv_len));

        /* Forward */
        Value *loss_sum = new_val(0.0, NULL, NULL, 0, 0, 0);
        for (int pos = 0; pos < n; pos++) {
            int token_id = tokens[pos];
            int target_id = tokens[pos + 1];
            Value *logits[MAX_VOCAB];
            gpt_forward(token_id, pos, logits);
            Value *probs[MAX_VOCAB];
            softmax(logits, vocab_size, probs);
            Value *loss_t = val_mul_scalar(val_log(probs[target_id]), -1.0);
            loss_sum = val_add(loss_sum, loss_t);
        }
        Value *loss = val_mul_scalar(loss_sum, 1.0 / n);

        /* Backward */
        backward(loss);

        /* Adam update */
        double lr_t = lr * (1.0 - (double)step / NUM_STEPS);
        for (int i = 0; i < num_params; i++) {
            double g = params_list[i]->grad;
            m_buf[i] = beta1 * m_buf[i] + (1.0 - beta1) * g;
            v_buf[i] = beta2 * v_buf[i] + (1.0 - beta2) * g * g;
            double m_hat = m_buf[i] / (1.0 - pow(beta1, step + 1));
            double v_hat = v_buf[i] / (1.0 - pow(beta2, step + 1));
            params_list[i]->data -= lr_t * m_hat / (sqrt(v_hat) + eps_adam);
        }

        printf("step %4d / %4d | loss %.4f\r", step + 1, NUM_STEPS, loss->data);
        fflush(stdout);
    }
    printf("\n");

    /* --- Inference --- */
    double temperature = 0.5;
    printf("--- inference (new, hallucinated names) ---\n");
    for (int sample_idx = 0; sample_idx < 20; sample_idx++) {
        pool_top = params_pool_end;
        memset(kv_len, 0, sizeof(kv_len));

        int token_id = BOS;
        char sample[MAX_DOC_LEN];
        int slen = 0;

        for (int pos = 0; pos < BLOCK_SIZE; pos++) {
            Value *logits[MAX_VOCAB];
            gpt_forward(token_id, pos, logits);
            /* scale by temperature */
            Value *scaled[MAX_VOCAB];
            for (int i = 0; i < vocab_size; i++)
                scaled[i] = val_mul_scalar(logits[i], 1.0 / temperature);
            Value *probs[MAX_VOCAB];
            softmax(scaled, vocab_size, probs);

            /* sample from distribution */
            double r = rng_uniform();
            double cum = 0.0;
            token_id = vocab_size - 1; /* fallback */
            for (int i = 0; i < vocab_size; i++) {
                cum += probs[i]->data;
                if (r < cum) { token_id = i; break; }
            }
            if (token_id == BOS) break;
            if (slen < MAX_DOC_LEN - 1) sample[slen++] = uchars[token_id];
        }
        sample[slen] = '\0';
        printf("sample %2d: %s\n", sample_idx + 1, sample);
    }

    return 0;
}
