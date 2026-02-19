/**
 * The most atomic way to train and run inference for a GPT in pure, dependency-free JavaScript.
 * This file is the complete algorithm.
 * Everything else is just efficiency.
 *
 * Ported from @karpathy's Python implementation.
 *
 * Usage (browser):
 *   <script src="gpt.js"></script>
 *   <script>runGPT().then(({samples}) => console.log(samples));</script>
 *
 * Usage (Node.js):
 *   node gpt.js
 */

// ---------------------------------------------------------------------------
// Seeded RNG (replaces Python's random with seed 42)
// ---------------------------------------------------------------------------
class SeededRandom {
    constructor(seed) {
        this.seed = seed >>> 0;
    }
    // Mulberry32
    next() {
        let t = (this.seed += 0x6D2B79F5);
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
    gauss(mu = 0, sigma = 1) {
        // Box-Muller transform
        const u = 1 - this.next();
        const v = this.next();
        return mu + sigma * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }
    choices(population, weights) {
        const total = weights.reduce((a, b) => a + b, 0);
        let r = this.next() * total;
        for (let i = 0; i < population.length; i++) {
            r -= weights[i];
            if (r <= 0) return population[i];
        }
        return population[population.length - 1];
    }
    shuffle(arr) {
        for (let i = arr.length - 1; i > 0; i--) {
            const j = Math.floor(this.next() * (i + 1));
            [arr[i], arr[j]] = [arr[j], arr[i]];
        }
    }
}

// ---------------------------------------------------------------------------
// Autograd: scalar Value node (micrograd-style)
// ---------------------------------------------------------------------------
class Value {
    constructor(data, children = [], localGrads = []) {
        this.data = data;
        this.grad = 0;
        this._children = children;
        this._localGrads = localGrads;
    }

    add(other) {
        other = other instanceof Value ? other : new Value(other);
        return new Value(this.data + other.data, [this, other], [1, 1]);
    }
    mul(other) {
        other = other instanceof Value ? other : new Value(other);
        return new Value(this.data * other.data, [this, other], [other.data, this.data]);
    }
    pow(exp) {
        return new Value(Math.pow(this.data, exp), [this], [exp * Math.pow(this.data, exp - 1)]);
    }
    log() {
        return new Value(Math.log(this.data), [this], [1 / this.data]);
    }
    exp() {
        const e = Math.exp(this.data);
        return new Value(e, [this], [e]);
    }
    relu() {
        return new Value(Math.max(0, this.data), [this], [this.data > 0 ? 1 : 0]);
    }
    neg()       { return this.mul(-1); }
    sub(other)  { return this.add((other instanceof Value ? other : new Value(other)).neg()); }
    div(other)  { return this.mul((other instanceof Value ? other : new Value(other)).pow(-1)); }

    backward() {
        const topo = [];
        const visited = new Set();
        const buildTopo = (v) => {
            if (!visited.has(v)) {
                visited.add(v);
                for (const child of v._children) buildTopo(child);
                topo.push(v);
            }
        };
        buildTopo(this);
        this.grad = 1;
        for (let i = topo.length - 1; i >= 0; i--) {
            const v = topo[i];
            for (let j = 0; j < v._children.length; j++) {
                v._children[j].grad += v._localGrads[j] * v.grad;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Neural network primitives
// ---------------------------------------------------------------------------
function linear(x, w) {
    return w.map(wo => wo.reduce((acc, wi, i) => acc.add(wi.mul(x[i])), new Value(0)));
}

function softmax(logits) {
    const maxVal = Math.max(...logits.map(v => v.data));
    const exps = logits.map(v => v.sub(maxVal).exp());
    const total = exps.reduce((a, b) => a.add(b), new Value(0));
    return exps.map(e => e.div(total));
}

function rmsnorm(x) {
    const ms = x.reduce((acc, xi) => acc.add(xi.mul(xi)), new Value(0)).mul(1 / x.length);
    const scale = ms.add(1e-5).pow(-0.5);
    return x.map(xi => xi.mul(scale));
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------
async function runGPT(options = {}) {
    const rng = new SeededRandom(42);

    // Hyperparameters (match Python defaults)
    const nLayer      = options.nLayer      ?? 1;
    const nEmbd       = options.nEmbd       ?? 16;
    const blockSize   = options.blockSize   ?? 16;
    const nHead       = options.nHead       ?? 4;
    const headDim     = nEmbd / nHead;
    const numSteps    = options.numSteps    ?? 1000;
    const numSamples  = options.numSamples  ?? 20;
    const temperature = options.temperature ?? 0.5;
    const log         = options.log         ?? console.log;

    // -------------------------------------------------------------------------
    // Dataset
    // -------------------------------------------------------------------------
    let docs;
    if (options.docs) {
        docs = options.docs;
    } else {
        const url = options.dataUrl ??
            'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt';
        log('Fetching dataset...');
        const resp = await fetch(url);
        const text = await resp.text();
        docs = text.split('\n').map(l => l.trim()).filter(l => l.length > 0);
    }
    rng.shuffle(docs);
    log(`num docs: ${docs.length}`);

    // Tokenizer: unique sorted characters become token ids 0..n-1, BOS = n
    const allChars  = [...new Set(docs.join(''))].sort();
    const BOS       = allChars.length;
    const vocabSize = allChars.length + 1;
    log(`vocab size: ${vocabSize}`);

    // -------------------------------------------------------------------------
    // Initialize parameters
    // -------------------------------------------------------------------------
    const makeMatrix = (nout, nin, std = 0.08) =>
        Array.from({ length: nout }, () =>
            Array.from({ length: nin }, () => new Value(rng.gauss(0, std))));

    const sd = {};
    sd['wte']     = makeMatrix(vocabSize, nEmbd);
    sd['wpe']     = makeMatrix(blockSize, nEmbd);
    sd['lm_head'] = makeMatrix(vocabSize, nEmbd);
    for (let i = 0; i < nLayer; i++) {
        sd[`layer${i}.attn_wq`] = makeMatrix(nEmbd, nEmbd);
        sd[`layer${i}.attn_wk`] = makeMatrix(nEmbd, nEmbd);
        sd[`layer${i}.attn_wv`] = makeMatrix(nEmbd, nEmbd);
        sd[`layer${i}.attn_wo`] = makeMatrix(nEmbd, nEmbd);
        sd[`layer${i}.mlp_fc1`] = makeMatrix(4 * nEmbd, nEmbd);
        sd[`layer${i}.mlp_fc2`] = makeMatrix(nEmbd, 4 * nEmbd);
    }
    const params = Object.values(sd).flatMap(mat => mat.flatMap(row => row));
    log(`num params: ${params.length}`);

    // -------------------------------------------------------------------------
    // GPT forward pass (token-by-token with KV cache)
    // -------------------------------------------------------------------------
    function gpt(tokenId, posId, keys, values) {
        // Token + position embedding
        const tokEmb = sd['wte'][tokenId];
        const posEmb = sd['wpe'][posId];
        let x = tokEmb.map((t, i) => t.add(posEmb[i]));
        x = rmsnorm(x); // not redundant: affects backward via residual

        for (let li = 0; li < nLayer; li++) {
            // 1) Multi-head Self-Attention
            const xResidual = x;
            x = rmsnorm(x);
            const q = linear(x, sd[`layer${li}.attn_wq`]);
            const k = linear(x, sd[`layer${li}.attn_wk`]);
            const v = linear(x, sd[`layer${li}.attn_wv`]);
            keys[li].push(k);
            values[li].push(v);

            const xAttn = [];
            for (let h = 0; h < nHead; h++) {
                const hs  = h * headDim;
                const q_h = q.slice(hs, hs + headDim);
                const k_h = keys[li].map(ki => ki.slice(hs, hs + headDim));
                const v_h = values[li].map(vi => vi.slice(hs, hs + headDim));
                const scale = 1 / Math.sqrt(headDim);
                const attnLogits = k_h.map(kt =>
                    q_h.reduce((acc, qj, j) => acc.add(qj.mul(kt[j])), new Value(0)).mul(scale)
                );
                const attnWeights = softmax(attnLogits);
                const headOut = Array.from({ length: headDim }, (_, j) =>
                    attnWeights.reduce((acc, aw, t) => acc.add(aw.mul(v_h[t][j])), new Value(0))
                );
                xAttn.push(...headOut);
            }

            x = linear(xAttn, sd[`layer${li}.attn_wo`]);
            x = x.map((xi, i) => xi.add(xResidual[i]));

            // 2) MLP block
            const xRes2 = x;
            x = rmsnorm(x);
            x = linear(x, sd[`layer${li}.mlp_fc1`]);
            x = x.map(xi => xi.relu());
            x = linear(x, sd[`layer${li}.mlp_fc2`]);
            x = x.map((xi, i) => xi.add(xRes2[i]));
        }

        return linear(x, sd['lm_head']);
    }

    // -------------------------------------------------------------------------
    // Adam optimizer buffers
    // -------------------------------------------------------------------------
    const lr    = 0.01, beta1 = 0.85, beta2 = 0.99, epsAdam = 1e-8;
    const mBuf  = new Float64Array(params.length);
    const vBuf  = new Float64Array(params.length);

    // -------------------------------------------------------------------------
    // Training loop
    // -------------------------------------------------------------------------
    for (let step = 0; step < numSteps; step++) {
        const doc    = docs[step % docs.length];
        const tokens = [BOS, ...Array.from(doc).map(ch => allChars.indexOf(ch)), BOS];
        const n      = Math.min(blockSize, tokens.length - 1);

        const keys   = Array.from({ length: nLayer }, () => []);
        const vals   = Array.from({ length: nLayer }, () => []);
        const losses = [];

        for (let posId = 0; posId < n; posId++) {
            const tokenId  = tokens[posId];
            const targetId = tokens[posId + 1];
            const logits = gpt(tokenId, posId, keys, vals);
            const probs  = softmax(logits);
            losses.push(probs[targetId].log().neg());
        }

        // Average loss over the sequence
        const loss = losses.reduce((a, b) => a.add(b), new Value(0)).mul(1 / n);

        // Backward pass
        loss.backward();

        // Adam parameter update with linear LR decay
        const lrT = lr * (1 - step / numSteps);
        for (let i = 0; i < params.length; i++) {
            const g    = params[i].grad;
            mBuf[i]    = beta1 * mBuf[i] + (1 - beta1) * g;
            vBuf[i]    = beta2 * vBuf[i] + (1 - beta2) * g * g;
            const mHat = mBuf[i] / (1 - Math.pow(beta1, step + 1));
            const vHat = vBuf[i] / (1 - Math.pow(beta2, step + 1));
            params[i].data -= lrT * mHat / (Math.sqrt(vHat) + epsAdam);
            params[i].grad  = 0;
        }

        log(`step ${String(step + 1).padStart(4)} / ${numSteps} | loss ${loss.data.toFixed(4)}`);

        // Yield to the browser event loop periodically to stay responsive
        if (typeof window !== 'undefined' && step % 50 === 49) {
            await new Promise(r => setTimeout(r, 0));
        }
    }

    // -------------------------------------------------------------------------
    // Inference: sample hallucinated names
    // -------------------------------------------------------------------------
    log('\n--- inference (new, hallucinated names) ---');
    const samples = [];
    for (let si = 0; si < numSamples; si++) {
        const keys = Array.from({ length: nLayer }, () => []);
        const vals = Array.from({ length: nLayer }, () => []);
        let tokenId = BOS;
        const sample = [];

        for (let posId = 0; posId < blockSize; posId++) {
            const logits      = gpt(tokenId, posId, keys, vals);
            const scaledLogits = logits.map(l => l.div(temperature));
            const probs       = softmax(scaledLogits);
            tokenId = rng.choices(
                Array.from({ length: vocabSize }, (_, i) => i),
                probs.map(p => p.data)
            );
            if (tokenId === BOS) break;
            sample.push(allChars[tokenId]);
        }

        const name = sample.join('');
        samples.push(name);
        log(`sample ${String(si + 1).padStart(2)}: ${name}`);
    }

    return { samples };
}

// ---------------------------------------------------------------------------
// Auto-run depending on environment
// ---------------------------------------------------------------------------
if (typeof window === 'undefined') {
    // Node.js: polyfill fetch with built-in https/http
    const https = require('https');
    const http  = require('http');
    global.fetch = (url) => new Promise((resolve, reject) => {
        const client = url.startsWith('https') ? https : http;
        client.get(url, (res) => {
            // follow redirects
            if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
                resolve(global.fetch(res.headers.location));
                return;
            }
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => resolve({ text: () => Promise.resolve(data) }));
        }).on('error', reject);
    });
    runGPT().catch(console.error);
} else {
    // Browser: expose globally
    window.runGPT = runGPT;
}
