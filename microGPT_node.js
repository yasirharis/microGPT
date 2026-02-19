/**
 * The most atomic way to train and run inference for a GPT in pure JS.
 * Port of @karpathy's Python script. No dependencies.
 *
 * Run with:  node gpt.js
 * (downloads input.txt automatically via https if not present)
 */

'use strict';

const fs   = require('fs');
const path = require('path');
const https = require('https');

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const N_LAYER    = 1;
const N_EMBD     = 16;
const BLOCK_SIZE = 16;
const N_HEAD     = 4;
const HEAD_DIM   = N_EMBD / N_HEAD;
const NUM_STEPS  = 1000;

// ---------------------------------------------------------------------------
// Deterministic RNG (LCG, seed=42) — mirrors Python random.seed(42) spirit
// ---------------------------------------------------------------------------
let rngState = 42n;
function rngUniform() {
  rngState = (rngState * 6364136223846793005n + 1442695040888963407n) & 0xFFFFFFFFFFFFFFFFn;
  return Number((rngState >> 33n) & 0x7FFFFFFFn) / 0x7FFFFFFF;
}
function rngGauss(std) {
  const u = rngUniform() + 1e-12;
  const v = rngUniform();
  return std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function rngChoices(n, weights) {
  const r = rngUniform();
  let cum = 0;
  for (let i = 0; i < n; i++) {
    cum += weights[i];
    if (r < cum) return i;
  }
  return n - 1;
}

// ---------------------------------------------------------------------------
// Autograd — scalar Value nodes with lazy backward
// ---------------------------------------------------------------------------

// We use a flat typed pool for speed.
// Each node stores: data, grad, child0_idx, child1_idx, lg0, lg1, n_children
// We keep JS objects for simplicity but reuse a pool array.
const POOL_SIZE = 1 << 24; // 16M nodes
const poolData  = new Float64Array(POOL_SIZE);
const poolGrad  = new Float64Array(POOL_SIZE);
const poolLg0   = new Float64Array(POOL_SIZE);
const poolLg1   = new Float64Array(POOL_SIZE);
const poolC0    = new Int32Array(POOL_SIZE);   // child index, -1 = none
const poolC1    = new Int32Array(POOL_SIZE);
const poolNC    = new Uint8Array(POOL_SIZE);   // n_children

poolC0.fill(-1);
poolC1.fill(-1);

let poolTop = 0;

function newVal(data, c0, c1, lg0, lg1, nc) {
  const i = poolTop++;
  poolData[i] = data;
  poolGrad[i] = 0;
  poolC0[i]   = c0 === -1 ? -1 : c0;
  poolC1[i]   = c1 === -1 ? -1 : c1;
  poolLg0[i]  = lg0;
  poolLg1[i]  = lg1;
  poolNC[i]   = nc;
  return i;
}

function valAdd(a, b) { return newVal(poolData[a] + poolData[b], a, b, 1, 1, 2); }
function valMul(a, b) { return newVal(poolData[a] * poolData[b], a, b, poolData[b], poolData[a], 2); }
function valMulS(a, s) {
  const tmp = newVal(s, -1, -1, 0, 0, 0);
  return valMul(a, tmp);
}
function valPow(a, exp) {
  const d = poolData[a];
  return newVal(Math.pow(d, exp), a, -1, exp * Math.pow(d, exp - 1), 0, 1);
}
function valLog(a)  { const d = poolData[a]; return newVal(Math.log(d), a, -1, 1/d, 0, 1); }
function valExp(a)  { const e = Math.exp(poolData[a]); return newVal(e, a, -1, e, 0, 1); }
function valRelu(a) { const d = poolData[a]; return newVal(d > 0 ? d : 0, a, -1, d > 0 ? 1 : 0, 0, 1); }
function valNeg(a)  { return valMulS(a, -1); }
function valSub(a, b) { return valAdd(a, valNeg(b)); }
function valDiv(a, b) { return valMul(a, valPow(b, -1)); }

// Backward: iterative topological sort then reverse accumulation
const topoArr   = new Int32Array(POOL_SIZE);
const visitBuf  = new Uint8Array(POOL_SIZE);
const stackBuf  = new Int32Array(POOL_SIZE);

function backward(lossIdx) {
  // clear visited flags only up to poolTop
  visitBuf.fill(0, 0, poolTop);

  let topoN = 0;
  let sp = 0;
  stackBuf[sp++] = lossIdx;

  while (sp > 0) {
    const v = stackBuf[sp - 1];
    if (visitBuf[v]) { sp--; continue; }
    let pushed = false;
    const nc = poolNC[v];
    if (nc >= 1) {
      const c0 = poolC0[v];
      if (c0 !== -1 && !visitBuf[c0]) { stackBuf[sp++] = c0; pushed = true; }
    }
    if (!pushed && nc >= 2) {
      const c1 = poolC1[v];
      if (c1 !== -1 && !visitBuf[c1]) { stackBuf[sp++] = c1; pushed = true; }
    }
    if (!pushed) {
      visitBuf[v] = 1;
      topoArr[topoN++] = v;
      sp--;
    }
  }

  poolGrad[lossIdx] = 1;
  for (let i = topoN - 1; i >= 0; i--) {
    const v = topoArr[i];
    const g = poolGrad[v];
    if (g === 0) continue;
    const nc = poolNC[v];
    if (nc >= 1 && poolC0[v] !== -1) poolGrad[poolC0[v]] += poolLg0[v] * g;
    if (nc >= 2 && poolC1[v] !== -1) poolGrad[poolC1[v]] += poolLg1[v] * g;
  }
}

// ---------------------------------------------------------------------------
// Parameter matrices
// ---------------------------------------------------------------------------
function makeMatrix(nout, nin, std = 0.08) {
  const data = new Int32Array(nout * nin); // stores pool indices
  for (let i = 0; i < nout * nin; i++) {
    data[i] = newVal(rngGauss(std), -1, -1, 0, 0, 0);
  }
  return { data, nout, nin };
}

// ---------------------------------------------------------------------------
// Model helpers
// ---------------------------------------------------------------------------
function linear(x, w) {
  const out = new Int32Array(w.nout);
  for (let i = 0; i < w.nout; i++) {
    let acc = newVal(0, -1, -1, 0, 0, 0);
    for (let j = 0; j < w.nin; j++) {
      acc = valAdd(acc, valMul(w.data[i * w.nin + j], x[j]));
    }
    out[i] = acc;
  }
  return out;
}

function softmax(logits) {
  let maxVal = -Infinity;
  for (let i = 0; i < logits.length; i++) if (poolData[logits[i]] > maxVal) maxVal = poolData[logits[i]];
  const maxNode = newVal(maxVal, -1, -1, 0, 0, 0);
  const exps = new Int32Array(logits.length);
  let total = newVal(0, -1, -1, 0, 0, 0);
  for (let i = 0; i < logits.length; i++) {
    exps[i] = valExp(valSub(logits[i], maxNode));
    total = valAdd(total, exps[i]);
  }
  const out = new Int32Array(logits.length);
  for (let i = 0; i < logits.length; i++) out[i] = valDiv(exps[i], total);
  return out;
}

function rmsnorm(x) {
  let ms = newVal(0, -1, -1, 0, 0, 0);
  for (let i = 0; i < x.length; i++) ms = valAdd(ms, valMul(x[i], x[i]));
  ms = valMulS(ms, 1 / x.length);
  const msEps = valAdd(ms, newVal(1e-5, -1, -1, 0, 0, 0));
  const scale = valPow(msEps, -0.5);
  const out = new Int32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = valMul(x[i], scale);
  return out;
}

// ---------------------------------------------------------------------------
// State dict
// ---------------------------------------------------------------------------
const wte     = makeMatrix(0, N_EMBD); // will be resized after vocab is known
const wpe     = makeMatrix(BLOCK_SIZE, N_EMBD);
const lmHead  = makeMatrix(0, N_EMBD); // will be resized after vocab is known

// Placeholder — rebuilt after vocab_size is known
let WTE, WPE, LM_HEAD;
const ATTN_WQ = [], ATTN_WK = [], ATTN_WV = [], ATTN_WO = [];
const MLP_FC1 = [], MLP_FC2 = [];

// KV cache: [layer][pos][dim] — stores pool indices
const keysCache  = Array.from({length: N_LAYER}, () => []);
const valsCache  = Array.from({length: N_LAYER}, () => []);

function gptForward(tokenId, posId) {
  let x = new Int32Array(N_EMBD);
  for (let i = 0; i < N_EMBD; i++)
    x[i] = valAdd(WTE.data[tokenId * N_EMBD + i], WPE.data[posId * N_EMBD + i]);

  x = rmsnorm(x);

  for (let li = 0; li < N_LAYER; li++) {
    // --- Attention ---
    const xRes = x.slice();
    const xn = rmsnorm(x);
    const q = linear(xn, ATTN_WQ[li]);
    const k = linear(xn, ATTN_WK[li]);
    const v = linear(xn, ATTN_WV[li]);

    keysCache[li].push(k);
    valsCache[li].push(v);
    const seqLen = keysCache[li].length;

    const xAttn = new Int32Array(N_EMBD);
    for (let h = 0; h < N_HEAD; h++) {
      const hs = h * HEAD_DIM;
      const scale = 1 / Math.sqrt(HEAD_DIM);
      const attnLogits = new Int32Array(seqLen);
      for (let t = 0; t < seqLen; t++) {
        let dot = newVal(0, -1, -1, 0, 0, 0);
        for (let j = 0; j < HEAD_DIM; j++)
          dot = valAdd(dot, valMul(q[hs + j], keysCache[li][t][hs + j]));
        attnLogits[t] = valMulS(dot, scale);
      }
      const attnW = softmax(attnLogits);
      for (let j = 0; j < HEAD_DIM; j++) {
        let acc = newVal(0, -1, -1, 0, 0, 0);
        for (let t = 0; t < seqLen; t++)
          acc = valAdd(acc, valMul(attnW[t], valsCache[li][t][hs + j]));
        xAttn[hs + j] = acc;
      }
    }

    const xProj = linear(xAttn, ATTN_WO[li]);
    for (let i = 0; i < N_EMBD; i++) x[i] = valAdd(xProj[i], xRes[i]);

    // --- MLP ---
    const xRes2 = x.slice();
    const xn2 = rmsnorm(x);
    let fc1 = linear(xn2, MLP_FC1[li]);
    fc1 = Int32Array.from(fc1, v => valRelu(v));
    const fc2 = linear(fc1, MLP_FC2[li]);
    for (let i = 0; i < N_EMBD; i++) x[i] = valAdd(fc2[i], xRes2[i]);
  }

  return linear(x, LM_HEAD);
}

// ---------------------------------------------------------------------------
// Main: load data, train, infer
// ---------------------------------------------------------------------------
async function main() {
  // --- Load dataset ---
  const inputFile = 'input.txt';
  if (!fs.existsSync(inputFile)) {
    console.log('Downloading input.txt...');
    await new Promise((resolve, reject) => {
      const file = fs.createWriteStream(inputFile);
      https.get('https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt', res => {
        res.pipe(file);
        file.on('finish', () => { file.close(); resolve(); });
      }).on('error', reject);
    });
  }

  let docs = fs.readFileSync(inputFile, 'utf8')
    .split('\n')
    .map(l => l.trim())
    .filter(l => l.length > 0);

  // Fisher-Yates shuffle
  for (let i = docs.length - 1; i > 0; i--) {
    const j = Math.floor(rngUniform() * (i + 1));
    [docs[i], docs[j]] = [docs[j], docs[i]];
  }
  console.log(`num docs: ${docs.length}`);

  // --- Vocabulary ---
  const ucharsSet = new Set(docs.join(''));
  const uchars = [...ucharsSet].sort();
  const BOS = uchars.length;
  const vocabSize = uchars.length + 1;
  const charToId = new Map(uchars.map((c, i) => [c, i]));
  console.log(`vocab size: ${vocabSize}`);

  // --- Initialize parameters (now that vocabSize is known) ---
  WTE    = makeMatrix(vocabSize, N_EMBD);
  WPE    = makeMatrix(BLOCK_SIZE, N_EMBD);
  LM_HEAD = makeMatrix(vocabSize, N_EMBD);
  for (let li = 0; li < N_LAYER; li++) {
    ATTN_WQ.push(makeMatrix(N_EMBD, N_EMBD));
    ATTN_WK.push(makeMatrix(N_EMBD, N_EMBD));
    ATTN_WV.push(makeMatrix(N_EMBD, N_EMBD));
    ATTN_WO.push(makeMatrix(N_EMBD, N_EMBD));
    MLP_FC1.push(makeMatrix(4 * N_EMBD, N_EMBD));
    MLP_FC2.push(makeMatrix(N_EMBD, 4 * N_EMBD));
  }

  // Collect all param indices
  const paramMatrices = [WTE, WPE, LM_HEAD,
    ...ATTN_WQ, ...ATTN_WK, ...ATTN_WV, ...ATTN_WO,
    ...MLP_FC1, ...MLP_FC2];
  const params = [];
  for (const m of paramMatrices)
    for (let i = 0; i < m.data.length; i++) params.push(m.data[i]);
  console.log(`num params: ${params.length}`);

  const paramsPoolEnd = poolTop; // everything below = permanent param nodes

  // --- Adam buffers ---
  const mBuf = new Float64Array(params.length);
  const vBuf = new Float64Array(params.length);
  const LR = 0.01, BETA1 = 0.85, BETA2 = 0.99, EPS_ADAM = 1e-8;

  // --- Training loop ---
  for (let step = 0; step < NUM_STEPS; step++) {
    // Reset computation graph, keep param nodes
    poolTop = paramsPoolEnd;
    // Reset grads on params
    for (let i = 0; i < params.length; i++) poolGrad[params[i]] = 0;
    // Reset KV cache
    for (let li = 0; li < N_LAYER; li++) { keysCache[li].length = 0; valsCache[li].length = 0; }

    // Tokenize document
    const doc = docs[step % docs.length];
    const tokens = [BOS, ...doc.split('').map(c => charToId.get(c)), BOS];
    const n = Math.min(BLOCK_SIZE, tokens.length - 1);

    // Forward
    let lossSum = newVal(0, -1, -1, 0, 0, 0);
    for (let pos = 0; pos < n; pos++) {
      const tokenId  = tokens[pos];
      const targetId = tokens[pos + 1];
      const logits = gptForward(tokenId, pos);
      const probs  = softmax(logits);
      const lossT  = valMulS(valLog(probs[targetId]), -1);
      lossSum = valAdd(lossSum, lossT);
    }
    const loss = valMulS(lossSum, 1 / n);

    // Backward
    backward(loss);

    // Adam update
    const lrT = LR * (1 - step / NUM_STEPS);
    const b1t = Math.pow(BETA1, step + 1);
    const b2t = Math.pow(BETA2, step + 1);
    for (let i = 0; i < params.length; i++) {
      const g = poolGrad[params[i]];
      mBuf[i] = BETA1 * mBuf[i] + (1 - BETA1) * g;
      vBuf[i] = BETA2 * vBuf[i] + (1 - BETA2) * g * g;
      const mHat = mBuf[i] / (1 - b1t);
      const vHat = vBuf[i] / (1 - b2t);
      poolData[params[i]] -= lrT * mHat / (Math.sqrt(vHat) + EPS_ADAM);
    }

    process.stdout.write(`step ${String(step+1).padStart(4)} / ${NUM_STEPS} | loss ${poolData[loss].toFixed(4)}\r`);
  }
  process.stdout.write('\n');

  // --- Inference ---
  const TEMPERATURE = 0.5;
  console.log('--- inference (new, hallucinated names) ---');
  for (let s = 0; s < 20; s++) {
    poolTop = paramsPoolEnd;
    for (let li = 0; li < N_LAYER; li++) { keysCache[li].length = 0; valsCache[li].length = 0; }

    let tokenId = BOS;
    const sample = [];
    for (let pos = 0; pos < BLOCK_SIZE; pos++) {
      const logits = gptForward(tokenId, pos);
      // scale by temperature
      const scaled = Int32Array.from(logits, l => valMulS(l, 1 / TEMPERATURE));
      const probs  = softmax(scaled);
      tokenId = rngChoices(vocabSize, Array.from(probs, p => poolData[p]));
      if (tokenId === BOS) break;
      sample.push(uchars[tokenId]);
    }
    console.log(`sample ${String(s+1).padStart(2)}: ${sample.join('')}`);
  }
}

main().catch(err => { console.error(err); process.exit(1); });
