/** transformers.js (ONNX Runtime) versions of the copilot models — same architectures/weights as the Python
 *  server (BAAI/bge-small-en-v1.5, ms-marco-MiniLM-L6-v2), int8-quantised ONNX exports so the browser downloads
 *  ~58 MB once instead of ~220 MB. Quality of this exact setup is measured by scripts/eval_copilot.ts. */
import type { Models } from "./copilot.ts";

export const EMBED_ID = "Xenova/bge-small-en-v1.5";
export const RERANK_ID = "Xenova/ms-marco-MiniLM-L-6-v2";
export const DTYPE = "q8";

export interface Progress { file: string; loaded: number; total: number }

export async function loadModels(onProgress?: (p: Progress) => void): Promise<Models> {
  const { pipeline, AutoTokenizer, AutoModelForSequenceClassification } = await import("@huggingface/transformers");
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const progress_callback = (p: any) => {
    if (onProgress && p?.status === "progress" && p.total) onProgress({ file: `${p.name}/${p.file}`, loaded: p.loaded, total: p.total });
  };
  const [embedder, tokenizer, reranker] = await Promise.all([
    pipeline("feature-extraction", EMBED_ID, { dtype: DTYPE, progress_callback }),
    AutoTokenizer.from_pretrained(RERANK_ID, { progress_callback }),
    AutoModelForSequenceClassification.from_pretrained(RERANK_ID, { dtype: DTYPE, progress_callback }),
  ]);
  return {
    async embed(texts: string[]) {
      const out = await embedder(texts, { pooling: "cls", normalize: true }); // bge uses CLS pooling (as in sentence-transformers)
      return out.data as Float32Array;
    },
    async rerank(query: string, docs: string[]) {
      if (!docs.length) return [];
      const inputs = tokenizer(new Array(docs.length).fill(query), { text_pair: docs, padding: true, truncation: true });
      const { logits } = await reranker(inputs);
      return Array.from(logits.data as Float32Array);
    },
  };
}
