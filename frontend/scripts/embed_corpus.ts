/** Embed the copilot corpus with the browser embedding model -> static-data/embeddings.bin (float32, row-major).
 *  Run: node scripts/embed_corpus.ts   (after python scripts/export_static.py) */
import { writeFileSync } from "node:fs";
import { join } from "node:path";
import type { Chunk } from "../src/static/copilot.ts";
import { loadModels } from "../src/static/models.ts";
import { DATA, loadJSON } from "./node_loader.ts";

const chunks = (await loadJSON("chunks.json")) as Chunk[];
const models = await loadModels();
const parts: Float32Array[] = [];
for (let i = 0; i < chunks.length; i += 16) parts.push(await models.embed(chunks.slice(i, i + 16).map((c) => c.text))); // passages: no query prefix
const dim = parts[0].length / Math.min(16, chunks.length);
const all = new Float32Array(chunks.length * dim);
let off = 0;
for (const p of parts) { all.set(p, off); off += p.length; }
if (off !== all.length) throw new Error("size mismatch");
writeFileSync(join(DATA, "embeddings.bin"), Buffer.from(all.buffer));
console.log(`embedded ${chunks.length} chunks, dim ${dim}`);
