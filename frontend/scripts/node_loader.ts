/** Shared helpers for the Node verification scripts (static data from disk). */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

export const FRONTEND = join(dirname(fileURLToPath(import.meta.url)), "..");
export const DATA = join(FRONTEND, "static-data");
export const REPO = join(FRONTEND, "..");

export const loadJSON = async (path: string): Promise<unknown> => JSON.parse(readFileSync(join(DATA, path), "utf-8"));

export function loadEmbeddings(): Float32Array {
  const b = readFileSync(join(DATA, "embeddings.bin"));
  return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
}
