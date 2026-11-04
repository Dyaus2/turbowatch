/** Python-compatible number formatting. JS toFixed rounds exact ties away from zero (1.25 -> "1.3"), Python rounds
 *  the exact binary value half-to-even (1.25 -> "1.2"). Using exact BigInt arithmetic keeps browser output identical
 *  to the Python API, e.g. f"{x:.1f}", f"{x:+.1f}" and round(x, n). */
const view = new DataView(new ArrayBuffer(8));

export function pyFixed(x: number, digits: number, plus = false): string {
  if (!Number.isFinite(x)) return String(x);
  view.setFloat64(0, x);
  const hi = view.getUint32(0);
  const lo = view.getUint32(4);
  const neg = hi >>> 31 === 1;
  const expBits = (hi >>> 20) & 0x7ff;
  let mant = (BigInt(hi & 0xfffff) << 32n) | BigInt(lo);
  let exp: number;
  if (expBits === 0) exp = -1074;
  else {
    mant |= 1n << 52n;
    exp = expBits - 1075;
  }
  // |x| = mant * 2^exp exactly; n = round_half_even(|x| * 10^digits)
  const scale = 10n ** BigInt(digits);
  let n: bigint;
  if (exp >= 0) n = (mant << BigInt(exp)) * scale;
  else {
    const num = mant * scale;
    const den = 1n << BigInt(-exp);
    const q = num / den;
    const twice = 2n * (num % den);
    n = twice > den ? q + 1n : twice < den ? q : q % 2n === 0n ? q : q + 1n;
  }
  let s = n.toString();
  if (digits > 0) {
    s = s.padStart(digits + 1, "0");
    s = `${s.slice(0, -digits)}.${s.slice(-digits)}`;
  }
  return (neg ? "-" : plus ? "+" : "") + s;
}

/** Python round(x, digits) for floats. */
export const pyRound = (x: number, digits: number): number => Number(pyFixed(x, digits));
