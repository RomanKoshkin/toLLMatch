export default function float32To16BitPCM(
  float32Arr: Float32Array,
): Int16Array {
  const pcm16bit = new Int16Array(float32Arr.length);
  for (let i = 0; i < float32Arr.length; ++i) {
    // force number in [-1,1]
    const s = Math.max(-1, Math.min(1, float32Arr[i]));

    /**
     * convert 32 bit float to 16 bit int pcm audio
     * 0x8000 = minimum int16 value, 0x7fff = maximum int16 value
     */
    pcm16bit[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return pcm16bit;
}
