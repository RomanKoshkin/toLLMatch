// Useful for ensuring switch statements are exhaustive
export default function exhaustivenessCheck(p: never): never {
  throw new Error(
    `This should never happen. Value received: ${JSON.stringify(p)}`,
  );
}
