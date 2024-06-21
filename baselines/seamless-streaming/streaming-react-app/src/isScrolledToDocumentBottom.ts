export default function isScrolledToDocumentBottom(
  bufferPx: number = 0,
): boolean {
  if (
    window.innerHeight + window.scrollY >=
    document.body.offsetHeight - bufferPx
  ) {
    return true;
  }
  return false;
}
