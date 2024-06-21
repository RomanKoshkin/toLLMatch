import robotoFontFamilyJson from '../assets/RobotoMono-Regular-msdf.json?url';

async function fetchSupportedCharSet(): Promise<Set<string>> {
  try {
    const response = await fetch(robotoFontFamilyJson);
    const fontFamily = await response.json();

    return new Set(fontFamily.info.charset);
  } catch (e) {
    console.error('Failed to fetch supported XR charset', e);
    return new Set();
  }
}

let charSet = new Set<string>();
fetchSupportedCharSet().then((result) => (charSet = result));

export default function supportedCharSet(): Set<string> {
  return charSet;
}
