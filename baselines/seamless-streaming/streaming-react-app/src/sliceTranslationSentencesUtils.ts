import {TranslationSentences} from './types/StreamingTypes';

export function getTotalSentencesLength(
  translatedSentences: TranslationSentences,
) {
  return translatedSentences.reduce((acc, curr) => acc + curr.length, 0);
}

/**
 * @returns A new array of strings where the total length of the strings === targetIndex,
 * aka it's as if we joined all the strings together, called joined.slice(0, targetIndex), and then
 * split the string back into an array of strings.
 */
export function sliceTranslationSentencesUpToIndex(
  translatedSentences: TranslationSentences,
  targetIndex: number,
): TranslationSentences {
  return translatedSentences.reduce<TranslationSentences>((acc, sentence) => {
    const accTotalLength = getTotalSentencesLength(acc);
    if (accTotalLength === targetIndex) {
      return acc;
    }
    // If adding the current sentence does not exceed the targetIndex, then add the whole sentence
    if (accTotalLength + sentence.length <= targetIndex) {
      return [...acc, sentence];
    }
    // If adding the current sentence DOES exceed the targetIndex, then slice the sentence and add it
    return [...acc, sentence.slice(0, targetIndex - accTotalLength)];
  }, []);
}
