import {ServerTextData, TranslationSentences} from './types/StreamingTypes';

export default function getTranslationSentencesFromReceivedData(
  receivedData: Array<ServerTextData>,
): TranslationSentences {
  return receivedData
    .reduce(
      (acc, data) => {
        const newAcc = [
          ...acc.slice(0, -1),
          acc[acc.length - 1].trim() + ' ' + data.payload,
        ];
        if (data.eos) {
          newAcc.push('');
        }

        return newAcc;
      },
      [''],
    )
    .filter((s) => s.trim().length !== 0);
}
