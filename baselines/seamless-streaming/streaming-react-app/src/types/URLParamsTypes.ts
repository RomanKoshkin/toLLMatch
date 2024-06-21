export type URLParamsObject = {
  animateTextDisplay: boolean;
  autoJoin: boolean;
  debug: boolean;
  enableServerLock: boolean;
  roomID: string | null;
  serverURL: string | null;
  skipARIntro: boolean;
  ARTranscriptionType:
  | 'single_block'
  | 'lines'
  | 'lines_with_background'
  | string;
};

export type URLParamNames = keyof URLParamsObject;
