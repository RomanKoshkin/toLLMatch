import {useEffect, useRef, useState} from 'react';
import robotoFontFamilyJson from '../assets/RobotoMono-Regular-msdf.json?url';
import robotoFontTexture from '../assets/RobotoMono-Regular.png';
import ThreeMeshUIText, {ThreeMeshUITextType} from './ThreeMeshUIText';
import supportedCharSet from './supportedCharSet';

const NUM_LINES = 3;

export const CHARS_PER_LINE = 37;
const MAX_WIDTH = 0.89;
const CHAR_WIDTH = 0.0235;
const Y_COORD_START = -0.38;
const Z_COORD = -1.3;
const LINE_HEIGHT = 0.062;
const BLOCK_SPACING = 0.02;
const FONT_SIZE = 0.038;

const SCROLL_Y_DELTA = 0.001;

// Overlay an extra block for padding due to inflexibilities of native padding
const OFFSET = 0.01;
const OFFSET_WIDTH = OFFSET * 3;

const CHARS_PER_SECOND = 10;

// The tick interval
const RENDER_INTERVAL = 300;

const CURSOR_BLINK_INTERVAL_MS = 1000;

type TextBlockProps = {
  content: string;
  // The actual position or end position when animating
  y: number;
  // The start position when animating
  startY: number;
  textOpacity: number;
  backgroundOpacity: number;
  index: number;
  isBottomLine: boolean;
  // key: number;
};

type TranscriptState = {
  textBlocksProps: TextBlockProps[];
  lastTranslationStringIndex: number;
  lastTranslationLineStartIndex: number;
  transcriptLines: string[];
  lastRenderTime: number;
};

function TextBlock({
  content,
  y,
  startY,
  textOpacity,
  backgroundOpacity,
  index,
  isBottomLine,
}: TextBlockProps) {
  const [scrollY, setScrollY] = useState<number>(y);
  // We are reusing text blocks so this keeps track of when we changed rows so we can restart animation
  const lastIndex = useRef<number>(index);
  useEffect(() => {
    if (index != lastIndex.current) {
      lastIndex.current = index;
      !isBottomLine && setScrollY(startY);
    } else if (scrollY < y) {
      setScrollY((prev) => prev + SCROLL_Y_DELTA);
    }
  }, [isBottomLine, index, scrollY, setScrollY, startY, y]);

  const [cursorBlinkOn, setCursorBlinkOn] = useState(false);
  useEffect(() => {
    if (isBottomLine) {
      const interval = setInterval(() => {
        setCursorBlinkOn((prev) => !prev);
      }, CURSOR_BLINK_INTERVAL_MS);

      return () => clearInterval(interval);
    } else {
      setCursorBlinkOn(false);
    }
  }, [isBottomLine]);

  const numChars = content.length;

  if (cursorBlinkOn) {
    content = content + '|';
  }

  // Accounting for potential cursor for block width (the +1)
  const width =
    (numChars + (isBottomLine ? 1.1 : 0) + (numChars < 10 ? 1 : 0)) *
    CHAR_WIDTH;

  const height = LINE_HEIGHT;

  // This is needed to update text content (doesn't work if we just update the content prop)
  const textRef = useRef<ThreeMeshUITextType>();
  useEffect(() => {
    if (textRef.current != null) {
      textRef.current.set({content});
    }
  }, [content, textRef, y, startY]);

  // Width starts from 0 and goes 1/2 in each direction
  const xPosition = width / 2 - MAX_WIDTH / 2 + OFFSET_WIDTH;
  return (
    <>
      <block
        args={[
          {
            backgroundOpacity,
            width: width + OFFSET_WIDTH,
            height: height,
            borderRadius: 0,
          },
        ]}
        position={[-OFFSET_WIDTH + xPosition, scrollY, Z_COORD]}></block>
      <block
        args={[{padding: 0, backgroundOpacity: 0, width, height}]}
        position={[xPosition, scrollY + OFFSET, Z_COORD]}>
        <block
          args={[
            {
              width,
              height,
              fontSize: FONT_SIZE,
              textAlign: 'left',
              backgroundOpacity: 0,
              // TODO: support more language charsets
              // This renders using MSDF format supported in WebGL. Renderable characters are defined in the "charset" json
              // Currently supports most default keyboard inputs but this would exclude many non latin charset based languages.
              // You can use https://msdf-bmfont.donmccurdy.com/ for easily generating these files
              // fontFamily: '/src/assets/Roboto-msdf.json',
              // fontTexture: '/src/assets/Roboto-msdf.png'
              fontFamily: robotoFontFamilyJson,
              fontTexture: robotoFontTexture,
            },
          ]}>
          <ThreeMeshUIText ref={textRef} content="" fontOpacity={textOpacity} />
        </block>
      </block>
    </>
  );
}

function initialTextBlockProps(count: number): TextBlockProps[] {
  return Array.from({length: count}).map(() => {
    // Push in non display blocks because mesh UI crashes if elements are add / removed from screen.
    return {
      y: Y_COORD_START,
      startY: 0,
      index: 0,
      textOpacity: 0,
      backgroundOpacity: 0,
      width: MAX_WIDTH,
      height: LINE_HEIGHT,
      content: '',
      isBottomLine: true,
    };
  });
}

export default function TextBlocks({
  translationText,
}: {
  translationText: string;
}) {
  const transcriptStateRef = useRef<TranscriptState>({
    textBlocksProps: initialTextBlockProps(NUM_LINES),
    lastTranslationStringIndex: 0,
    lastTranslationLineStartIndex: 0,
    transcriptLines: [],
    lastRenderTime: new Date().getTime(),
  });

  const transcriptState = transcriptStateRef.current;
  const {textBlocksProps, lastTranslationStringIndex, lastRenderTime} =
    transcriptState;

  const [charsToRender, setCharsToRender] = useState<number>(0);

  useEffect(() => {
    const interval = setInterval(() => {
      const currentTime = new Date().getTime();
      const charsToRender = Math.round(
        ((currentTime - lastRenderTime) * CHARS_PER_SECOND) / 1000,
      );
      setCharsToRender(charsToRender);
    }, RENDER_INTERVAL);

    return () => clearInterval(interval);
  }, [lastRenderTime]);

  const currentTime = new Date().getTime();
  if (charsToRender < 1) {
    return textBlocksProps.map((props, idx) => (
      <TextBlock {...props} key={idx} />
    ));
  }

  const nextTranslationStringIndex = Math.min(
    lastTranslationStringIndex + charsToRender,
    translationText.length,
  );
  const newString = translationText.substring(
    lastTranslationStringIndex,
    nextTranslationStringIndex,
  );
  if (nextTranslationStringIndex === lastTranslationStringIndex) {
    transcriptState.lastRenderTime = currentTime;
    return textBlocksProps.map((props, idx) => (
      <TextBlock {...props} key={idx} />
    ));
  }

  // Wait until more characters are accumulated if its just blankspace
  if (/^\s*$/.test(newString)) {
    transcriptState.lastRenderTime = currentTime;
    return textBlocksProps.map((props, idx) => (
      <TextBlock {...props} key={idx} />
    ));
  }

  // Ideally we continue where we left off but this is complicated when we have mid-words. Recalculating for now
  const runAll = true;
  const newSentences = runAll
    ? translationText.substring(0, nextTranslationStringIndex).split('\n')
    : newString.split('\n');
  const transcriptLines = runAll ? [''] : transcriptState.transcriptLines;
  newSentences.forEach((newSentence, sentenceIdx) => {
    const words = newSentence.split(/\s+/);
    words.forEach((word) => {
      const filteredWord = [...word]
        .filter((c) => {
          if (supportedCharSet().has(c)) {
            return true;
          }
          console.error(
            `Unsupported char ${c} - make sure this is supported in the font family msdf file`,
          );
          return false;
        })
        .join('');

      const lastLineSoFar = transcriptLines[0];
      const charCount = lastLineSoFar.length + filteredWord.length + 1;

      if (charCount <= CHARS_PER_LINE) {
        transcriptLines[0] = lastLineSoFar + ' ' + filteredWord;
      } else {
        transcriptLines.unshift(filteredWord);
      }
    });

    if (sentenceIdx < newSentences.length - 1) {
      transcriptLines.unshift('\n');
      transcriptLines.unshift('');
    }
  });

  transcriptState.transcriptLines = transcriptLines;
  transcriptState.lastTranslationStringIndex = nextTranslationStringIndex;

  const newTextBlocksProps: TextBlockProps[] = [];
  let currentY = Y_COORD_START;

  transcriptLines.forEach((line, i) => {
    if (newTextBlocksProps.length == NUM_LINES) {
      return;
    }

    // const line = transcriptLines[i];
    if (line === '\n') {
      currentY += BLOCK_SPACING;
      return;
    }
    const y = currentY + LINE_HEIGHT / 2;
    const isBottomLine = newTextBlocksProps.length === 0;

    const textOpacity = 1 - 0.1 * newTextBlocksProps.length;
    newTextBlocksProps.push({
      y,
      startY: currentY,
      index: i,
      textOpacity,
      backgroundOpacity: 0.98,
      content: line,
      isBottomLine,
    });

    currentY = y + LINE_HEIGHT / 2;
  });

  const numRemainingBlocks = NUM_LINES - newTextBlocksProps.length;
  if (numRemainingBlocks > 0) {
    newTextBlocksProps.push(...initialTextBlockProps(numRemainingBlocks));
  }

  transcriptState.textBlocksProps = newTextBlocksProps;
  transcriptState.lastRenderTime = currentTime;
  return newTextBlocksProps.map((props, idx) => (
    <TextBlock {...props} key={idx} />
  ));
}
