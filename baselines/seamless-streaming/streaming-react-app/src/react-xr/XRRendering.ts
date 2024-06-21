import * as THREE from 'three';
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls.js';

import ThreeMeshUI, {Block, Text} from 'three-mesh-ui';

import FontJSON from '../assets/RobotoMono-Regular-msdf.json?url';
import FontImage from '../assets/RobotoMono-Regular.png';
import {TranslationSentences} from '../types/StreamingTypes';
import supportedCharSet from './supportedCharSet';

// Augment three-mesh-ui types which aren't implemented
declare module 'three-mesh-ui' {
  interface Block {
    add(any: any);
    set(props: BlockOptions);
    position: {
      x: number;
      y: number;
      z: number;
      set: (x: number, y: number, z: number) => void;
    };
  }
  interface Text {
    set(props: {content: string});
  }
}

// Various configuration parameters
const INITIAL_PROMPT = 'Listening...\n';
const NUM_LINES = 3;
const CHARS_PER_LINE = 37;
const CHARS_PER_SECOND = 15;

const MAX_WIDTH = 0.89;
const CHAR_WIDTH = 0.0233;
const Y_COORD_START = -0.38;
const Z_COORD = -1.3;
const LINE_HEIGHT = 0.062;
const BLOCK_SPACING = 0.02;
const FONT_SIZE = 0.038;

// Speed of scrolling of text lines
const SCROLL_Y_DELTA = 0.01;

// Overlay an extra block for padding due to inflexibilities of native padding
const OFFSET = 0.01;
const OFFSET_WIDTH = OFFSET * 3;

// The tick interval
const CURSOR_BLINK_INTERVAL_MS = 500;

type TranscriptState = {
  translationText: string;
  textBlocksProps: TextBlockProps[];
  lastTranslationStringIndex: number;
  lastTranslationLineStartIndex: number;
  transcriptLines: string[];
  lastUpdateTime: number;
};

type TextBlockProps = {
  content: string;
  // The end position when animating
  targetY: number;
  // Current scroll position that caps at targetY
  currentY: number;
  textOpacity: number;
  backgroundOpacity: number;
  index: number;
  isBottomLine: boolean;
};

function initialTextBlockProps(count: number): TextBlockProps[] {
  return Array.from({length: count}).map(() => {
    // Push in non display blocks because mesh UI crashes if elements are add / removed from screen.

    return {
      // key: textBlocksProps.length,
      targetY: Y_COORD_START,
      currentY: Y_COORD_START,
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

function initialState(): TranscriptState {
  return {
    translationText: '',
    textBlocksProps: initialTextBlockProps(NUM_LINES),
    lastTranslationStringIndex: 0,
    lastTranslationLineStartIndex: 0,
    transcriptLines: [],
    lastUpdateTime: new Date().getTime(),
  };
}

let transcriptState: TranscriptState = initialState();

let scene: THREE.Scene | null;
let camera: THREE.PerspectiveCamera | null;
let renderer: THREE.WebGLRenderer | null;
let controls: THREE.OrbitControls | null;

let cursorBlinkOn: boolean = false;

setInterval(() => {
  cursorBlinkOn = !cursorBlinkOn;
}, CURSOR_BLINK_INTERVAL_MS);

type TextBlock = {
  textBlockOuterContainer: Block;
  textBlockInnerContainer: Block;
  text: Text;
};
const textBlocks: TextBlock[] = [];

export function getRenderer(): THREE.WebGLRenderer | null {
  return renderer;
}

export function init(
  width: number,
  height: number,
  parentElement: HTMLDivElement | null,
): THREE.WebGLRenderer {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x505050);

  camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
  camera.position.z = 1;

  renderer = new THREE.WebGLRenderer({
    antialias: true,
  });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(width, height);
  renderer.xr.enabled = true;

  renderer.xr.setReferenceSpaceType('local');

  parentElement?.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.update();

  scene.add(camera);

  textBlocks.push(
    ...initialTextBlockProps(NUM_LINES).map((props) => makeTextBlock(props)),
  );

  renderer.setAnimationLoop(loop);
  return renderer;
}

export function updatetranslationText(
  translationSentences: TranslationSentences,
): void {
  const newText = INITIAL_PROMPT + translationSentences.join('\n');
  if (transcriptState.translationText === newText) {
    return;
  }
  transcriptState.translationText = newText;
}

export function resetState(): void {
  transcriptState = initialState();
}

function makeTextBlock({
  content,
  backgroundOpacity,
}: TextBlockProps): TextBlock {
  const width = MAX_WIDTH;
  const height = LINE_HEIGHT;

  const fontProps = {
    fontSize: FONT_SIZE,
    textAlign: 'left',
    // TODO: support more language charsets
    // This renders using MSDF format supported in WebGL. Renderable characters are defined in the "charset" json
    // Currently supports most default keyboard inputs but this would exclude many non latin charset based languages.
    // You can use https://msdf-bmfont.donmccurdy.com/ for easily generating these files
    fontFamily: FontJSON,
    fontTexture: FontImage,
  };

  const textBlockOuterContainer = new Block({
    backgroundOpacity,
    width: width + OFFSET_WIDTH,
    height: height,
    borderRadius: 0,
    ...fontProps,
  });

  const text = new Text({content});
  const textBlockInnerContainer = new Block({
    padding: 0,
    backgroundOpacity: 0,
    width,
    height,
  });

  // Adding it to the camera makes the UI follow it.
  camera.add(textBlockOuterContainer);
  textBlockOuterContainer.add(textBlockInnerContainer);
  textBlockInnerContainer.add(text);

  return {
    textBlockOuterContainer,
    textBlockInnerContainer,
    text,
  };
}

// Updates the position and text of a text block from its props
function updateTextBlock(
  id: number,
  {content, targetY, currentY, backgroundOpacity, isBottomLine}: TextBlockProps,
): void {
  const {textBlockOuterContainer, textBlockInnerContainer, text} =
    textBlocks[id];

  const {lastTranslationStringIndex, translationText} = transcriptState;

  // Add blinking cursor if we don't have any new input to render
  const numChars = content.length;

  if (
    isBottomLine &&
    cursorBlinkOn &&
    lastTranslationStringIndex >= translationText.length
  ) {
    content = content + '|';
  }

  // Accounting for potential cursor for block width (the +1)
  const width =
    (numChars + (isBottomLine ? 1.1 : 0) + (numChars < 10 ? 1 : 0)) *
    CHAR_WIDTH;
  const height = LINE_HEIGHT;

  // Width starts from 0 and goes 1/2 in each direction so offset x
  const xPosition = width / 2 - MAX_WIDTH / 2 + OFFSET_WIDTH;
  textBlockOuterContainer?.set({
    backgroundOpacity,
    width: width + 2 * OFFSET_WIDTH,
    height: height + OFFSET / 3,
    borderRadius: 0,
  });

  // Scroll up line toward target
  const y = isBottomLine
    ? targetY
    : Math.min(currentY + SCROLL_Y_DELTA, targetY);
  transcriptState.textBlocksProps[id].currentY = y;

  textBlockOuterContainer.position.set(-OFFSET_WIDTH + xPosition, y, Z_COORD);
  textBlockInnerContainer.set({
    padding: 0,
    backgroundOpacity: 0,
    width,
    height,
  });
  text.set({content});
}

// We split the text so it fits line by line into the UI
function chunkTranslationTextIntoLines(
  translationText: string,
  nextTranslationStringIndex: number,
): string[] {
  // Ideally we continue where we left off but this is complicated when we have mid-words. Recalculating for now
  const newSentences = translationText
    .substring(0, nextTranslationStringIndex)
    .split('\n');
  const transcriptLines = [''];
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
        .join('')
        // Filter out unknown symbol
        .replace('<unk>', '');

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
  return transcriptLines;
}

// The main loop,
function updateTextBlocksProps(): void {
  const {translationText, lastTranslationStringIndex, lastUpdateTime} =
    transcriptState;

  const currentTime = new Date().getTime();
  const charsToRender = Math.round(
    ((currentTime - lastUpdateTime) * CHARS_PER_SECOND) / 1000,
  );

  if (charsToRender < 1) {
    // Wait some more until we render more characters
    return;
  }

  const nextTranslationStringIndex = Math.min(
    lastTranslationStringIndex + charsToRender,
    translationText.length,
  );
  if (nextTranslationStringIndex === lastTranslationStringIndex) {
    // No new characters to render
    transcriptState.lastUpdateTime = currentTime;
    return;
  }

  // Ideally we continue where we left off but this is complicated when we have mid-words. Recalculating for now
  const transcriptLines = chunkTranslationTextIntoLines(
    translationText,
    nextTranslationStringIndex,
  );
  transcriptState.transcriptLines = transcriptLines;
  transcriptState.lastTranslationStringIndex = nextTranslationStringIndex;

  // Compute the new props for each text block
  const newTextBlocksProps: TextBlockProps[] = [];
  // We start with the most recent line and increment the y coordinate for older lines.
  // If it is a new sentence we increment the y coordinate a little more to leave a visible space
  let y = Y_COORD_START;
  transcriptLines.forEach((line, i) => {
    if (newTextBlocksProps.length == NUM_LINES) {
      return;
    }

    if (line === '\n') {
      y += BLOCK_SPACING;
      return;
    }

    const isBottomLine = newTextBlocksProps.length === 0;

    const textOpacity = 1 - 0.1 * newTextBlocksProps.length;

    const previousProps = transcriptState.textBlocksProps.find(
      (props) => props.index === i,
    );
    const props = {
      targetY: y + LINE_HEIGHT / 2,
      currentY: isBottomLine ? y : previousProps?.currentY || y,
      index: i,
      textOpacity,
      backgroundOpacity: 1,
      content: line,
      isBottomLine,
    };
    newTextBlocksProps.push(props);

    y += LINE_HEIGHT;
  });

  transcriptState.textBlocksProps = newTextBlocksProps;
  transcriptState.lastUpdateTime = currentTime;
}

// The main render loop, everything gets rendered here.
function loop() {
  updateTextBlocksProps();

  transcriptState.textBlocksProps.map((props, i) => updateTextBlock(i, props));

  ThreeMeshUI.update();

  controls.update();
  renderer.render(scene, camera);
}
