import {useCallback, useEffect, useRef, useState} from 'react';
import {
  Canvas,
  createPortal,
  extend,
  useFrame,
  useThree,
} from '@react-three/fiber';
import ThreeMeshUI from 'three-mesh-ui';

import {ARButton, XR, Hands, XREvent} from '@react-three/xr';

import {TextGeometry} from 'three/examples/jsm/geometries/TextGeometry.js';
import {TranslationSentences} from '../types/StreamingTypes';
import Button from './Button';
import {RoomState} from '../types/RoomState';
import ThreeMeshUIText, {ThreeMeshUITextType} from './ThreeMeshUIText';
import {BLACK, WHITE} from './Colors';

/**
 * Using `?url` at the end of this import tells vite this is a static asset, and
 * provides us a URL to the hashed version of the file when the project is built.
 * See: https://vitejs.dev/guide/assets.html#explicit-url-imports
 */
import robotoFontFamilyJson from '../assets/RobotoMono-Regular-msdf.json?url';
import robotoFontTexture from '../assets/RobotoMono-Regular.png';
import {getURLParams} from '../URLParams';
import TextBlocks from './TextBlocks';
import {BufferedSpeechPlayer} from '../createBufferedSpeechPlayer';
import {CURSOR_BLINK_INTERVAL_MS} from '../cursorBlinkInterval';
import supportedCharSet from './supportedCharSet';

// Adds on react JSX for add-on libraries to react-three-fiber
extend(ThreeMeshUI);
extend({TextGeometry});

// This component wraps any children so it is positioned relative to the camera, rather than from the origin
function CameraLinkedObject({children}) {
  const camera = useThree((state) => state.camera);
  return createPortal(<>{children}</>, camera);
}

function ThreeMeshUIComponents({
  translationSentences,
  skipARIntro,
  roomState,
  animateTextDisplay,
}: XRConfigProps & {skipARIntro: boolean}) {
  // The "loop" for re-rendering required for threemeshUI
  useFrame(() => {
    ThreeMeshUI.update();
  });
  const [started, setStarted] = useState<boolean>(skipARIntro);
  return (
    <>
      <CameraLinkedObject>
        {getURLParams().ARTranscriptionType === 'single_block' ? (
          <TranscriptPanelSingleBlock
            started={started}
            animateTextDisplay={animateTextDisplay}
            roomState={roomState}
            translationSentences={translationSentences}
          />
        ) : (
          <TranscriptPanelBlocks translationSentences={translationSentences} />
        )}
        {skipARIntro ? null : (
          <IntroPanel started={started} setStarted={setStarted} />
        )}
      </CameraLinkedObject>
    </>
  );
}

// Original UI that just uses a single block to render 6 lines in a panel
function TranscriptPanelSingleBlock({
  animateTextDisplay,
  started,
  translationSentences,
  roomState,
}: {
  animateTextDisplay: boolean;
  started: boolean;
  translationSentences: TranslationSentences;
  roomState: RoomState | null;
}) {
  const textRef = useRef<ThreeMeshUITextType>();
  const [didReceiveTranslationSentences, setDidReceiveTranslationSentences] =
    useState(false);

  const hasActiveTranscoders = (roomState?.activeTranscoders ?? 0) > 0;

  const [cursorBlinkOn, setCursorBlinkOn] = useState(false);

  // Normally we don't setState in render, but here we need to for computed state, and this if statement assures it won't loop infinitely
  if (!didReceiveTranslationSentences && translationSentences.length > 0) {
    setDidReceiveTranslationSentences(true);
  }

  const width = 1;
  const height = 0.3;
  const fontSize = 0.03;

  useEffect(() => {
    if (animateTextDisplay && hasActiveTranscoders) {
      const interval = setInterval(() => {
        setCursorBlinkOn((prev) => !prev);
      }, CURSOR_BLINK_INTERVAL_MS);

      return () => clearInterval(interval);
    } else {
      setCursorBlinkOn(false);
    }
  }, [animateTextDisplay, hasActiveTranscoders]);

  useEffect(() => {
    if (textRef.current != null) {
      const initialPrompt =
        'Welcome to the presentation. We are excited to share with you the work we have been doing... Our model can now translate languages in less than 2 second latency.';
      // These are rough ratios based on spot checking
      const maxLines = 6;
      const charsPerLine = 55;

      const transcriptSentences: string[] = didReceiveTranslationSentences
        ? translationSentences
        : [initialPrompt];

      // The transcript is an array of sentences. For each sentence we break this down into an array of words per line.
      // This is needed so we can "scroll" through without changing the order of words in the transcript
      const linesToDisplay = transcriptSentences.flatMap((sentence, idx) => {
        const blinkingCursor =
          cursorBlinkOn && idx === transcriptSentences.length - 1 ? '|' : ' ';
        const words = sentence.concat(blinkingCursor).split(/\s+/);
        // Here we break each sentence up with newlines so all words per line fit within the panel
        return words.reduce(
          (wordChunks, currentWord) => {
            const filteredWord = [...currentWord]
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
            const lastLineSoFar = wordChunks[wordChunks.length - 1];
            const charCount = lastLineSoFar.length + filteredWord.length + 1;
            if (charCount <= charsPerLine) {
              wordChunks[wordChunks.length - 1] =
                lastLineSoFar + ' ' + filteredWord;
            } else {
              wordChunks.push(filteredWord);
            }
            return wordChunks;
          },
          [''],
        );
      });

      // Only keep the last maxLines so new text keeps scrolling up from the bottom
      linesToDisplay.splice(0, linesToDisplay.length - maxLines);
      textRef.current.set({content: linesToDisplay.join('\n')});
    }
  }, [
    translationSentences,
    textRef,
    didReceiveTranslationSentences,
    cursorBlinkOn,
  ]);

  const opacity = started ? 1 : 0;
  return (
    <block
      args={[{padding: 0.05, backgroundOpacity: opacity}]}
      position={[0, -0.4, -1.3]}>
      <block
        args={[
          {
            width,
            height,
            fontSize,
            textAlign: 'left',
            backgroundOpacity: opacity,
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
        <ThreeMeshUIText
          ref={textRef}
          content={'Transcript'}
          fontOpacity={opacity}
        />
      </block>
    </block>
  );
}

// Splits up the lines into separate blocks to treat each one separately.
// This allows changing of opacity, animating per line, changing height / width per line etc
function TranscriptPanelBlocks({
  translationSentences,
}: {
  translationSentences: TranslationSentences;
}) {
  return (
    <TextBlocks
      translationText={'Listening...\n' + translationSentences.join('\n')}
    />
  );
}

function IntroPanel({started, setStarted}) {
  const width = 0.5;
  const height = 0.4;
  const padding = 0.03;

  // Kind of hacky but making the panel disappear by moving it completely off the camera view.
  // If we try to remove elements we end up throwing and stopping the experience
  // opacity=0 also runs into weird bugs where not everything is invisible
  const xCoordinate = started ? 1000000 : 0;

  const commonArgs = {
    backgroundColor: WHITE,
    width,
    height,
    padding,
    backgroundOpacity: 1,
    textAlign: 'center',
    fontFamily: robotoFontFamilyJson,
    fontTexture: robotoFontTexture,
  };
  return (
    <>
      <block
        args={[
          {
            ...commonArgs,
            fontSize: 0.02,
          },
        ]}
        position={[xCoordinate, -0.1, -0.5]}>
        <ThreeMeshUIText
          content="FAIR Seamless Streaming Demo"
          fontColor={BLACK}
        />
      </block>
      <block
        args={[
          {
            ...commonArgs,
            fontSize: 0.016,
            backgroundOpacity: 0,
          },
        ]}
        position={[xCoordinate, -0.15, -0.5001]}>
        <ThreeMeshUIText
          fontColor={BLACK}
          content="Welcome to the Seamless team streaming demo experience! In this demo, you would experience AI powered text and audio translation in real time."
        />
      </block>
      <block
        args={[
          {
            width: 0.1,
            height: 0.1,
            backgroundOpacity: 1,
            backgroundColor: BLACK,
          },
        ]}
        position={[xCoordinate, -0.23, -0.5002]}>
        <Button
          onClick={() => setStarted(true)}
          content={'Start Experience'}
          width={0.2}
          height={0.035}
          fontSize={0.015}
          padding={0.01}
          borderRadius={0.01}
        />
      </block>
    </>
  );
}

export type XRConfigProps = {
  animateTextDisplay: boolean;
  bufferedSpeechPlayer: BufferedSpeechPlayer;
  translationSentences: TranslationSentences;
  roomState: RoomState | null;
  roomID: string | null;
  startStreaming: () => Promise<void>;
  stopStreaming: () => Promise<void>;
  debugParam: boolean | null;
  onARVisible?: () => void;
  onARHidden?: () => void;
};

export default function XRConfig(props: XRConfigProps) {
  const {bufferedSpeechPlayer, debugParam} = props;
  const skipARIntro = getURLParams().skipARIntro;
  const defaultDimensions = {width: 500, height: 500};
  const [dimensions, setDimensions] = useState(
    debugParam ? defaultDimensions : {width: 0, height: 0},
  );
  const {width, height} = dimensions;

  // Make sure to reset buffer when headset is taken off / on so we don't get an endless stream
  // of audio. The oculus actually runs for some time after the headset is taken off.
  const resetBuffers = useCallback(
    (event: XREvent<XRSessionEvent>) => {
      const session = event.target;
      if (!(session instanceof XRSession)) {
        return;
      }
      switch (session.visibilityState) {
        case 'visible':
          bufferedSpeechPlayer.start();
          break;
        case 'hidden':
          bufferedSpeechPlayer.stop();
          break;
      }
    },
    [bufferedSpeechPlayer],
  );

  return (
    <div style={{height, width, margin: '0 auto', border: '1px solid #ccc'}}>
      {/* This is the button that triggers AR flow if available via a button */}
      <ARButton
        onError={(e) => console.error(e)}
        onClick={() => setDimensions(defaultDimensions)}
        style={{
          position: 'absolute',
          bottom: '24px',
          left: '50%',
          transform: 'translateX(-50%)',
          padding: '12px 24px',
          border: '1px solid white',
          borderRadius: '4px',
          backgroundColor: '#465a69',
          color: 'white',
          font: 'normal 0.8125rem sans-serif',
          outline: 'none',
          zIndex: 99999,
          cursor: 'pointer',
        }}
      />
      {/* Canvas to draw if in browser but if in AR mode displays in pass through mode */}
      {/* The camera here just works in 2D mode. In AR mode it starts at at origin */}
      {/* <Canvas camera={{position: [0, 0, 1], fov: 60}}> */}
      <Canvas camera={{position: [0, 0, 0.001], fov: 60}}>
        <color attach="background" args={['grey']} />
        <XR referenceSpace="local" onVisibilityChange={resetBuffers}>
          {/*
            Uncomment this for controllers to show up
            <Controllers />
          */}
          <Hands />

          {/*
            Uncomment this for moving with controllers
            <MovementController />
          */}
          {/*
            Uncomment this for turning the view in non-vr mode
            <OrbitControls
              autoRotateSpeed={0.85}
              zoomSpeed={1}
              minPolarAngle={Math.PI / 2.5}
              maxPolarAngle={Math.PI / 2.55}
            />
          */}
          <ThreeMeshUIComponents {...props} skipARIntro={skipARIntro} />
          {/* Just for testing */}
          {/* <RandomComponents /> */}
        </XR>
      </Canvas>
    </div>
  );
}
