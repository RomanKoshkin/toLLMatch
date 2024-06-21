import {useCallback, useEffect, useLayoutEffect, useRef, useState} from 'react';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';
import InputLabel from '@mui/material/InputLabel';
import FormControl from '@mui/material/FormControl';
import Select, {SelectChangeEvent} from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import Stack from '@mui/material/Stack';
import seamlessLogoUrl from './assets/seamless.svg';
import {
  AgentCapabilities,
  BaseResponse,
  BrowserAudioStreamConfig,
  DynamicConfig,
  PartialDynamicConfig,
  SUPPORTED_INPUT_SOURCES,
  SUPPORTED_OUTPUT_MODES,
  ServerExceptionData,
  ServerSpeechData,
  ServerState,
  ServerTextData,
  StartStreamEventConfig,
  StreamingStatus,
  SupportedInputSource,
  SupportedOutputMode,
  TranslationSentences,
} from './types/StreamingTypes';
import FormLabel from '@mui/material/FormLabel';
import RadioGroup from '@mui/material/RadioGroup';
import FormControlLabel from '@mui/material/FormControlLabel';
import Radio from '@mui/material/Radio';
import './StreamingInterface.css';
import RoomConfig from './RoomConfig';
import Divider from '@mui/material/Divider';
import {useSocket} from './useSocket';
import {RoomState} from './types/RoomState';
import useStable from './useStable';
import float32To16BitPCM from './float32To16BitPCM';
import createBufferedSpeechPlayer from './createBufferedSpeechPlayer';
import Checkbox from '@mui/material/Checkbox';
import Alert from '@mui/material/Alert';
import isScrolledToDocumentBottom from './isScrolledToDocumentBottom';
import Box from '@mui/material/Box';
import Slider from '@mui/material/Slider';
import VolumeDown from '@mui/icons-material/VolumeDown';
import VolumeUp from '@mui/icons-material/VolumeUp';
import Mic from '@mui/icons-material/Mic';
import MicOff from '@mui/icons-material/MicOff';
import XRDialog from './react-xr/XRDialog';
import getTranslationSentencesFromReceivedData from './getTranslationSentencesFromReceivedData';
import {
  sliceTranslationSentencesUpToIndex,
  getTotalSentencesLength,
} from './sliceTranslationSentencesUtils';
import Blink from './Blink';
import {CURSOR_BLINK_INTERVAL_MS} from './cursorBlinkInterval';
import {getURLParams} from './URLParams';
import debug from './debug';
import DebugSection from './DebugSection';
import Switch from '@mui/material/Switch';
import Grid from '@mui/material/Grid';
import {getLanguageFromThreeLetterCode} from './languageLookup';
import HeadphonesIcon from '@mui/icons-material/Headphones';

const AUDIO_STREAM_DEFAULTS = {
  userMedia: {
    echoCancellation: false,
    noiseSuppression: true,
  },
  displayMedia: {
    echoCancellation: false,
    noiseSuppression: false,
  },
} as const;

async function requestUserMediaAudioStream(
  config: BrowserAudioStreamConfig = AUDIO_STREAM_DEFAULTS['userMedia'],
) {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {...config, channelCount: 1},
  });
  console.debug(
    '[requestUserMediaAudioStream] stream created with settings:',
    stream.getAudioTracks()?.[0]?.getSettings(),
  );
  return stream;
}

async function requestDisplayMediaAudioStream(
  config: BrowserAudioStreamConfig = AUDIO_STREAM_DEFAULTS['displayMedia'],
) {
  const stream = await navigator.mediaDevices.getDisplayMedia({
    audio: {...config, channelCount: 1},
  });
  console.debug(
    '[requestDisplayMediaAudioStream] stream created with settings:',
    stream.getAudioTracks()?.[0]?.getSettings(),
  );
  return stream;
}

const buttonLabelMap: {[key in StreamingStatus]: string} = {
  stopped: 'Start Streaming',
  running: 'Stop Streaming',
  starting: 'Starting...',
};

const BUFFER_LIMIT = 1;

const SCROLLED_TO_BOTTOM_THRESHOLD_PX = 36;

const GAIN_MULTIPLIER_OVER_1 = 3;

const getGainScaledValue = (value) =>
  value > 1 ? (value - 1) * GAIN_MULTIPLIER_OVER_1 + 1 : value;

const TOTAL_ACTIVE_TRANSCODER_WARNING_THRESHOLD = 2;

const MAX_SERVER_EXCEPTIONS_TRACKED = 500;

export const TYPING_ANIMATION_DELAY_MS = 6;

export default function StreamingInterface() {
  const urlParams = getURLParams();
  const debugParam = urlParams.debug;
  const [animateTextDisplay, setAnimateTextDisplay] = useState<boolean>(
    urlParams.animateTextDisplay,
  );

  const socketObject = useSocket();
  const {socket, clientID} = socketObject;

  const [serverState, setServerState] = useState<ServerState | null>(null);
  const [agent, setAgent] = useState<AgentCapabilities | null>(null);
  const model = agent?.name ?? null;
  const agentsCapabilities: Array<AgentCapabilities> =
    serverState?.agentsCapabilities ?? [];
  const currentAgent: AgentCapabilities | null =
    agentsCapabilities.find((agent) => agent.name === model) ?? null;

  const [serverExceptions, setServerExceptions] = useState<
    Array<ServerExceptionData>
  >([]);
  const [roomState, setRoomState] = useState<RoomState | null>(null);
  const roomID = roomState?.room_id ?? null;
  const isSpeaker =
    (clientID != null && roomState?.speakers.includes(clientID)) ?? false;
  const isListener =
    (clientID != null && roomState?.listeners.includes(clientID)) ?? false;

  const [streamingStatus, setStreamingStatus] =
    useState<StreamingStatus>('stopped');

  const isStreamConfiguredRef = useRef<boolean>(false);
  const [hasMaxSpeakers, setHasMaxSpeakers] = useState<boolean>(false);

  const [outputMode, setOutputMode] = useState<SupportedOutputMode>('s2s&t');
  const [inputSource, setInputSource] =
    useState<SupportedInputSource>('userMedia');
  const [enableNoiseSuppression, setEnableNoiseSuppression] = useState<
    boolean | null
  >(null);
  const [enableEchoCancellation, setEnableEchoCancellation] = useState<
    boolean | null
  >(null);

  // Dynamic Params:
  const [targetLang, setTargetLang] = useState<string | null>(null);
  const [enableExpressive, setEnableExpressive] = useState<boolean | null>(
    null,
  );

  const [serverDebugFlag, setServerDebugFlag] = useState<boolean>(
    debugParam ?? false,
  );

  const [receivedData, setReceivedData] = useState<Array<ServerTextData>>([]);
  const [
    translationSentencesAnimatedIndex,
    setTranslationSentencesAnimatedIndex,
  ] = useState<number>(0);

  const lastTranslationResultRef = useRef<HTMLDivElement | null>(null);

  const [inputStream, setInputStream] = useState<MediaStream | null>(null);
  const [inputStreamSource, setInputStreamSource] =
    useState<MediaStreamAudioSourceNode | null>(null);
  const audioContext = useStable<AudioContext>(() => new AudioContext());
  const [scriptNodeProcessor, setScriptNodeProcessor] =
    useState<ScriptProcessorNode | null>(null);

  const [muted, setMuted] = useState<boolean>(false);
  // The onaudioprocess script needs an up-to-date reference to the muted state, so
  // we use a ref here and keep it in sync via useEffect
  const mutedRef = useRef<boolean>(muted);
  useEffect(() => {
    mutedRef.current = muted;
  }, [muted]);

  const [gain, setGain] = useState<number>(1);

  const isScrolledToBottomRef = useRef<boolean>(isScrolledToDocumentBottom());

  // Some config options must be set when starting streaming and cannot be chaned dynamically.
  // This controls whether they are disabled or not
  const streamFixedConfigOptionsDisabled =
    streamingStatus !== 'stopped' || roomID == null;

  const bufferedSpeechPlayer = useStable(() => {
    const player = createBufferedSpeechPlayer({
      onStarted: () => {
        console.debug('📢 PLAYBACK STARTED 📢');
      },
      onEnded: () => {
        console.debug('🛑 PLAYBACK ENDED 🛑');
      },
    });

    // Start the player now so it eagerly plays audio when it arrives
    player.start();
    return player;
  });

  const translationSentencesBase: TranslationSentences =
    getTranslationSentencesFromReceivedData(receivedData);

  const translationSentencesBaseTotalLength = getTotalSentencesLength(
    translationSentencesBase,
  );

  const translationSentences: TranslationSentences = animateTextDisplay
    ? sliceTranslationSentencesUpToIndex(
        translationSentencesBase,
        translationSentencesAnimatedIndex,
      )
    : translationSentencesBase;

  // We want the blinking cursor to show before any text has arrived, so let's add an empty string so that the cursor shows up
  const translationSentencesWithEmptyStartingString =
    streamingStatus === 'running' && translationSentences.length === 0
      ? ['']
      : translationSentences;

  /******************************************
   * Event Handlers
   ******************************************/

  const setAgentAndUpdateParams = useCallback(
    (newAgent: AgentCapabilities | null) => {
      setAgent((prevAgent) => {
        if (prevAgent?.name !== newAgent?.name) {
          setTargetLang(newAgent?.targetLangs[0] ?? null);
          setEnableExpressive(null);
        }
        return newAgent;
      });
    },
    [],
  );

  const onSetDynamicConfig = useCallback(
    async (partialConfig: PartialDynamicConfig) => {
      return new Promise<void>((resolve, reject) => {
        if (socket == null) {
          reject(new Error('[onSetDynamicConfig] socket is null '));
          return;
        }

        socket.emit(
          'set_dynamic_config',
          partialConfig,
          (result: BaseResponse) => {
            console.log('[emit result: set_dynamic_config]', result);
            if (result.status === 'ok') {
              resolve();
            } else {
              reject();
            }
          },
        );
      });
    },
    [socket],
  );

  const configureStreamAsync = ({sampleRate}: {sampleRate: number}) => {
    return new Promise<void>((resolve, reject) => {
      if (socket == null) {
        reject(new Error('[configureStreamAsync] socket is null '));
        return;
      }
      const modelName = agent?.name ?? null;
      if (modelName == null) {
        reject(new Error('[configureStreamAsync] modelName is null '));
        return;
      }

      const config: StartStreamEventConfig = {
        event: 'config',
        rate: sampleRate,
        model_name: modelName,
        debug: serverDebugFlag,
        // synchronous processing isn't implemented on the v2 pubsub server, so hardcode this to true
        async_processing: true,
        buffer_limit: BUFFER_LIMIT,
        model_type: outputMode,
      };

      console.log('[configureStreamAsync] sending config', config);

      socket.emit('configure_stream', config, (statusObject) => {
        setHasMaxSpeakers(statusObject.message === 'max_speakers')
        if (statusObject.status === 'ok') {
          isStreamConfiguredRef.current = true;
          console.debug(
            '[configureStreamAsync] stream configured!',
            statusObject,
          );
          resolve();
        } else {
          isStreamConfiguredRef.current = false;
          reject(
            new Error(
              `[configureStreamAsync] configure_stream returned status: ${statusObject.status}`,
            ),
          );
          return;
        }
      });
    });
  };

  const startStreaming = async () => {
    if (streamingStatus !== 'stopped') {
      console.warn(
        `Attempting to start stream when status is ${streamingStatus}`,
      );
      return;
    }

    setStreamingStatus('starting');

    if (audioContext.state === 'suspended') {
      console.warn('audioContext was suspended! resuming...');
      await audioContext.resume();
    }

    let stream: MediaStream | null = null;

    try {
      if (inputSource === 'userMedia') {
        stream = await requestUserMediaAudioStream({
          noiseSuppression:
            enableNoiseSuppression ??
            AUDIO_STREAM_DEFAULTS['userMedia'].noiseSuppression,
          echoCancellation:
            enableEchoCancellation ??
            AUDIO_STREAM_DEFAULTS['userMedia'].echoCancellation,
        });
      } else if (inputSource === 'displayMedia') {
        stream = await requestDisplayMediaAudioStream({
          noiseSuppression:
            enableNoiseSuppression ??
            AUDIO_STREAM_DEFAULTS['displayMedia'].noiseSuppression,
          echoCancellation:
            enableEchoCancellation ??
            AUDIO_STREAM_DEFAULTS['displayMedia'].echoCancellation,
        });
      } else {
        throw new Error(`Unsupported input source requested: ${inputSource}`);
      }
      setInputStream(stream);
    } catch (e) {
      console.error('[startStreaming] media stream request failed:', e);
      setStreamingStatus('stopped');
      return;
    }

    const mediaStreamSource = audioContext.createMediaStreamSource(stream);
    setInputStreamSource(mediaStreamSource);
    /**
     * NOTE: This currently uses a deprecated way of processing the audio (createScriptProcessor), but
     * which is easy and convenient for our purposes.
     *
     * Documentation for the deprecated way of doing it is here: https://developer.mozilla.org/en-US/docs/Web/API/BaseAudioContext/createScriptProcessor
     *
     * In an ideal world this would be migrated to something like this SO answer: https://stackoverflow.com/a/65448287
     */
    const scriptProcessor = audioContext.createScriptProcessor(16384, 1, 1);
    setScriptNodeProcessor(scriptProcessor);

    scriptProcessor.onaudioprocess = (event) => {
      if (isStreamConfiguredRef.current === false) {
        console.debug('[onaudioprocess] stream is not configured yet!');
        return;
      }
      if (socket == null) {
        console.warn('[onaudioprocess] socket is null in onaudioprocess');
        return;
      }

      if (mutedRef.current) {
        // We still want to send audio to the server when we're muted to ensure we
        // get any remaining audio back from the server, so let's pass an array length 1 with a value of 0
        const mostlyEmptyInt16Array = new Int16Array(1);
        socket.emit('incoming_audio', mostlyEmptyInt16Array);
      } else {
        const float32Audio = event.inputBuffer.getChannelData(0);
        const pcm16Audio = float32To16BitPCM(float32Audio);
        socket.emit('incoming_audio', pcm16Audio);
      }

      debug()?.sentAudio(event);
    };

    mediaStreamSource.connect(scriptProcessor);
    scriptProcessor.connect(audioContext.destination);

    bufferedSpeechPlayer.start();

    try {
      if (targetLang == null) {
        throw new Error('[startStreaming] targetLang cannot be nullish');
      }

      // When we are starting the stream we want to pass all the dynamic config values
      // available before actually configuring and starting the stream
      const fullDynamicConfig: DynamicConfig = {
        targetLanguage: targetLang,
        expressive: enableExpressive,
      };

      await onSetDynamicConfig(fullDynamicConfig);

      // NOTE: this needs to be the *audioContext* sample rate, not the sample rate of the input stream. Not entirely sure why.
      await configureStreamAsync({
        sampleRate: audioContext.sampleRate,
      });
    } catch (e) {
      console.error('configureStreamAsync failed', e);
      setStreamingStatus('stopped');
      return;
    }

    setStreamingStatus('running');
  };

  const stopStreaming = useCallback(async () => {
    if (streamingStatus === 'stopped') {
      console.warn(
        `Attempting to stop stream when status is ${streamingStatus}`,
      );
      return;
    }

    // Stop the speech playback right away
    bufferedSpeechPlayer.stop();

    if (inputStreamSource == null || scriptNodeProcessor == null) {
      console.error(
        'inputStreamSource || scriptNodeProcessor is null in stopStreaming',
      );
    } else {
      inputStreamSource.disconnect(scriptNodeProcessor);
      scriptNodeProcessor.disconnect(audioContext.destination);

      // Release the mic input so we stop showing the red recording icon in the browser
      inputStream?.getTracks().forEach((track) => track.stop());
    }

    if (socket == null) {
      console.warn('Unable to emit stop_stream because socket is null');
    } else {
      socket.emit('stop_stream', (result) => {
        console.debug('[emit result: stop_stream]', result);
      });
    }

    setStreamingStatus('stopped');
  }, [
    audioContext.destination,
    bufferedSpeechPlayer,
    inputStream,
    inputStreamSource,
    scriptNodeProcessor,
    socket,
    streamingStatus,
  ]);

  const onClearTranscriptForAll = useCallback(() => {
    if (socket != null) {
      socket.emit('clear_transcript_for_all');
    }
  }, [socket]);

  /******************************************
   * Effects
   ******************************************/

  useEffect(() => {
    if (socket == null) {
      return;
    }

    const onRoomStateUpdate = (roomState: RoomState) => {
      setRoomState(roomState);
    };

    socket.on('room_state_update', onRoomStateUpdate);

    return () => {
      socket.off('room_state_update', onRoomStateUpdate);
    };
  }, [socket]);

  useEffect(() => {
    if (socket != null) {
      const onTranslationText = (data: ServerTextData) => {
        setReceivedData((prev) => [...prev, data]);
        debug()?.receivedText(data.payload);
      };

      const onTranslationSpeech = (data: ServerSpeechData) => {
        bufferedSpeechPlayer.addAudioToBuffer(data.payload, data.sample_rate);
      };

      socket.on('translation_text', onTranslationText);
      socket.on('translation_speech', onTranslationSpeech);

      return () => {
        socket.off('translation_text', onTranslationText);
        socket.off('translation_speech', onTranslationSpeech);
      };
    }
  }, [bufferedSpeechPlayer, socket]);

  useEffect(() => {
    if (socket != null) {
      const onServerStateUpdate = (newServerState: ServerState) => {
        setServerState(newServerState);

        // If a client creates a server lock, we want to stop streaming if we're not them
        if (
          newServerState.serverLock?.isActive === true &&
          newServerState.serverLock?.clientID !== clientID &&
          streamingStatus === 'running'
        ) {
          stopStreaming();
        }

        const firstAgentNullable = newServerState.agentsCapabilities[0];
        if (agent == null && firstAgentNullable != null) {
          setAgentAndUpdateParams(firstAgentNullable);
        }
      };

      socket.on('server_state_update', onServerStateUpdate);

      return () => {
        socket.off('server_state_update', onServerStateUpdate);
      };
    }
  }, [
    agent,
    clientID,
    setAgentAndUpdateParams,
    socket,
    stopStreaming,
    streamingStatus,
  ]);

  useEffect(() => {
    if (socket != null) {
      const onServerException = (
        exceptionDataWithoutClientTime: ServerExceptionData,
      ) => {
        const exceptionData = {
          ...exceptionDataWithoutClientTime,
          timeStringClient: new Date(
            exceptionDataWithoutClientTime['timeEpochMs'],
          ).toLocaleString(),
        };

        setServerExceptions((prev) =>
          [exceptionData, ...prev].slice(0, MAX_SERVER_EXCEPTIONS_TRACKED),
        );
        console.error(
          `[server_exception] The server encountered an exception: ${exceptionData['message']}`,
          exceptionData,
        );
      };

      socket.on('server_exception', onServerException);

      return () => {
        socket.off('server_exception', onServerException);
      };
    }
  }, [socket]);

  useEffect(() => {
    if (socket != null) {
      const onClearTranscript = () => {
        setReceivedData([]);
        setTranslationSentencesAnimatedIndex(0);
      };

      socket.on('clear_transcript', onClearTranscript);

      return () => {
        socket.off('clear_transcript', onClearTranscript);
      };
    }
  }, [socket]);

  useEffect(() => {
    const onScroll = () => {
      if (isScrolledToDocumentBottom(SCROLLED_TO_BOTTOM_THRESHOLD_PX)) {
        isScrolledToBottomRef.current = true;
        return;
      }
      isScrolledToBottomRef.current = false;
      return;
    };

    document.addEventListener('scroll', onScroll);

    return () => {
      document.removeEventListener('scroll', onScroll);
    };
  }, []);

  useLayoutEffect(() => {
    if (
      lastTranslationResultRef.current != null &&
      isScrolledToBottomRef.current
    ) {
      // Scroll the div to the most recent entry
      lastTranslationResultRef.current.scrollIntoView();
    }
    // Run the effect every time data is received, so that
    // we scroll to the bottom even if we're just adding text to
    // a pre-existing chunk
  }, [receivedData]);

  useEffect(() => {
    if (!animateTextDisplay) {
      return;
    }

    if (
      translationSentencesAnimatedIndex < translationSentencesBaseTotalLength
    ) {
      const timeout = setTimeout(() => {
        setTranslationSentencesAnimatedIndex((prev) => prev + 1);
        debug()?.startRenderText();
      }, TYPING_ANIMATION_DELAY_MS);

      return () => clearTimeout(timeout);
    } else {
      debug()?.endRenderText();
    }
  }, [
    animateTextDisplay,
    translationSentencesAnimatedIndex,
    translationSentencesBaseTotalLength,
  ]);

  /******************************************
   * Sub-components
   ******************************************/

  const volumeSliderNode = (
    <Stack
      spacing={2}
      direction="row"
      sx={{mb: 1, width: '100%'}}
      alignItems="center">
      <VolumeDown color="primary" />
      <Slider
        aria-label="Volume"
        defaultValue={1}
        scale={getGainScaledValue}
        min={0}
        max={3}
        step={0.1}
        marks={[
          {value: 0, label: '0%'},
          {value: 1, label: '100%'},
          {value: 2, label: '400%'},
          {value: 3, label: '700%'},
        ]}
        valueLabelFormat={(value) => `${(value * 100).toFixed(0)}%`}
        valueLabelDisplay="auto"
        value={gain}
        onChange={(_event: Event, newValue: number | number[]) => {
          if (typeof newValue === 'number') {
            const scaledGain = getGainScaledValue(newValue);
            // We want the actual gain node to use the scaled value
            bufferedSpeechPlayer.setGain(scaledGain);
            // But we want react state to keep track of the non-scaled value
            setGain(newValue);
          } else {
            console.error(
              `[volume slider] Unexpected non-number value: ${newValue}`,
            );
          }
        }}
      />
      <VolumeUp color="primary" />
    </Stack>
  );

  const xrDialogComponent = (
    <XRDialog
      animateTextDisplay={
        animateTextDisplay &&
        translationSentencesAnimatedIndex == translationSentencesBaseTotalLength
      }
      bufferedSpeechPlayer={bufferedSpeechPlayer}
      translationSentences={translationSentences}
      roomState={roomState}
      roomID={roomID}
      startStreaming={startStreaming}
      stopStreaming={stopStreaming}
      debugParam={debugParam}
      onARHidden={() => {
        setAnimateTextDisplay(urlParams.animateTextDisplay);
      }}
      onARVisible={() => setAnimateTextDisplay(false)}
    />
  );

  return (
    <div className="app-wrapper-sra">
      <Box
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-ignore Not sure why it's complaining about complexity here
        sx={{width: '100%', maxWidth: '660px', minWidth: '320px'}}>
        <div className="main-container-sra">
          <div className="top-section-sra horizontal-padding-sra">
            <div className="header-container-sra">
              <img
                src={seamlessLogoUrl}
                className="header-icon-sra"
                alt="Seamless Translation Logo"
                height={24}
                width={24}
              />

              <div>
                <Typography variant="h1" sx={{color: '#65676B'}}>
                  Seamless Translation
                </Typography>
              </div>
            </div>
            <div className="header-container-sra">
              <div>
                <Typography variant="body2" sx={{color: '#65676B'}}>
                  Welcome! This space is limited to one speaker at a time. 
                  If using the live HF space, sharing room code to listeners on another 
                  IP address may not work because it's running on different replicas. 
                  Use headphones if you are both speaker and listener to prevent feedback.
                  <br/>
                  If max speakers reached, please duplicate the space <a target="_blank" rel="noopener noreferrer" href="https://huggingface.co/spaces/facebook/seamless-streaming?duplicate=true">here</a>. 
                  In your duplicated space, join a room as speaker or listener (or both), 
                  and share the room code to invite listeners.
                  <br/>
                  Check out the seamless_communication <a target="_blank" rel="noopener noreferrer" href="https://github.com/facebookresearch/seamless_communication/tree/main">README</a> for more information.
                  <br/>
                  SeamlessStreaming model is a research model and is not released
                  for production deployment. It is important to use a microphone with 
                  noise cancellation (for e.g. a smartphone), otherwise you may see model hallucination on noises. 
                  It works best if you pause every couple of sentences, or you may wish adjust the VAD threshold
                  in the model config. The real-time performance will degrade
                  if you try streaming multiple speakers at the same time.
                </Typography>
              </div>
            </div>
            <Stack spacing="22px" direction="column">
              <Box>
                <RoomConfig
                  roomState={roomState}
                  serverState={serverState}
                  streamingStatus={streamingStatus}
                  onJoinRoomOrUpdateRoles={() => {
                    // If the user has switched from speaker to listener we need to tell the
                    // player to play eagerly, since currently the listener doesn't have any stop/start controls
                    bufferedSpeechPlayer.start();
                  }}
                />

                {isListener && !isSpeaker && (
                  <Box
                    sx={{
                      paddingX: 6,
                      paddingBottom: 2,
                      marginY: 2,
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                    }}>
                    {volumeSliderNode}
                  </Box>
                )}
              </Box>

              {isSpeaker && (
                <>
                  <Divider />

                  <Stack spacing="12px" direction="column">
                    <FormLabel id="output-modes-radio-group-label">
                      Model
                    </FormLabel>
                    <FormControl
                      disabled={
                        streamFixedConfigOptionsDisabled ||
                        agentsCapabilities.length === 0
                      }
                      fullWidth
                      sx={{minWidth: '14em'}}>
                      <InputLabel id="model-selector-input-label">
                        Model
                      </InputLabel>
                      <Select
                        labelId="model-selector-input-label"
                        label="Model"
                        onChange={(e: SelectChangeEvent) => {
                          const newAgent =
                            agentsCapabilities.find(
                              (agent) => e.target.value === agent.name,
                            ) ?? null;
                          if (newAgent == null) {
                            console.error(
                              'Unable to find agent with name',
                              e.target.value,
                            );
                          }
                          setAgentAndUpdateParams(newAgent);
                        }}
                        value={model ?? ''}>
                        {agentsCapabilities.map((agent) => (
                          <MenuItem value={agent.name} key={agent.name}>
                            {agent.name}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>

                  </Stack>

                  <Stack spacing={0.5}>
                    <FormLabel id="output-modes-radio-group-label">
                      Output
                    </FormLabel>

                    <Box sx={{paddingTop: 2, paddingBottom: 1}}>
                      <FormControl fullWidth sx={{minWidth: '14em'}}>
                        <InputLabel id="target-selector-input-label">
                          Target Language
                        </InputLabel>
                        <Select
                          labelId="target-selector-input-label"
                          label="Target Language"
                          onChange={(e: SelectChangeEvent) => {
                            setTargetLang(e.target.value);
                            onSetDynamicConfig({
                              targetLanguage: e.target.value,
                            });
                          }}
                          value={targetLang ?? ''}>
                          {currentAgent?.targetLangs.map((langCode) => (
                            <MenuItem value={langCode} key={langCode}>
                              {getLanguageFromThreeLetterCode(langCode) != null
                                ? `${getLanguageFromThreeLetterCode(
                                    langCode,
                                  )} (${langCode})`
                                : langCode}
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                    </Box>

                    <Grid container>
                      <Grid item xs={12} sm={4}>
                        <FormControl
                          disabled={streamFixedConfigOptionsDisabled}>
                          <RadioGroup
                            aria-labelledby="output-modes-radio-group-label"
                            value={outputMode}
                            onChange={(e) =>
                              setOutputMode(
                                e.target.value as SupportedOutputMode,
                              )
                            }
                            name="output-modes-radio-buttons-group">
                            {
                              // TODO: Use supported modalities from agentCapabilities
                              SUPPORTED_OUTPUT_MODES.map(({value, label}) => (
                                <FormControlLabel
                                  key={value}
                                  value={value}
                                  control={<Radio />}
                                  label={label}
                                />
                              ))
                            }
                          </RadioGroup>
                        </FormControl>
                      </Grid>

                      <Grid item xs={12} sm={8}>
                        <Stack
                          direction="column"
                          spacing={1}
                          alignItems="flex-start"
                          sx={{flexGrow: 1}}>
                          {currentAgent?.dynamicParams?.includes(
                            'expressive',
                          ) && (
                            <FormControlLabel
                              control={
                                <Switch
                                  checked={enableExpressive ?? false}
                                  onChange={(
                                    event: React.ChangeEvent<HTMLInputElement>,
                                  ) => {
                                    const newValue = event.target.checked;
                                    setEnableExpressive(newValue);
                                    onSetDynamicConfig({
                                      expressive: newValue,
                                    });
                                  }}
                                />
                              }
                              label="Expressive"
                            />
                          )}

                          {isListener && (
                            <Box
                              sx={{
                                flexGrow: 1,
                                paddingX: 1.5,
                                paddingY: 1.5,
                                width: '100%',
                              }}>
                              {volumeSliderNode}
                            </Box>
                          )}
                        </Stack>
                      </Grid>
                    </Grid>
                  </Stack>

                  <Stack
                    direction="row"
                    spacing={2}
                    justifyContent="space-between">
                    <Box sx={{flex: 1}}>
                      <FormControl disabled={streamFixedConfigOptionsDisabled}>
                        <FormLabel id="input-source-radio-group-label">
                          Input Source
                        </FormLabel>
                        <RadioGroup
                          aria-labelledby="input-source-radio-group-label"
                          value={inputSource}
                          onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                            setInputSource(
                              e.target.value as SupportedInputSource,
                            )
                          }
                          name="input-source-radio-buttons-group">
                          {SUPPORTED_INPUT_SOURCES.map(({label, value}) => (
                            <FormControlLabel
                              key={value}
                              value={value}
                              control={<Radio />}
                              label={label}
                            />
                          ))}
                        </RadioGroup>
                      </FormControl>
                    </Box>

                    <Box sx={{flex: 1, flexGrow: 2}}>
                    <FormControl disabled={streamFixedConfigOptionsDisabled}>
                        <FormLabel>Options</FormLabel>
                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={
                                enableNoiseSuppression ??
                                AUDIO_STREAM_DEFAULTS[inputSource]
                                  .noiseSuppression
                              }
                              onChange={(
                                event: React.ChangeEvent<HTMLInputElement>,
                              ) =>
                                setEnableNoiseSuppression(event.target.checked)
                              }
                            />
                          }
                          label="Noise Suppression"
                        />
                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={
                                enableEchoCancellation ??
                                AUDIO_STREAM_DEFAULTS[inputSource]
                                  .echoCancellation
                              }
                              onChange={(
                                event: React.ChangeEvent<HTMLInputElement>,
                              ) =>
                                setEnableEchoCancellation(event.target.checked)
                              }
                            />
                          }
                          label="Echo Cancellation (not recommended)"
                        />
                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={serverDebugFlag}
                              onChange={(
                                event: React.ChangeEvent<HTMLInputElement>,
                              ) => setServerDebugFlag(event.target.checked)}
                            />
                          }
                          label="Enable Server Debugging"
                        />
                      </FormControl>
                    </Box>
                  </Stack>

                  {isSpeaker &&
                    isListener &&
                    inputSource === 'userMedia' &&
                    !enableEchoCancellation &&
                    gain !== 0 && (
                      <div>
                        <Alert severity="warning" icon={<HeadphonesIcon />}>
                          Headphones required to prevent feedback.
                        </Alert>
                      </div>
                    )}

                  {isSpeaker && enableEchoCancellation && (
                    <div>
                      <Alert severity="warning">
                        We don't recommend using echo cancellation as it may
                        distort the input audio. If possible, use headphones and
                        disable echo cancellation instead.
                      </Alert>
                    </div>
                  )}

                  <Stack direction="row" spacing={2}>
                    {streamingStatus === 'stopped' ? (
                      <Button
                        variant="contained"
                        onClick={startStreaming}
                        disabled={
                          roomID == null ||
                          // Prevent users from starting streaming if there is a server lock with an active session
                          (serverState?.serverLock?.isActive === true &&
                            serverState.serverLock.clientID !== clientID)
                        }>
                        {buttonLabelMap[streamingStatus]}
                      </Button>
                    ) : (
                      <Button
                        variant="contained"
                        color={
                          streamingStatus === 'running' ? 'error' : 'primary'
                        }
                        disabled={
                          streamingStatus === 'starting' || roomID == null
                        }
                        onClick={stopStreaming}>
                        {buttonLabelMap[streamingStatus]}
                      </Button>
                    )}

                    <Box>
                      <Button
                        variant="contained"
                        aria-label={muted ? 'Unmute' : 'Mute'}
                        color={muted ? 'info' : 'primary'}
                        onClick={() => setMuted((prev) => !prev)}
                        sx={{
                          borderRadius: 100,
                          paddingX: 0,
                          minWidth: '36px',
                        }}>
                        {muted ? <MicOff /> : <Mic />}
                      </Button>
                    </Box>

                    {roomID == null ? null : (
                      <Box
                        sx={{
                          flexGrow: 1,
                          display: 'flex',
                          justifyContent: 'flex-end',
                        }}>
                        {xrDialogComponent}
                      </Box>
                    )}
                  </Stack>

                  {serverExceptions.length > 0 && (
                    <div>
                      <Alert severity="error">
                        {`The server encountered an exception. See the browser console for details. You may need to refresh the page to continue using the app.`}
                      </Alert>
                    </div>
                  )}
                  {serverState != null && hasMaxSpeakers && (
                    <div>
                      <Alert severity="error">
                        {`Maximum number of speakers reached. Please try again at a later time.`}
                      </Alert>
                    </div>
                  )}
                  {serverState != null &&
                    serverState.totalActiveTranscoders >=
                      TOTAL_ACTIVE_TRANSCODER_WARNING_THRESHOLD && (
                      <div>
                        <Alert severity="warning">
                          {`The server currently has ${serverState?.totalActiveTranscoders} active streaming sessions. Performance may be degraded.`}
                        </Alert>
                      </div>
                    )}

                  {serverState?.serverLock != null &&
                    serverState.serverLock.clientID !== clientID && (
                      <div>
                        <Alert severity="warning">
                          {`The server is currently locked. Priority will be given to that client when they are streaming, and your streaming session may be halted abruptly.`}
                        </Alert>
                      </div>
                    )}
                </>
              )}
            </Stack>

            {isListener && !isSpeaker && (
              <Box sx={{marginBottom: 1, marginTop: 2}}>
                {xrDialogComponent}
              </Box>
            )}
          </div>

          {debugParam && roomID != null && <DebugSection />}

          <div className="translation-text-container-sra horizontal-padding-sra">
            <Stack
              direction="row"
              spacing={2}
              sx={{mb: '16px', alignItems: 'center'}}>
              <Typography variant="h1" sx={{fontWeight: 700, flexGrow: 1}}>
                Transcript
              </Typography>
              {isSpeaker && (
                <Button
                  variant="text"
                  size="small"
                  onClick={onClearTranscriptForAll}>
                  Clear Transcript for All
                </Button>
              )}
            </Stack>
            <Stack direction="row">
              <div className="translation-text-sra">
                {translationSentencesWithEmptyStartingString.map(
                  (sentence, index, arr) => {
                    const isLast = index === arr.length - 1;
                    const maybeRef = isLast
                      ? {ref: lastTranslationResultRef}
                      : {};
                    return (
                      <div className="text-chunk-sra" key={index} {...maybeRef}>
                        <Typography variant="body1">
                          {sentence}
                          {animateTextDisplay && isLast && (
                            <Blink
                              intervalMs={CURSOR_BLINK_INTERVAL_MS}
                              shouldBlink={
                                (roomState?.activeTranscoders ?? 0) > 0
                              }>
                              <Typography
                                component="span"
                                variant="body1"
                                sx={{
                                  display: 'inline-block',
                                  transform: 'scaleY(1.25) translateY(-1px)',
                                }}>
                                {'|'}
                              </Typography>
                            </Blink>
                          )}
                        </Typography>
                      </div>
                    );
                  },
                )}
              </div>
            </Stack>
          </div>
        </div>
      </Box>
    </div>
  );
}
