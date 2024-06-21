import debug from './debug';

type AddAudioToBufferFunction = (
  samples: Array<number>,
  sampleRate: number,
) => void;

export type BufferedSpeechPlayer = {
  addAudioToBuffer: AddAudioToBufferFunction;
  setGain: (gain: number) => void;
  start: () => void;
  stop: () => void;
};

type Options = {
  onEnded?: () => void;
  onStarted?: () => void;
};

export default function createBufferedSpeechPlayer({
  onStarted,
  onEnded,
}: Options): BufferedSpeechPlayer {
  const audioContext = new AudioContext();
  const gainNode = audioContext.createGain();
  gainNode.connect(audioContext.destination);

  let unplayedAudioBuffers: Array<AudioBuffer> = [];

  let currentPlayingBufferSource: AudioBufferSourceNode | null = null;

  let isPlaying = false;

  // This means that the player starts in the 'stopped' state, and you need to call player.start() for it to start playing
  let shouldPlayWhenAudioAvailable = false;

  const setGain = (gain: number) => {
    gainNode.gain.setValueAtTime(gain, audioContext.currentTime);
  };

  const start = () => {
    shouldPlayWhenAudioAvailable = true;
    debug()?.start();
    playNextBufferIfNotAlreadyPlaying();
  };

  // Stop will stop the audio and clear the buffers
  const stop = () => {
    shouldPlayWhenAudioAvailable = false;

    // Stop the current buffers
    currentPlayingBufferSource?.stop();
    currentPlayingBufferSource = null;

    unplayedAudioBuffers = [];

    onEnded != null && onEnded();
    isPlaying = false;
    return;
  };

  const playNextBufferIfNotAlreadyPlaying = () => {
    if (!isPlaying) {
      playNextBuffer();
    }
  };

  const playNextBuffer = () => {
    if (shouldPlayWhenAudioAvailable === false) {
      console.debug(
        '[BufferedSpeechPlayer][playNextBuffer] Not playing any more audio because shouldPlayWhenAudioAvailable is false.',
      );
      // NOTE: we do not need to set isPlaying = false or call onEnded because that will be handled in the stop() function
      return;
    }
    if (unplayedAudioBuffers.length === 0) {
      console.debug(
        '[BufferedSpeechPlayer][playNextBuffer] No buffers to play.',
      );
      if (isPlaying) {
        isPlaying = false;
        onEnded != null && onEnded();
      }
      return;
    }

    // If isPlaying is false, then we are starting playback fresh rather than continuing it, and should call onStarted
    if (isPlaying === false) {
      isPlaying = true;
      onStarted != null && onStarted();
    }

    const source = audioContext.createBufferSource();

    // Get the first unplayed buffer from the array, and remove it from the array
    const buffer = unplayedAudioBuffers.shift() ?? null;
    source.buffer = buffer;
    console.debug(
      `[BufferedSpeechPlayer] Playing buffer with ${source.buffer?.length} samples`,
    );

    source.connect(gainNode);

    const startTime = new Date().getTime();
    source.start();
    currentPlayingBufferSource = source;
    // This is probably not necessary, but it doesn't hurt
    isPlaying = true;

    // TODO: consider changing this to a while loop to avoid deep recursion
    const onThisBufferPlaybackEnded = () => {
      console.debug(
        `[BufferedSpeechPlayer] Buffer with ${source.buffer?.length} samples ended.`,
      );
      source.removeEventListener('ended', onThisBufferPlaybackEnded);
      const endTime = new Date().getTime();
      debug()?.playedAudio(startTime, endTime, buffer);
      currentPlayingBufferSource = null;

      // We don't set isPlaying = false here because we are attempting to continue playing. It will get set to false if there are no more buffers to play
      playNextBuffer();
    };

    source.addEventListener('ended', onThisBufferPlaybackEnded);
  };

  const addAudioToBuffer: AddAudioToBufferFunction = (samples, sampleRate) => {
    const incomingArrayBufferChunk = audioContext.createBuffer(
      // 1 channel
      1,
      samples.length,
      sampleRate,
    );

    incomingArrayBufferChunk.copyToChannel(
      new Float32Array(samples),
      // first channel
      0,
    );

    console.debug(
      `[addAudioToBufferAndPlay] Adding buffer with ${incomingArrayBufferChunk.length} samples to queue.`,
    );

    unplayedAudioBuffers.push(incomingArrayBufferChunk);
    debug()?.receivedAudio(
      incomingArrayBufferChunk.length / incomingArrayBufferChunk.sampleRate,
    );
    const audioBuffersTableInfo = unplayedAudioBuffers.map((buffer, i) => {
      return {
        index: i,
        duration: buffer.length / buffer.sampleRate,
        samples: buffer.length,
      };
    });
    const totalUnplayedDuration = unplayedAudioBuffers.reduce((acc, buffer) => {
      return acc + buffer.length / buffer.sampleRate;
    }, 0);

    console.debug(
      `[addAudioToBufferAndPlay] Current state of incoming audio buffers (${totalUnplayedDuration.toFixed(
        1,
      )}s unplayed):`,
    );
    console.table(audioBuffersTableInfo);

    if (shouldPlayWhenAudioAvailable) {
      playNextBufferIfNotAlreadyPlaying();
    }
  };

  return {addAudioToBuffer, setGain, stop, start};
}
