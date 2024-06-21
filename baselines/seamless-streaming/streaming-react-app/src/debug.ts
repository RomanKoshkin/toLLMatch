import {TYPING_ANIMATION_DELAY_MS} from './StreamingInterface';
import {getURLParams} from './URLParams';
import audioBuffertoWav from 'audiobuffer-to-wav';
import './StreamingInterface.css';

type StartEndTime = {
  start: number;
  end: number;
};

type StartEndTimeWithAudio = StartEndTime & {
  float32Audio: Float32Array;
};

type Text = {
  time: number;
  chars: number;
};

type DebugTimings = {
  receivedAudio: StartEndTime[];
  playedAudio: StartEndTimeWithAudio[];
  receivedText: Text[];
  renderedText: StartEndTime[];
  sentAudio: StartEndTimeWithAudio[];
  startRenderTextTime: number | null;
  startRecordingTime: number | null;
  receivedAudioSampleRate: number | null;
};

function getInitialTimings(): DebugTimings {
  return {
    receivedAudio: [],
    playedAudio: [],
    receivedText: [],
    renderedText: [],
    sentAudio: [],
    startRenderTextTime: null,
    startRecordingTime: null,
    receivedAudioSampleRate: null,
  };
}

function downloadAudioBuffer(audioBuffer: AudioBuffer, fileName: string): void {
  const wav = audioBuffertoWav(audioBuffer);
  const wavBlob = new Blob([new DataView(wav)], {
    type: 'audio/wav',
  });
  const url = URL.createObjectURL(wavBlob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.target = '_blank';
  anchor.download = fileName;
  anchor.click();
}

// Uncomment for debugging without download
// function playAudioBuffer(audioBuffer: AudioBuffer): void {
//   const audioContext = new AudioContext();
//   const source = audioContext.createBufferSource();

//   source.buffer = audioBuffer;
//   source.connect(audioContext.destination);
//   source.start();
// }

// Accumulate timings and audio / text translation samples for debugging and exporting
class DebugTimingsManager {
  timings: DebugTimings = getInitialTimings();

  start(): void {
    this.timings = getInitialTimings();
    this.timings.startRecordingTime = new Date().getTime();
  }

  sentAudio(event: AudioProcessingEvent): void {
    const end = new Date().getTime();
    const start = end - event.inputBuffer.duration * 1000;
    // Copy or else buffer seems to be re-used
    const float32Audio = new Float32Array(event.inputBuffer.getChannelData(0));
    this.timings.sentAudio.push({
      start,
      end,
      float32Audio,
    });
  }

  receivedText(text: string): void {
    this.timings.receivedText.push({
      time: new Date().getTime(),
      chars: text.length,
    });
  }

  startRenderText(): void {
    if (this.timings.startRenderTextTime == null) {
      this.timings.startRenderTextTime = new Date().getTime();
    }
  }

  endRenderText(): void {
    if (this.timings.startRenderTextTime == null) {
      console.warn(
        'Wrong timings of start / end rendering text. startRenderText is null',
      );
      return;
    }

    this.timings.renderedText.push({
      start: this.timings.startRenderTextTime as number,
      end: new Date().getTime(),
    });
    this.timings.startRenderTextTime = null;
  }

  receivedAudio(duration: number): void {
    const start = new Date().getTime();
    this.timings.receivedAudio.push({
      start,
      end: start + duration * 1000,
    });
  }

  playedAudio(start: number, end: number, buffer: AudioBuffer | null): void {
    if (buffer != null) {
      if (this.timings.receivedAudioSampleRate == null) {
        this.timings.receivedAudioSampleRate = buffer.sampleRate;
      }
      if (this.timings.receivedAudioSampleRate != buffer.sampleRate) {
        console.error(
          'Sample rates of received audio are unequal, will fail to reconstruct debug audio',
          this.timings.receivedAudioSampleRate,
          buffer.sampleRate,
        );
      }
    }
    this.timings.playedAudio.push({
      start,
      end,
      float32Audio:
        buffer == null
          ? new Float32Array()
          : new Float32Array(buffer.getChannelData(0)),
    });
  }

  getChartData() {
    const columns = [
      {type: 'string', id: 'Series'},
      {type: 'date', id: 'Start'},
      {type: 'date', id: 'End'},
    ];
    return [
      columns,
      ...this.timings.sentAudio.map((sentAudio) => [
        'Sent Audio',
        new Date(sentAudio.start),
        new Date(sentAudio.end),
      ]),
      ...this.timings.receivedAudio.map((receivedAudio) => [
        'Received Audio',
        new Date(receivedAudio.start),
        new Date(receivedAudio.end),
      ]),
      ...this.timings.playedAudio.map((playedAudio) => [
        'Played Audio',
        new Date(playedAudio.start),
        new Date(playedAudio.end),
      ]),
      // Best estimate duration by multiplying length with animation duration for each letter
      ...this.timings.receivedText.map((receivedText) => [
        'Received Text',
        new Date(receivedText.time),
        new Date(
          receivedText.time + receivedText.chars * TYPING_ANIMATION_DELAY_MS,
        ),
      ]),
      ...this.timings.renderedText.map((renderedText) => [
        'Rendered Text',
        new Date(renderedText.start),
        new Date(renderedText.end),
      ]),
    ];
  }

  downloadInputAudio() {
    const audioContext = new AudioContext();
    const totalLength = this.timings.sentAudio.reduce((acc, cur) => {
      return acc + cur?.float32Audio?.length ?? 0;
    }, 0);
    if (totalLength === 0) {
      return;
    }

    const incomingArrayBuffer = audioContext.createBuffer(
      1, // 1 channel
      totalLength,
      audioContext.sampleRate,
    );

    const buffer = incomingArrayBuffer.getChannelData(0);
    let i = 0;
    this.timings.sentAudio.forEach((sentAudio) => {
      sentAudio.float32Audio.forEach((bytes) => {
        buffer[i++] = bytes;
      });
    });

    // Play for debugging
    // playAudioBuffer(incomingArrayBuffer);
    downloadAudioBuffer(incomingArrayBuffer, `input_audio.wav`);
  }

  downloadOutputAudio() {
    const playedAudio = this.timings.playedAudio;
    const sampleRate = this.timings.receivedAudioSampleRate;
    if (
      playedAudio.length === 0 ||
      this.timings.startRecordingTime == null ||
      sampleRate == null
    ) {
      return null;
    }

    let previousEndTime = this.timings.startRecordingTime;
    const audioArray: number[] = [];
    playedAudio.forEach((audio) => {
      const delta = (audio.start - previousEndTime) / 1000;
      for (let i = 0; i < delta * sampleRate; i++) {
        audioArray.push(0.0);
      }
      audio.float32Audio.forEach((bytes) => audioArray.push(bytes));
      previousEndTime = audio.end;
    });
    const audioContext = new AudioContext();
    const incomingArrayBuffer = audioContext.createBuffer(
      1, // 1 channel
      audioArray.length,
      sampleRate,
    );

    incomingArrayBuffer.copyToChannel(
      new Float32Array(audioArray),
      0, // first channel
    );

    // Play for debugging
    // playAudioBuffer(incomingArrayBuffer);
    downloadAudioBuffer(incomingArrayBuffer, 'output_audio.wav');
  }
}

const debugSingleton = new DebugTimingsManager();
export default function debug(): DebugTimingsManager | null {
  const debugParam = getURLParams().debug;
  return debugParam ? debugSingleton : null;
}
