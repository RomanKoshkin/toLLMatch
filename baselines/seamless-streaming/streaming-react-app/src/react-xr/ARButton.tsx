import * as THREE from 'three';
import {Button} from '@mui/material';
import {useCallback, useEffect, useState} from 'react';
import {BufferedSpeechPlayer} from '../createBufferedSpeechPlayer';

type Props = {
  bufferedSpeechPlayer: BufferedSpeechPlayer;
  renderer: THREE.WebGLRenderer | null;
  onARVisible?: () => void;
  onARHidden?: () => void;
};

export default function ARButton({
  bufferedSpeechPlayer,
  renderer,
  onARVisible,
  onARHidden,
}: Props) {
  const [session, setSession] = useState<XRSession | null>(null);
  const [supported, setSupported] = useState<boolean>(true);

  useEffect(() => {
    if (!navigator.xr) {
      setSupported(false);
      return;
    }
    navigator.xr.isSessionSupported('immersive-ar').then((supported) => {
      setSupported(supported);
    });
  }, []);

  const resetBuffers = useCallback(
    (event: XRSessionEvent) => {
      const session = event.target;
      if (!(session instanceof XRSession)) {
        return;
      }
      switch (session.visibilityState) {
        case 'visible':
          console.log('Restarting speech player, device is visible');
          bufferedSpeechPlayer.stop();
          bufferedSpeechPlayer.start();
          onARVisible?.();
          break;
        case 'hidden':
          console.log('Stopping speech player, device is hidden');
          bufferedSpeechPlayer.stop();
          bufferedSpeechPlayer.start();
          onARHidden?.();
          break;
      }
    },
    [bufferedSpeechPlayer],
  );

  async function onSessionStarted(session: XRSession) {
    setSession(session);

    session.onvisibilitychange = resetBuffers;
    session.onend = onSessionEnded;

    await renderer.xr.setSession(session);
  }

  function onSessionEnded() {
    setSession(null);
  }

  const onClick = () => {
    if (session === null) {
      navigator.xr!.requestSession('immersive-ar').then(onSessionStarted);
    } else {
      session.end();
    }
  };
  return (
    <Button
      variant="contained"
      onClick={onClick}
      disabled={!supported || renderer == null}
      sx={{mt: 1}}>
      {supported
        ? renderer != null
          ? 'Enter AR'
          : 'Initializing AR...'
        : 'AR Not Supported'}
    </Button>
  );
}
