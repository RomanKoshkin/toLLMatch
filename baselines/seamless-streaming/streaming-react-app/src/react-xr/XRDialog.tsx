import * as THREE from 'three';
import {
  Button,
  Dialog,
  DialogContent,
  DialogTitle,
  IconButton,
  Typography,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import {XRConfigProps} from './XRConfig';
import {useEffect, useRef, useState} from 'react';
import './XRDialog.css';
import {getRenderer, init, updatetranslationText} from './XRRendering';
import ARButton from './ARButton';
import {getURLParams} from '../URLParams';

function XRContent(props: XRConfigProps) {
  const debugParam = getURLParams().debug;
  const {translationSentences} = props;
  useEffect(() => {
    updatetranslationText(translationSentences);
  }, [translationSentences]);

  const [renderer, setRenderer] = useState<THREE.WebGLRenderer | null>(null);
  const canvasRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (canvasRef.current != null || debugParam === false) {
      const existingRenderer = getRenderer();
      if (existingRenderer) {
        setRenderer(existingRenderer);
      } else {
        const newRenderer = init(
          400,
          300,
          debugParam ? canvasRef.current : null,
        );
        setRenderer(newRenderer);
      }
    }
  }, [canvasRef.current]);

  return (
    <DialogContent
      dividers
      className="xr-dialog-container xr-dialog-text-center">
      <Typography gutterBottom>
        Welcome to the Seamless team streaming demo experience! In this demo you
        will experience AI powered text and audio translation in real time.
      </Typography>
      <div ref={canvasRef} className="xr-dialog-canvas-container" />
      <ARButton
        bufferedSpeechPlayer={props.bufferedSpeechPlayer}
        renderer={renderer}
        onARHidden={props.onARHidden}
        onARVisible={props.onARVisible}
      />
    </DialogContent>
  );
}

export default function XRDialog(props: XRConfigProps) {
  const [isDialogOpen, setIsDialogOpen] = useState<boolean>(false);

  return (
    <>
      <Button variant="contained" onClick={() => setIsDialogOpen(true)}>
        Enter AR Experience
      </Button>
      {isDialogOpen && (
        <Dialog onClose={() => setIsDialogOpen(false)} open={true}>
          <DialogTitle sx={{m: 0, p: 2}} className="xr-dialog-text-center">
            FAIR Seamless Streaming Demo
          </DialogTitle>
          <IconButton
            aria-label="close"
            onClick={() => setIsDialogOpen(false)}
            sx={{
              position: 'absolute',
              right: 8,
              top: 8,
              color: (theme) => theme.palette.grey[500],
            }}>
            <CloseIcon />
          </IconButton>
          <XRContent {...props} />
        </Dialog>
      )}
    </>
  );
}
