import Stack from '@mui/material/Stack';
import TextField from '@mui/material/TextField';
import {isValidRoomID, isValidPartialRoomID} from './generateNewRoomID';
import {useCallback, useEffect, useState} from 'react';
import Button from '@mui/material/Button';
import {useSocket} from './useSocket';
import FormGroup from '@mui/material/FormGroup';
import FormControlLabel from '@mui/material/FormControlLabel';
import Checkbox from '@mui/material/Checkbox';
import {RoomState} from './types/RoomState';
import setURLParam from './setURLParam';
import {getURLParams} from './URLParams';
import {
  JoinRoomConfig,
  Roles,
  ServerState,
  StreamingStatus,
} from './types/StreamingTypes';
import Alert from '@mui/material/Alert';

function capitalize(str: string): string {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

type Props = {
  roomState: RoomState | null;
  serverState: ServerState | null;
  onJoinRoomOrUpdateRoles?: () => void;
  streamingStatus: StreamingStatus;
};

export default function RoomConfig({
  roomState,
  serverState,
  onJoinRoomOrUpdateRoles,
  streamingStatus,
}: Props) {
  const {socket, clientID} = useSocket();

  const urlParams = getURLParams();
  const roomIDParam = urlParams.roomID;
  const autoJoinRoom = urlParams.autoJoin;

  const [roomID, setRoomID] = useState<string>(
    (roomIDParam ?? '').toUpperCase(),
  );
  const [roomIDError, setRoomIDError] = useState<boolean>(false);
  const [roles, setRoles] = useState<{speaker: boolean; listener: boolean}>({
    speaker: true,
    listener: true,
  });
  const [lockServer, setLockServer] = useState<boolean>(false);
  const [lockServerName, setLockServerName] = useState<string>('');

  const [joinInProgress, setJoinInProgress] = useState<boolean>(false);
  const [didAttemptAutoJoin, setDidAttemptAutoJoin] = useState<boolean>(false);

  const isValidServerLock =
    lockServer === false ||
    (lockServerName != null && lockServerName.length > 0);
  const isValidRoles = Object.values(roles).filter(Boolean).length > 0;
  const isValidAllInputs =
    isValidRoomID(roomID) && isValidRoles && isValidServerLock;
  const roomIDFromServer = roomState?.room_id ?? null;

  const onJoinRoom = useCallback(
    (createNewRoom: boolean) => {
      if (socket == null) {
        console.error('Socket is null, cannot join room');
        return;
      }
      console.debug(`Attempting to join roomID ${roomID}...`);

      const lockServerValidated: string | null =
        lockServer && roles['speaker'] ? lockServerName : null;

      setJoinInProgress(true);

      const configObject: JoinRoomConfig = {
        roles: (Object.keys(roles) as Array<Roles>).filter(
          (role) => roles[role] === true,
        ),
        lockServerName: lockServerValidated,
      };

      socket.emit(
        'join_room',
        clientID,
        createNewRoom ? null : roomID,
        configObject,
        (result) => {
          console.log('join_room result:', result);
          if (createNewRoom) {
            setRoomID(result.roomID);
          }
          if (onJoinRoomOrUpdateRoles != null) {
            onJoinRoomOrUpdateRoles();
          }
          setURLParam('roomID', result.roomID);
          setJoinInProgress(false);
        },
      );
    },
    [
      clientID,
      lockServer,
      lockServerName,
      onJoinRoomOrUpdateRoles,
      roles,
      roomID,
      socket,
    ],
  );

  useEffect(() => {
    if (
      autoJoinRoom === true &&
      didAttemptAutoJoin === false &&
      socket != null
    ) {
      // We want to consider this an attempt whether or not we actually try to join, because
      // we only want auto-join to happen on initial load
      setDidAttemptAutoJoin(true);
      if (
        isValidAllInputs &&
        joinInProgress === false &&
        roomIDFromServer == null
      ) {
        console.debug('Attempting to auto-join room...');

        onJoinRoom(false);
      } else {
        console.debug('Unable to auto-join room', {
          isValidAllInputs,
          joinInProgress,
          roomIDFromServer,
        });
      }
    }
  }, [
    autoJoinRoom,
    didAttemptAutoJoin,
    isValidAllInputs,
    joinInProgress,
    onJoinRoom,
    roomIDFromServer,
    socket,
  ]);

  return (
    <Stack direction="column" spacing="12px">
      <Stack direction="row" spacing="12px" sx={{alignItems: 'center'}}>
        <TextField
          size="small"
          label="Room Code"
          variant="outlined"
          disabled={roomState?.room_id != null}
          value={roomID}
          error={roomIDError}
          onChange={(e) => {
            const id = e.target.value.toUpperCase();
            if (isValidPartialRoomID(id)) {
              setRoomIDError(false);
              setRoomID(id);
            } else {
              setRoomIDError(true);
            }
          }}
          sx={{width: '8em'}}
        />

        <div>
          <Button
            variant="contained"
            disabled={
              isValidAllInputs === false ||
              joinInProgress ||
              streamingStatus !== 'stopped'
            }
            onClick={() => onJoinRoom(false)}>
            {roomState?.room_id != null ? 'Update Roles' : 'Join Room'}
          </Button>
        </div>

        {roomState?.room_id == null && (
          <div>
            <Button
              variant="contained"
              disabled={
                roomState?.room_id != null ||
                joinInProgress ||
                streamingStatus !== 'stopped'
              }
              onClick={() => onJoinRoom(true)}>
              {'Create New Room'}
            </Button>
          </div>
        )}
      </Stack>

      <FormGroup>
        {Object.keys(roles).map((role) => {
          return (
            <FormControlLabel
              disabled={streamingStatus !== 'stopped'}
              key={role}
              control={
                <Checkbox
                  checked={roles[role]}
                  onChange={(event: React.ChangeEvent<HTMLInputElement>) => {
                    setRoles((prevRoles) => ({
                      ...prevRoles,
                      [role]: event.target.checked,
                    }));
                  }}
                />
              }
              label={capitalize(role)}
            />
          );
        })}

        {urlParams.enableServerLock && roles['speaker'] === true && (
          <>
            <FormControlLabel
              disabled={streamingStatus !== 'stopped'}
              control={
                <Checkbox
                  checked={lockServer}
                  onChange={(event: React.ChangeEvent<HTMLInputElement>) => {
                    setLockServer(event.target.checked);
                  }}
                />
              }
              label="Lock Server (prevent other users from streaming)"
            />
          </>
        )}
      </FormGroup>

      {urlParams.enableServerLock &&
        roles['speaker'] === true &&
        lockServer && (
          <TextField
            disabled={streamingStatus !== 'stopped'}
            label="Enter Your Name + Expected Lock End Time"
            variant="outlined"
            value={lockServerName}
            onChange={(event: React.ChangeEvent<HTMLInputElement>) => {
              setLockServerName(event.target.value);
            }}
            helperText="Locking the server will prevent anyone else from using it until you close the page, in order to maximize server performance. Please only use this for live demos."
          />
        )}

      {serverState?.serverLock != null &&
        serverState.serverLock.clientID === clientID && (
          <Alert severity="success">{`The server is now locked for your use (${serverState?.serverLock?.name}). Close this window to release the lock so that others may use the server.`}</Alert>
        )}
    </Stack>
  );
}
