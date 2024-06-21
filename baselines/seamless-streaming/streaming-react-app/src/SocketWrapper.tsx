import {useContext, useEffect, useMemo, useRef, useState} from 'react';
import socketIOClient, {Socket} from 'socket.io-client';
import useStable from './useStable';
import {v4 as uuidv4} from 'uuid';
import {SocketContext} from './useSocket';
import {AppResetKeyContext} from './App';
import Backdrop from '@mui/material/Backdrop';
import CircularProgress from '@mui/material/CircularProgress';
import Typography from '@mui/material/Typography';
import {getURLParams} from './URLParams';

// The time to wait before showing a "disconnected" screen upon initial app load
const INITIAL_DISCONNECT_SCREEN_DELAY = 2000;
const SERVER_URL_DEFAULT = `${window.location.protocol === "https:" ? "wss" : "ws"
                    }://${window.location.host}`;
                    
export default function SocketWrapper({children}) {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [connected, setConnected] = useState<boolean | null>(null);
  // Default to true:
  const [willAttemptReconnect] = useState<boolean>(true);
  const serverIDRef = useRef<string | null>(null);

  const setAppResetKey = useContext(AppResetKeyContext);

  /**
   * Previously we had stored the clientID in local storage, but in that case
   * if a user refreshes their page they'll still have the same clientID, and
   * will be put back into the same room, which may be confusing if they're trying
   * to join a new room or reset the app interface. So now clientIDs persist only as
   * long as the react app full lifecycle
   */
  const clientID = useStable<string>(() => {
    const newID = uuidv4();
    // Set the clientID in session storage so if the page reloads the person
    // still retains their member/room config
    return newID;
  });

  const socketObject = useMemo(
    () => ({socket, clientID, connected: connected ?? false}),
    [socket, clientID, connected],
  );

  useEffect(() => {
    const queryParams = {
      clientID: clientID,
    };

    const serverURLFromParams = getURLParams().serverURL;
    const serverURL = serverURLFromParams ?? SERVER_URL_DEFAULT;

    console.log(
      `Opening socket connection to ${
        serverURL?.length === 0 ? 'window.location.host' : serverURL
      } with query params:`,
      queryParams,
    );

    const newSocket: Socket = socketIOClient(serverURL, {
      query: queryParams,
      // Normally socket.io will fallback to http polling, but we basically never
      // want that because that'd mean awful performance. It'd be better for the app
      // to simply break in that case and not connect.
      transports: ['websocket'],
      path: '/ws/socket.io'
    });

    const onServerID = (serverID: string) => {
      console.debug('Received server ID:', serverID);
      if (serverIDRef.current != null) {
        if (serverIDRef.current !== serverID) {
          console.error(
            'Server ID changed. Resetting the app using the app key',
          );
          setAppResetKey(serverID);
        }
      }
      serverIDRef.current = serverID;
    };

    newSocket.on('server_id', onServerID);

    setSocket(newSocket);

    return () => {
      newSocket.off('server_id', onServerID);
      console.log(
        'Closing socket connection in the useEffect cleanup function...',
      );
      newSocket.disconnect();
      setSocket(null);
    };
  }, [clientID, setAppResetKey]);

  useEffect(() => {
    if (socket != null) {
      const onAny = (eventName: string, ...args) => {
        console.debug(`[event: ${eventName}] args:`, ...args);
      };

      socket.onAny(onAny);

      return () => {
        socket.offAny(onAny);
      };
    }
    return () => {};
  }, [socket]);

  useEffect(() => {
    if (socket != null) {
      const onConnect = (...args) => {
        console.debug('Connected to server with args:', ...args);
        setConnected(true);
      };

      const onConnectError = (err) => {
        console.error(`Connection error due to ${err.message}`);
      };

      const onDisconnect = (reason) => {
        setConnected(false);
        console.log(`Disconnected due to ${reason}`);
      };

      socket.on('connect', onConnect);
      socket.on('connect_error', onConnectError);
      socket.on('disconnect', onDisconnect);

      return () => {
        socket.off('connect', onConnect);
        socket.off('connect_error', onConnectError);
        socket.off('disconnect', onDisconnect);
      };
    }
  }, [socket]);

  useEffect(() => {
    if (socket != null) {
      const onReconnectError = (err) => {
        console.log(`Reconnect error due to ${err.message}`);
      };

      socket.io.on('reconnect_error', onReconnectError);

      const onError = (err) => {
        console.log(`General socket error with message ${err.message}`);
      };
      socket.io.on('error', onError);

      const onReconnect = (attempt) => {
        console.log(`Reconnected after ${attempt} attempt(s)`);
      };
      socket.io.on('reconnect', onReconnect);

      const disconnectOnBeforeUnload = () => {
        console.log('Disconnecting due to beforeunload event...');
        socket.disconnect();
        setSocket(null);
      };
      window.addEventListener('beforeunload', disconnectOnBeforeUnload);

      return () => {
        socket.io.off('reconnect_error', onReconnectError);
        socket.io.off('error', onError);
        socket.io.off('reconnect', onReconnect);
        window.removeEventListener('beforeunload', disconnectOnBeforeUnload);
      };
    }
  }, [clientID, setAppResetKey, socket]);

  /**
   * Wait to show the disconnected screen on initial app load
   */
  useEffect(() => {
    window.setTimeout(() => {
      setConnected((prev) => {
        if (prev === null) {
          return false;
        }
        return prev;
      });
    }, INITIAL_DISCONNECT_SCREEN_DELAY);
  }, []);

  return (
    <SocketContext.Provider value={socketObject}>
      {children}

      <Backdrop
        open={connected === false && willAttemptReconnect === true}
        sx={{
          color: '#fff',
          zIndex: (theme) => theme.zIndex.drawer + 1,
        }}>
        <div
          style={{
            alignItems: 'center',
            flexDirection: 'column',
            textAlign: 'center',
          }}>
          <CircularProgress color="inherit" />
          <Typography
            align="center"
            fontSize={{sm: 18, xs: 16}}
            sx={{
              fontFamily:
                'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
              fontWeight: 'bold',
            }}>
            {'Disconnected. Attempting to reconnect...'}
          </Typography>
        </div>
      </Backdrop>
    </SocketContext.Provider>
  );
}
