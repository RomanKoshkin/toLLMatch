import {createContext, useContext} from 'react';
import {Socket} from 'socket.io-client';

type SocketObject = {
  socket: Socket | null;
  clientID: string | null;
  connected: boolean;
};

export const SocketContext = createContext<SocketObject>({
  socket: null,
  clientID: null,
  connected: false,
});

export function useSocket(): SocketObject {
  return useContext(SocketContext);
}
