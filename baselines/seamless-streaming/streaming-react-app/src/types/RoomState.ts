export type MemberID = string;

export type Member = {
  client_id: MemberID;
  session_id: string;
  name: string;
  connection_status: 'connected' | 'disconnected';
};

export type RoomState = {
  activeTranscoders: number;
  room_id: string;
  members: Array<Member>;
  listeners: Array<MemberID>;
  speakers: Array<MemberID>;
};
